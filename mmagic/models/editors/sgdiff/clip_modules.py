# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os
import urllib
import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from mmagic.registry import MODELS


@MODELS.register_module()
class ClipAttnEmbedding(nn.Module):
    MODELS = {
        'ViT-B/32':
        'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'  # noqa
    }

    def __init__(self,
                 name='ViT-B/32',
                 download_root=None,
                 clip_norm=True,
                 cross_attn_cfg=None,
                 residual_cfg=None,
                 pretrained: str = None,
                 last_layer_proj=True):
        super().__init__()
        assert name in self.MODELS, \
            '''SSDiff only support to convert ViT-B/32 instead
            of {}'''.format(name)
        # download pretrained clip model
        model_path = _download(
            self.MODELS[name], download_root
            or os.path.expanduser('~/.cache/clip'))
        with open(model_path, 'rb') as opened_file:
            ckpt = torch.jit.load(opened_file, map_location='cpu').state_dict()
        self.model = self._build_model_from_ckpt(ckpt)

        # set clip image preprocess
        self.clip_norm = clip_norm
        if self.clip_norm:
            self.mean = torch.tensor((0.48145466, 0.4578275, 0.40821073))
            self.std = torch.tensor((0.26862954, 0.26130258, 0.27577711))

        if cross_attn_cfg:
            self.cross_attn = MODELS.build(cross_attn_cfg)
        else:
            self.cross_attn = None

        if residual_cfg is not None:
            self.learned_length = residual_cfg.get('learned_length', None)
            self.learned_width = residual_cfg.get('learned_width', None)
            zero_init = residual_cfg.get('zero_init', False)
            if self.learned_length is not None:
                self.skip_module = nn.Linear(self.learned_length,
                                             self.learned_length)
            elif self.learned_width:
                self.skip_module = nn.Linear(self.learned_width,
                                             self.learned_width)
            else:
                self.skip_module = nn.Identity()
            if zero_init:
                zero_conv_wrapper(self.skip_module)
        else:
            self.skip_module = None

        if pretrained:
            state_dict = {}
            print('Load pretrained style encoder at: ' + pretrained)
            loaded_state = torch.load(pretrained)['state_dict']
            for state_key in loaded_state:
                if state_key.find('unet.style_encoder.') != -1:
                    state_dict[state_key.replace('unet.style_encoder.',
                                                 '')] = loaded_state[state_key]
            self.load_state_dict(state_dict)

        self.last_layer_proj = last_layer_proj
        if not self.last_layer_proj:
            del self.model.proj

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img, text_emb):
        if self.clip_norm:
            img = (img + 1) / 2
            img = F.batch_norm(img, self.mean.to(self.device),
                               self.std.to(self.device))
            img = F.interpolate(img, (224, 224))
        image_features = self.model(img)
        if self.last_layer_proj:
            image_features = torch.einsum('bld,ds->bls', image_features,
                                          self.model.proj)
        if self.cross_attn is not None:
            emb_features = self.cross_attn(
                text_emb.permute(0, 2, 1),
                image_features.permute(0, 2, 1)).permute(0, 2, 1)
        if self.skip_module is not None:
            if self.learned_length:
                residual = self.skip_module(emb_features.permute(0, 2, 1))
                residual = residual.permute(0, 2, 1)
            else:
                residual = self.skip_module(emb_features)
            text_emb += residual
        return text_emb

    def _build_model_from_ckpt(self, state_dict, mae_training=None):
        # parse input arguments
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')
        ])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round(
            (state_dict['visual.positional_embedding'].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size
        embed_dim = state_dict['text_projection'].shape[1]
        vision_heads = vision_width // 64

        # instantiate vison transformer
        model = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            mae_training=mae_training)

        visual_state_dict = {}
        for key in state_dict:
            if key.find('visual.') != -1:
                visual_state_dict[key.replace('visual.', '')] = state_dict[key]

        model.load_state_dict(visual_state_dict)
        return model


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split('/')[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f'{download_target} exists and is not a regular file')

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target,
                               'rb').read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f'{download_target} exists, but the SHA256 checksum does not \
                    match; re-downloading the file')

    with urllib.request.urlopen(url) as source, open(download_target,
                                                     'wb') as output:
        with tqdm(
                total=int(source.info().get('Content-Length')),
                ncols=80,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target,
                           'rb').read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            'Model has been downloaded but the SHA256 checksum does \
                not not match')

    return download_target


def zero_conv_wrapper(layers):
    for param in layers.parameters():
        param.data = torch.nn.Parameter(
            torch.zeros_like(param.data), requires_grad=param.requires_grad)


class VisionTransformer(nn.Module):

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 mae_training: dict = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.mae_training = mae_training
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # the original clip only project the 0th token
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        # in ssdiff, we project all image tokens
        x_normed = self.ln_post(x)
        # x = torch.einsum('bld,ds->bls', x_normed, self.proj)

        return x_normed

    def get_unmasked_patches(self, x: torch.Tensor):
        # 90% of the maximum sum of a completely masked patch
        patch_threshold = self.patch_size**2 * 3 * -255 * 0.9
        patches = x.unfold(2, self.patch_size,
                           self.patch_size).unfold(3, self.patch_size,
                                                   self.patch_size)
        patches = patches.reshape(
            patches.size(0), patches.size(1),
            patches.size(2) * patches.size(3), -1)
        patch_sums = patches.sum(dim=-1)
        unmasked_indices = (patch_sums > patch_threshold).nonzero(
            as_tuple=True)[2]
        return unmasked_indices


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock
        # (width, heads, attn_mask) for _ in range(layers)])
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        h = []
        for blocks in self.resblocks:
            x = blocks(x)
            h.append(x.permute(1, 0, 2))
        return x, h
