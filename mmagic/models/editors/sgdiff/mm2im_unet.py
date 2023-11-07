import torch
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix

from mmagic.registry import MODELS
from ..glide import Text2ImUNet


@MODELS.register_module()
class MM2ImUNet(Text2ImUNet):

    def __init__(self,
                 pretrained_cfg=None,
                 style_encoder_cfg=None,
                 edge_encoder_cfg=None,
                 fix_glide=True,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        # load pretrained Glide
        self.fix_glide = fix_glide
        if pretrained_cfg is not None:
            self._load_pretrained_and_fix(pretrained_cfg)

        if style_encoder_cfg is not None:
            self.style_encoder = MODELS.build(style_encoder_cfg)
        else:
            self.style_encoder = None

        if edge_encoder_cfg is not None:
            self.edge_encoder = MODELS.build(edge_encoder_cfg)
        else:
            self.edge_encoder = None

    def init_weights(self, *args, **kwargs):
        pass

    def _load_pretrained_and_fix(self, pretrained_cfg: dict):
        prefix = pretrained_cfg.get('prefix', 'unet')
        map_location = pretrained_cfg.get('map_location', 'cpu')
        strict = pretrained_cfg.get('strict', True)
        ckpt_path = pretrained_cfg.get('ckpt_path')

        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)

        # Freeze the parameters
        if self.fix_glide:
            for param in self.parameters():
                param.requires_grad = False

    def get_text_emb(self, tokens, token_mask, **conditions):
        assert tokens is not None

        if self.cache_text_emb and self.cache is not None:
            assert (tokens == self.cache['tokens']).all(
            ), f"Tokens {tokens.cpu().numpy().tolist()} do not match \
            cache {self.cache['tokens'].cpu().numpy().tolist()}"

            return self.cache
        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            assert token_mask is not None
            xf_in = torch.where(token_mask[..., None], xf_in,
                                self.padding_embedding[None])
        xf_out = self.transformer(xf_in.to(self.dtype))
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)

        # embedding style
        if self.style_encoder and 'style' in conditions:
            style = conditions['style']
            xf_out = self.style_encoder(style, text_emb=xf_out)

        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(),
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, **kwargs):
        assert 'tokens' in kwargs and 'token_mask' in kwargs
        hs = []
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                timesteps, dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        if len(timesteps.shape) == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] != x.shape[0]:
            timesteps = timesteps.repeat(x.shape[0])
        ori_emb = self.time_embedding(timesteps)
        if self.xf_width:
            text_outputs = self.get_text_emb(**kwargs)
            xf_proj, xf_out = text_outputs['xf_proj'], text_outputs['xf_out']
            emb = ori_emb + xf_proj.to(ori_emb)
        else:
            xf_out = None
        h = x.type(self.dtype)
        for module in self.in_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)

        #  embedding edge
        if self.edge_encoder:
            # dirty code for testing
            in_channel = next(self.edge_encoder.parameters()).shape[0]
            edge = kwargs['edge'][:, :in_channel]
            hs = self.edge_encoder(edge, hs, time_emb=ori_emb)

        h = self.mid_blocks(h, emb, xf_out)
        for module in self.out_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h
