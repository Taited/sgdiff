import mmengine
import torch
from mmengine.model import is_model_wrapper
from mmengine.optim import OptimWrapperDict
from tqdm import tqdm

from mmagic.registry import MODELS
from ..glide.glide import Glide


@MODELS.register_module()
class SGDiff(Glide):
    MODALITIES = {
        'txt': ['tokens', 'token_mask'],
        'style': ['style'],
        'edge': ['edge']
    }

    def __init__(self,
                 modalities: list = ['txt', 'style'],
                 cond_prob=0.2,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert set(modalities) <= set(
            self.MODALITIES), 'Only support {}, yet you provide {}'.format(
                self.MODALITIES.keys(), modalities)
        self.tokenizer = self.unet.tokenizer
        self.modalities = modalities
        if isinstance(cond_prob, dict):
            assert set(cond_prob.keys()) <= set(modalities)

    @torch.no_grad()
    def infer(self,
              init_image=None,
              batch_size=1,
              guidance_scale=3.,
              num_inference_steps=50,
              show_progress=False,
              modalities: list = None,
              **conditions):
        """_summary_

        Args:
            init_image (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 1.
            num_inference_steps (int, optional): _description_.
                Defaults to 1000.
            labels (_type_, optional): _description_. Defaults to None.
            show_progress (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert num_inference_steps > 0
        if is_model_wrapper(self.unet):
            model = self.unet.module
        else:
            model = self.unet

        # Sample gaussian noise to begin loop
        if init_image is None:
            image = torch.randn(
                (2 * batch_size, 3, model.image_size, model.image_size))
            image = image.to(self.device)
        else:
            image = init_image

        # set to evaluation
        model.eval()
        ori_time_steps = len(self.diffusion_scheduler.timesteps)
        self.diffusion_scheduler.set_timesteps(num_inference_steps)

        timesteps = self.diffusion_scheduler.timesteps

        # text embedding
        if ('tokens' not in conditions
                or 'token_mask' not in conditions) and 'prompt' in conditions:
            prompt = conditions.pop('prompt')
            tokens = self.tokenizer.encode(prompt)
            tokens, token_mask = self.tokenizer.padded_tokens_and_mask(
                tokens, 128)
            tokens = torch.tensor([tokens] * batch_size, device=self.device)
            token_mask = torch.tensor(
                [token_mask] * batch_size, device=self.device)
            conditions['tokens'], conditions['token_mask'] = tokens, token_mask

        # prepare unconditions for classifier-free guidance
        conditions = self.set_uncond(
            conditions, batch_size, modalities=modalities)

        if show_progress and mmengine.dist.is_main_process():
            timesteps = tqdm(timesteps)

        for t in timesteps:
            # 1. predicted model_output
            half = image[:len(image) // 2]
            combined = torch.cat([half, half], dim=0)
            model_output = model(combined, t, **conditions)
            eps, rest = model_output[:, :3], model_output[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            noise_pred = torch.cat([eps, rest], dim=1)

            # 2. compute previous image: x_t -> x_t-1
            diffusion_scheduler_output = self.diffusion_scheduler.step(
                noise_pred, t, image)

            # 3. applying classifier guide
            image = diffusion_scheduler_output['prev_sample']

        # abandon unconditon
        image = image[:batch_size]
        tokens = conditions['tokens'][:batch_size]
        token_mask = conditions['token_mask'][:batch_size]
        if init_image is not None:
            init_image = init_image[:batch_size]

        # upsample image
        if self.unet_up:
            image = self.infer_up(
                low_res_img=image,
                batch_size=batch_size,
                tokens=tokens,
                mask=token_mask,
                show_progress=show_progress)

        # set back to train
        model.train()
        self.diffusion_scheduler.set_timesteps(ori_time_steps)
        return {'samples': image}

    @torch.no_grad()
    def infer_mm(self,
                 init_image=None,
                 batch_size=1,
                 modality_order_cfg: dict = {
                     'txt': 3.,
                     'style': 1.
                 },
                 num_inference_steps=50,
                 up_inference_steps=35,
                 show_progress=False,
                 **conditions):
        """inference for multi-modality.

        Args:
            init_image (_type_, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 1.
            modality_order_cfg (dict): modality configs.
            num_inference_steps (int, optional): _description_.
                Defaults to 1000.
            show_progress (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert num_inference_steps > 0
        assert set(modality_order_cfg) <= set(self.MODALITIES), \
            'SSDiff only support {}, yet you provide {}'.format(
                self.MODALITIES.keys(), modality_order_cfg.keys())

        # SSDiff will forward on modality condition and one uncondition
        forward_times = len(modality_order_cfg) + 1
        guidance_scale = [
            modality_order_cfg[modality] for modality in modality_order_cfg
        ]
        # when calculating the weighted eps, it is reversed
        guidance_scale = list(reversed(guidance_scale))
        guidance_scale = torch.tensor(
            guidance_scale, dtype=self.dtype, device=self.device)

        if is_model_wrapper(self.unet):
            model = self.unet.module
        else:
            model = self.unet

        # Sample gaussian noise to begin loop
        if init_image is None:
            image = torch.randn((forward_times * batch_size, 3,
                                 model.image_size, model.image_size))
            image = image.to(self.device)
        else:
            image = init_image

        # set to evaluation
        model.eval()
        ori_time_steps = len(self.diffusion_scheduler.timesteps)
        self.diffusion_scheduler.set_timesteps(num_inference_steps)

        timesteps = self.diffusion_scheduler.timesteps

        # text embedding
        if ('tokens' not in conditions
                or 'token_mask' not in conditions) and 'prompt' in conditions:
            prompt = conditions.pop('prompt')
            tokens = self.tokenizer.encode(prompt)
            tokens, token_mask = self.tokenizer.padded_tokens_and_mask(
                tokens, 128)
            tokens = torch.tensor([tokens] * batch_size, device=self.device)
            token_mask = torch.tensor(
                [token_mask] * batch_size, device=self.device)
            conditions['tokens'], conditions['token_mask'] = tokens, token_mask

        # batch size check for inference
        for single_condition in conditions:
            if conditions[single_condition].shape[0] != batch_size:
                repeat_cfg = [batch_size] + [
                    1 for _ in range(
                        len(conditions[single_condition].shape) - 1)
                ]
                conditions[single_condition] = conditions[
                    single_condition].repeat(repeat_cfg)

        # prepare unconditions for classifier-free guidance
        conditions = self.set_uncond_mm(conditions, batch_size,
                                        modality_order_cfg)

        if show_progress and mmengine.dist.is_main_process():
            timesteps = tqdm(timesteps)

        for t in timesteps:
            # 1. predicted model_output
            half = image[:batch_size]
            combined = torch.cat([half for _ in range(forward_times)], dim=0)
            model_output = model(combined, t, **conditions)
            eps, rest = model_output[:, :3], model_output[:, 3:]

            # multi-modality classifier-free inference
            _, c, w, h = eps.shape
            eps_reshaped = eps.reshape(forward_times, batch_size, c, w, h)
            uncond_eps = eps_reshaped[-1:]
            cond_eps = eps_reshaped[:-1] - eps_reshaped[1:]
            weighted_cond_eps = torch.einsum('m,mbcwh->bcwh', guidance_scale,
                                             cond_eps).unsqueeze(0)
            half_eps = uncond_eps + weighted_cond_eps
            eps = torch.cat([half_eps for _ in range(forward_times)], dim=0)
            eps = eps.reshape(batch_size * forward_times, c, w, h)
            noise_pred = torch.cat([eps, rest], dim=1)

            # 2. compute previous image: x_t -> x_t-1
            diffusion_scheduler_output = self.diffusion_scheduler.step(
                noise_pred, t, image)

            # 3. applying classifier guide
            image = diffusion_scheduler_output['prev_sample']

        # abandon unconditon
        image = image[:batch_size]
        tokens = conditions['tokens'][:batch_size]
        token_mask = conditions['token_mask'][:batch_size]
        if init_image is not None:
            init_image = init_image[:batch_size]

        # upsample image
        if self.unet_up:
            image = self.infer_up(
                low_res_img=image,
                batch_size=batch_size,
                tokens=tokens,
                mask=token_mask,
                num_inference_steps=up_inference_steps,
                show_progress=show_progress)

        # set back to train cfg
        model.train()
        self.diffusion_scheduler.set_timesteps(ori_time_steps)
        return {'samples': image}

    def train_step(self, data: dict, optim_wrapper: OptimWrapperDict):
        raise NotImplementedError

    def val_step(self, data: dict, show_progress=False):
        raise NotImplementedError

    def test_step(self, data: dict):
        return self.val_step(data, show_progress=True)

    def set_uncond(self, cond_dict: dict, batch_size, modalities: list = None):
        if modalities is None:
            modalities = self.modalities
        uncond_flag = {}
        for modality in modalities:
            uncond_flag[modality] = True
        uncond_dict = self.get_uncond(
            cond_dict, batch_size, uncond_flag, modalities=modalities)

        for modality in modalities:
            for cond in self.MODALITIES[modality]:
                cond_dict[cond] = torch.cat(
                    (cond_dict[cond], uncond_dict['uncond_' + cond]))

        return cond_dict

    def set_uncond_mm(self, cond_dict: dict, batch_size: int,
                      modality_order_cfg: dict):
        uncond_flag = {}
        for modality in self.modalities:
            uncond_flag[modality] = True
        uncond_dict = self.get_uncond(
            cond_dict, batch_size, uncond_flag, prefix='')
        modality_order = list(modality_order_cfg.keys())

        collected_cond = {}
        for forward_id in range(len(modality_order)):
            modality = modality_order[forward_id]

            # set condition flag for current modality
            cond_flag = []
            for _ in range(len(modality_order) - forward_id, 0, -1):
                cond_flag.append(True)
            # repeat uncondition
            for _ in range(forward_id + 1):
                cond_flag.append(False)

            # there is special operation for text condition
            if modality == 'txt':
                collected_cond['tokens'] = self._collect_cond_(
                    'tokens', cond_dict, uncond_dict, cond_flag)
                collected_cond['token_mask'] = self._collect_cond_(
                    'token_mask', cond_dict, uncond_dict, cond_flag)
            else:
                collected_cond[modality] = self._collect_cond_(
                    modality, cond_dict, uncond_dict, cond_flag)

        return collected_cond

    def _collect_cond_(self, modality: str, cond_dict: dict, uncond_dict: dict,
                       cond_flags: list):
        condition = []
        for flag in cond_flags:
            if flag:
                condition.append(cond_dict[modality])
            else:
                condition.append(uncond_dict[modality])
        condition = torch.cat(condition, dim=0)
        return condition

    def get_uncond(self,
                   cond_dict: dict,
                   batch_size,
                   uncond_flag: dict,
                   modalities: list = None,
                   prefix: str = 'uncond_'):
        uncond_dict = {}
        if modalities is None:
            modalities = self.modalities

        for modality in modalities:
            if modality == 'txt' and uncond_flag[modality]:
                uncond_tokens, uncond_mask = \
                    self.tokenizer.padded_tokens_and_mask(
                        [], 128)
                uncond_dict[prefix + 'tokens'] = torch.tensor(
                    [uncond_tokens] * batch_size, device=self.device)
                uncond_dict[prefix + 'token_mask'] = torch.tensor(
                    [uncond_mask] * batch_size, device=self.device)
            elif modality == 'edge' and uncond_flag[modality]:
                cond_item = cond_dict[modality]
                uncond_item = torch.zeros_like(cond_item)
                uncond_dict[prefix + modality] = uncond_item
            elif uncond_flag[modality]:
                for cond_key in self.MODALITIES[modality]:
                    cond_item = cond_dict[cond_key]
                    uncond_item = -3 * torch.ones_like(cond_item)
                    uncond_dict[prefix + cond_key] = uncond_item

        return uncond_dict

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
