import argparse

import numpy as np
import torch
from mmengine.registry import init_default_scope
from mmengine.runner import set_random_seed
from PIL import Image
from torchvision.utils import save_image

from mmagic.apis import init_model

parser = argparse.ArgumentParser(description='Run inference with SGDiff model')
parser.add_argument(
    '--ckpt',
    type=str,
    default='sgdiff.pth',
    help='Path to the model checkpoint file')
parser.add_argument(
    '--prompt',
    type=str,
    default='sleeveless jumpsuit',
    help='Attribute level description of cloth')
parser.add_argument(
    '--img_path',
    type=str,
    default='examples/starry_night.jpg',
    help='Path to the input image file')
parser.add_argument(
    '--output_path',
    type=str,
    default='results.png',
    help='Path to the output image file')
args = parser.parse_args()

init_default_scope('mmagic')
set_random_seed(100)


def load_img(img_path: str):
    img = Image.open(img_path).resize((256, 256))
    img = np.array(img) / 255.
    img = (img - 0.5) / 0.5
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).cuda()
    return img.to(torch.float32)


if __name__ == '__main__':
    config = 'configs/sgdiff/sgdiff-ddim-sg_fashion-64x64.py'
    model = init_model(config, args.ckpt).cuda().eval()
    prompt = args.prompt
    img = load_img(args.img_path)

    modality_order_cfg = {'txt': 1.5, 'style': 2}

    with torch.no_grad():
        conditions = {
            'style': img,
            'prompt': prompt,
        }
        data = model.infer_mm(
            modality_order_cfg=modality_order_cfg,
            show_progress=True,
            **conditions)
        samples = data['samples']
        save_image(
            samples,
            args.output_path,
            nrow=4,
            normalize=True,
            value_range=(-1, 1))
