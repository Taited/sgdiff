import numpy as np
import torch
from mmengine.registry import init_default_scope
from mmengine.runner import set_random_seed
from PIL import Image
from torchvision.utils import save_image

from mmagic.apis import init_model

init_default_scope('mmagic')
set_random_seed(100)

config = 'configs/sgdiff/sgdiff-ddim-sg_fashion-64x64.py'
ckpt = 'sgdiff.pth'
model = init_model(config, ckpt).cuda().eval()
prompt = 'sleeveless jumpsuit'
image_path = 'starry_night.jpg'
modality_order_cfg = {'txt': 1.5, 'style': 2}


def load_img(img_path: str):
    img = Image.open(img_path).resize((256, 256))
    img = np.array(img) / 255.
    img = (img - 0.5) / 0.5
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).cuda()
    return img.to(torch.float32)


with torch.no_grad():
    img = load_img(image_path)
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
        samples, 'results.png', nrow=4, normalize=True, value_range=(-1, 1))
