# Official Implementation of SGDiff (ACM MM '23)

<a href="https://taited.github.io/sgdiff-project" target="_blank">
  <img src="https://img.shields.io/badge/Project-Page-Green">
</a>
<a href="https://arxiv.org/abs/2308.07605" target="_blank">
  <img src="https://img.shields.io/badge/Paper-Arxiv-red">
</a>

This is the official implementation of SGDiff: A Style Guided Diffusion Model for Fashion Synthesis (ACM MM '23). SGDiff is developed based on the MMagic framework (version V1.1.0). The training scripts and dataset used in this paper will be released soon.

## Todo List
To ensure reproducibility, this project was extensively re-implemented based on MMagic. We anticipate a release date for the training code and dataset in late January or early February.

- [ ] Release the training scripts.
- [ ] Make the dataset publicly available.

## SG-Fashion Dataset Preview
The SG-Fashion Dataset collects 17,000 images of fashion products sourced from e-commerce websites such as ASOS, Uniqlo, and H&M. We set aside 1,700 of these images as the test set. The dataset covers 72 product categories, encompassing a wide range of garment items.
![SG-Fashion](/media/SG-Fashion.jpg "Magic Gardens")

## Installation Guide

To use SGDiff, you need to install a compatible version of PyTorch with CUDA support. We recommend using PyTorch version 1.10 with CUDA 11.1. However, our codebase does not specifically depend on this exact version of PyTorch or CUDA, and other versions may also work but have not been extensively tested. Please refer to the [MMagic installation guide](https://github.com/open-mmlab/mmagic#%EF%B8%8F-installation) for more details on setting up your environment.

1. (Optional if you already have)Install a compatible version of PyTorch with CUDA
   ```bash
   pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```
2. MMagic dependencies
   ```bash
   pip3 install openmim
   mim install mmcv>=2.0.0
   mim install mmengine
   ```
3. Install this repository as editable version
   ```bash
   git clone https://github.com/Taited/sgdiff
   cd sgdiff
   pip3 install -e .
   ```

## Inference Code Now Available 🔥

The inference code for SGDiff is now available in this repository.

Before running inference, download the model checkpoint from the
[Google Drive](https://drive.google.com/drive/folders/1hnXb9PCmhXc7W05qsK69FSQzFdgIDdo9?usp=sharing).

After downloading, you can generate images using the SGDiff model by the following command:

```shell
python inference.py --ckpt sgdiff.pth --img_path examples/starry_night.jpg --prompt "long sleeve jumpsuit"
```
| Prompt                  | sleeveless jumpsuit             | long sleeve jumpsuit             | v-neck jumpsuit                  |
|:-----------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
|                         | ![sleeveless jumpsuit](/media/sleeveless%20jumpsuit.png) | ![long sleeve jumpsuit](/media/long%20sleeve%20jumpsuit.png) | ![V-Neck jumpsuit](/media/V-Neck%20jumpsuit.png)  |


## Citation

If this repository is helpful to your research, please cite it as below.

```bibtex
@inproceedings{10.1145/3581783.3613806,
author = {Sun, Zhengwentai and Zhou, Yanghong and He, Honghong and Mok, P.Y.},
title = {SGDiff: A Style Guided Diffusion Model for Fashion Synthesis},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3613806},
doi = {10.1145/3581783.3613806},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {8433–8442},
numpages = {10},
keywords = {style guidance, denoising diffusion probabilistic models, text-to-image, fashion synthesis},
location = {Ottawa ON, Canada},
series = {MM '23}
}

```

## Acknowledgement

This work builds upon the MMagic library. We appreciate the MMagic team for their substantial contributions to the community. For the exact version of MMagic we used (V1.1.0), please refer to their [repository](https://github.com/open-mmlab/mmagic).

Stay tuned for updates on the release of additional resources!
