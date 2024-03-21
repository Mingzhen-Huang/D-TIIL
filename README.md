# D-TIIL
This repository contains the code and data for the following paper:

[Exposing Text-Image Inconsistency Using Diffusion Models](https://openreview.net/forum?id=Ny150AblPu) (ICLR 2024)

# Dataset
We are still cleaning the dataset for public use, it will be released in the next few weeks

# Installation
We tested on the environment of torch 1.13.1 with a cuda version of 11.7
```
pip install -r requirements.txt
```

# Quick Start
Please check the provided jupyter notebook for details, or you can easily run the model using following code:
```
import torch
from PIL import Image
from pipeline import DTIILPipeline
im = Image.open('./asset/exampe.jpg').resize((512,512)).convert("RGB")git a
model_id = "runwayml/stable-diffusion-v1-5"
pipe = DTIILPipeline.from_pretrained(model_id, safety_checker=None)
mask = pipe(prompt, im)['final_mask']
```


# Citation
If you find our code or dataset useful, please cite:
```
@inproceedings{
huang2024exposing,
title={{EXPOSING} {TEXT}-{IMAGE} {INCONSISTENCY} {USING} {DIFFUSION} {MODELS}},
author={Mingzhen Huang and Shan Jia and Zhou Zhou and Yan Ju and Jialing Cai and Siwei Lyu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=Ny150AblPu}
}
```