# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:15:09 2024

@author: MaxGr
"""


import os
import cv2
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import argparse, os, sys, glob
import PIL
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext


import time
import copy
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


from pytorch_lightning import seed_everything


from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler




# path = opt.init_img
def load_img(image_name, path, noise_std):
    # style = Image.open('./train/style.png').convert("RGB")
    image = Image.open(path).convert("RGB")
    # image = image.convert("RGB")
    
    # mask_path = 'D:/Data/Flame 2/254p Dataset/254p Thermal Images/'
    # mask_path = './wildfire_images/mask/'
    # mask_file = random.choice(os.listdir(mask_path))
    # mask = Image.open(mask_path+mask_file).convert("L")
    # mask = Image.open(path).convert("L")


    
    
    z = len(image.mode)
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    ratio = w / h
    if w > h:
        w = 512
        h = int(w / ratio)
        if h % 64 != 0:
            h = int((h // 64 + 1) * 64)
    else:
        h = 512
        w = int(h * ratio)
        if w % 64 != 0:
            w = int((w // 64 + 1) * 64)
    print(f"loaded input image from {path}, resize to ({w}, {h}) ")
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # mask = mask.resize((w, h), resample=PIL.Image.LANCZOS)


    # style = np.array(style)
    image = np.array(image)

    mask_out = copy.deepcopy(image)


    # Add noise
    # mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.random.normal(0, noise_std, size=(w,h,3))
    mask = (mask - np.min(mask))/(np.max(mask)-np.min(mask)) *255
    mask = mask.astype(np.uint8)
    

    image = 0.5*np.array(mask) + 0.5*np.array(image)
    image[image>255] = 255
    image = image.astype(np.uint8)

    
    if random.random() > 0.5:
        image = 255-image
        
    # Generate random factors for each layer
    random_factors = np.random.uniform(0, 1, size=3)
    # random_factors = [1, 0.5, 0.25]
    for i in range(3):
        image[:, :, i] = image[:, :, i] * random_factors[i]
    
    # image = copy.deepcopy(mask)

    

    # To tensor
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    
    # mask =  cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    return (2.*image - 1.), mask_out


