# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:13:04 2024

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

# import torch
# import clip
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16

device = "cuda" if torch.cuda.is_available() else "cpu"


import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from scipy import linalg
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    

# Function to load images from a directory and close files properly
def load_images_from_directory(directory, max_images=None):
    images = []
    file_list = os.listdir(directory)
    random.shuffle(file_list)

    for filename in tqdm(file_list, desc=f"Load images from {directory}"):
        if max_images and len(images) >= max_images: break
        if filename.endswith(".jpg") or filename.endswith(".png"):
            with Image.open(os.path.join(directory, filename)) as img:
                images.append(img.convert("RGB"))
                
    return images


# Function to extract features using Inception v3 with tqdm progress bar
def get_inception_features(images, model):
    model.eval()
    model.to(device)
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="Extracting features"):
            img = transform(img).unsqueeze(0).to(device)
            feat = model(img)[0].flatten().cpu().numpy()
            features.append(feat)
    return np.array(features)


# Function to calculate FID score
def calculate_fid(real_features, generated_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid



# Load Inception v3 model
inception_model = models.inception_v3(pretrained=True, transform_input=False)


path = './'
# image_folder = path + '/test/samples_test/'
output_folder = path + '/FID_test/'

# real_images_dir = f'{output_folder}/Micro_Organism'
real_images_dir = f'{output_folder}/EMDS-6'
real_images = load_images_from_directory(real_images_dir)
real_features = get_inception_features(real_images, inception_model)



noise_results = []
# noise_test = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
# noise_test = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
noise_test = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
noise_index = np.array(noise_test)

for noise in noise_test: 
    generated_images_dir = f'{output_folder}/FID_noise/FID_{noise}'
    # generated_images_dir = f'{output_folder}/FID_prism/FID_{noise}'
    print(generated_images_dir)

    generated_images = load_images_from_directory(generated_images_dir)
    generated_features = get_inception_features(generated_images, inception_model)
    
    fid_score = calculate_fid(real_features, generated_features)
    noise_results.append(fid_score)

    print('FID Score:', fid_score)
    
print(noise_results)
    
plt.plot(noise_results)
    
norm_FID = np.array(noise_results)/1500

plt.plot(norm_FID)




# generated_images_dir = f'{output_folder}/controlnet_10_10_0.9'
# print(generated_images_dir)

# generated_images = load_images_from_directory(generated_images_dir)
# generated_features = get_inception_features(generated_images, inception_model)

# fid_score = calculate_fid(real_features, generated_features)
# print('FID Score:', fid_score)










