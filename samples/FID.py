# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:51:39 2024

@author: MaxGr
"""

import os
import numpy as np
import random
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from scipy import linalg
from PIL import Image
from tqdm import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    
# Define data augmentation transforms
data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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



# Directory paths for real and generated images
real_images_dir = './wildfire_images/FLAME 2/fire_smoke'

# generated_images_dir = './wildfire_images/FLAME 2/no_fire'
# generated_images_dir = './test_data/valid_images_all/valid_images_1'
# generated_images_dir = './test_data/flame_mask_1'
generated_images_dir = './test_data/std_0.5'


# Load images (specify max_images if needed)
real_images = load_images_from_directory(real_images_dir, max_images=9800)
generated_images = load_images_from_directory(generated_images_dir, max_images=9800)


# Load Inception v3 model
inception_model = models.inception_v3(pretrained=True, transform_input=False)

# Extract features
real_features = get_inception_features(real_images, inception_model)
generated_features = get_inception_features(generated_images, inception_model)

# Calculate FID
fid_score = calculate_fid(real_features, generated_features)
print('FID Score:', fid_score)








