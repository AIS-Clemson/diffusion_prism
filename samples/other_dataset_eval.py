# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:56:40 2024

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import torch
import clip
from PIL import Image

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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



image_folder = './test/samples/'

mask_path = '75.jpg'
mask = Image.open(mask_path)
mask = preprocess(mask).unsqueeze(0).to(device)

prompt = "a dendrite sample with rich background"

CLIP_score_list = []
CLIP_similarity_list = []

start_time = time.time()
file_list = os.listdir(image_folder)
for file in file_list:
    image_path = image_folder+file
    print(image_path)
    
    image_i = Image.open(image_path)
    image = preprocess(image_i).unsqueeze(0).to(device)
    
    text = prompt.strip()
    text = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        image_feature = model.encode_image(image)
        text_feature = model.encode_text(text)
        mask_features = model.encode_image(mask)

    similarities = (image_feature @ text_feature.T).diag().cpu().numpy()
    image_similarity = torch.cosine_similarity(image_feature, mask_features).cpu().numpy()

    # Calculate CLIP score
    clip_score = similarities.mean()
    similarity_score = image_similarity.mean()
    print(f'{file} | CLIP Score: {clip_score} | Similarity Score: {similarity_score}')
    
    CLIP_score_list.append(clip_score)
    CLIP_similarity_list.append(similarity_score)
    
end_time = time.time()
print(f'Total tine cost: {end_time-start_time}')

# valid_data = np.array(valid_data, dtype=object)
save_folder = './CLIP_test/'
# np.save(f'{save_folder}CLIP_data.npy', dataset_info)
# plt.plot(CLIP_score_list)
# plt.plot(CLIP_similarity_list)
# plt.scatter(CLIP_similarity_list, CLIP_score_list)

clip_score = np.mean(CLIP_score_list)
similarity_score = np.mean(CLIP_similarity_list)
print(f'CLIP_score: {clip_score} | CLIP_similarity: {similarity_score}')







fid_score_list = []
# Directory paths for real and generated images
real_images_dir = f'./FID_test/Micro_Organism'
generated_images_dir = image_folder

# Load images (specify max_images if needed)
real_images = load_images_from_directory(real_images_dir)
generated_images = load_images_from_directory(generated_images_dir)

# Load Inception v3 model
inception_model = models.inception_v3(pretrained=True, transform_input=False)

# Extract features
real_features = get_inception_features(real_images, inception_model)
generated_features = get_inception_features(generated_images, inception_model)

# Calculate FID
fid_score = calculate_fid(real_features, generated_features)
fid_score_list.append(fid_score)

print('FID Score:', fid_score)





print(clip_score, similarity_score, fid_score)





