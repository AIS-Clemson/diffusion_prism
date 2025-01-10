# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:53:43 2024

@author: MaxGr
"""

import os
import random
import numpy as np

from scipy.stats import entropy
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader



transform = Compose([
    Resize((299, 299)),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def load_images_from_directory(directory, max_images=None):
    images = []
    file_list = os.listdir(directory)
    random.shuffle(file_list)

    for filename in tqdm(file_list, desc=f"Loading images from {directory}"):
        if max_images and len(images) >= max_images:
            break
        if filename.endswith(".jpg") or filename.endswith(".png"):
            with Image.open(os.path.join(directory, filename)) as img:
                images.append(transform(img.convert("RGB")))
    return images

def load_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    model.to(device)
    return model

def get_inception_probabilities(images, model, batch_size=32):
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=False)
    probabilities = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating probabilities"):
            batch = batch.to(device)
            preds = F.softmax(model(batch), dim=1)
            probabilities.append(preds.cpu().numpy())

    return np.concatenate(probabilities, axis=0)

def calculate_inception_score(probabilities, splits=10):
    split_scores = []

    for i in range(splits):
        part = probabilities[i * (len(probabilities) // splits):(i + 1) * (len(probabilities) // splits)]
        p_y = np.mean(part, axis=0)
        scores = [entropy(p_yx, p_y) for p_yx in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# if __name__ == "__main__":
    
    
device = "cuda" if torch.cuda.is_available() else "cpu"

inception_model = load_inception_model()

# Load Inception v3 model
# inception_model = models.inception_v3(pretrained=True, transform_input=False)


path = './'
# # image_folder = path + '/test/samples_test/'
output_folder = path + '/FID_test/'

# real_images_dir = f'{output_folder}/Micro_Organism'
real_images_dir = f'{output_folder}/EMDS-6'

real_images = load_images_from_directory(real_images_dir)
real_features = get_inception_probabilities(real_images, inception_model)
is_mean, is_std = calculate_inception_score(real_features)
print(f"Inception Score: {is_mean} ± {is_std}")




prism_results = []
# noise_test = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
# noise_test = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
noise_list = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
noise_index = np.array(noise_list)

for noise in noise_list: 
    cat = 'FID_prism'

    generated_images_dir = f'{output_folder}/{cat}/FID_{noise}'
    generated_images = load_images_from_directory(generated_images_dir)    
    probabilities = get_inception_probabilities(generated_images, inception_model)
    is_mean, is_std = calculate_inception_score(probabilities)

    IS_score = is_mean
    prism_results.append(IS_score)
    print(generated_images_dir)
    print(f"Inception Score: {is_mean} ± {is_std}")
    
    
    
noise_results = []
for noise in noise_list: 
    cat = 'FID_noise'

    generated_images_dir = f'{output_folder}/{cat}/FID_{noise}'
    generated_images = load_images_from_directory(generated_images_dir)    
    probabilities = get_inception_probabilities(generated_images, inception_model)
    is_mean, is_std = calculate_inception_score(probabilities)

    IS_score = is_mean
    noise_results.append(IS_score)
    print(generated_images_dir)
    print(f"Inception Score: {is_mean} ± {is_std}")
    
    
        
        

    
# generated_images_dir = "./generated_images/"


controlnet_images_dir = f'{output_folder}/controlnet_10_10_0.9'
controlnet_images = load_images_from_directory(controlnet_images_dir)
controlnet_features = get_inception_probabilities(controlnet_images, inception_model)
is_mean, is_std = calculate_inception_score(controlnet_features)
print(f"Inception Score: {is_mean} ± {is_std}")













