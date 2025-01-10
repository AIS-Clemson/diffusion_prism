# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:02:48 2023

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

# import sys
# from utils import *

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# from transformers import CLIPProcessor, CLIPModel
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = './color_test/'

# dataset_info = np.load(path + 'dataset.npy', allow_pickle=True)
# dataset_info = np.load('dataset.npy', allow_pickle=True)

# image_folder = path + '/flame_mask_1/'
# mask_folder  = path + '/mask/'

# CLASSES = ["fire"]
# CLASSES = ["fire","smoke", "tree", "rock", "people", "building", "car", "cloud", "snow"]

prompt = "a dendrite sample with rich background"

image_list = os.listdir(path)

image_CLIP = []
valid_images = []
valid_data = []
rgb_values = []

stat = []

start_time = time.time()
for i in range(len(image_list)):
    name = image_list[i]
    if 'mask.jpg' in name.split('_'): continue
    
    image_i = Image.open(path+name)
    
    image = preprocess(image_i).unsqueeze(0).to(device)
    text = clip.tokenize(prompt).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    
    class_id = probs[0].argmax()
    confidence = probs[0][class_id]

    classification_results.append(f"{CLASSES[class_id]},{confidence:0.2f}")
    
    image_CLIP.append([CLASSES[class_id], confidence])
    
    if CLASSES[class_id] == 'fire':
        valid_images.append(image_name)
        valid_info = image_info.tolist()
        valid_info.extend([CLASSES[class_id], confidence, rgb])
        valid_data.append(valid_info)
        rgb_values.append(rgb)

        cv2.imwrite(f'./{path}/valid_images/{image_name}', image_bgr)
        
    temp = image_info.tolist()
    temp.extend([CLASSES[class_id], confidence, rgb])
    stat.append(temp)
    
        
end_time = time.time()
print(f'Total tine cost: {end_time-start_time}')


valid_data = np.array(valid_data, dtype=object)

np.save('./test_data/valid_data.npy',np.array(valid_data, dtype=object))





























