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
# model, preprocess = clip.load("ViT-B/32", device=device)

# def img_uint8(image):
#     image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
#     image = image.astype(np.uint8)
#     return image

import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = './'
image_folder = path + '/test/samples_test/'
output_folder = path + '/FID_test/'
# os.makedirs(output_folder, exist_ok=True)

dataset_info = np.load(path + 'dataset_2024-09-06_02-17-51.npy', allow_pickle=True)
print(dataset_info.shape)

# Append the new column
# new_column = np.zeros(len(dataset_info))  # Example: random values, replace with your actual data
# dataset_info = np.column_stack((dataset_info, new_column))


# noise_test = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
noise_test = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for noise in noise_test:
    os.makedirs(f'{output_folder}/FID_{noise}', exist_ok=True)


start_time = time.time()
for i in range(len(dataset_info)):
    image_info = dataset_info[i]
    
    index,image_save_name,mask_save_name,date_time_string,time_cost,prompt,scale,strength,ddim_steps,noise_std = image_info
    
    image_path = image_folder+image_save_name
    mask_path  = image_folder+mask_save_name 

    print(index, image_path)
    
    # image_rgb = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image_i = np.array(Image.open(image_path))
    
    noise_index = np.argmin(abs(noise_std - np.array(noise_test)))
    noise_folder = noise_test[noise_index]
    
    image_save_name = image_save_name[:-4]+'.jpg'
    cv2.imwrite(f'{output_folder}/FID_{noise_folder}/{image_save_name}', image_i)

    # plt.imshow(image_rgb)
    # plt.imshow(mask)
    
    # image_i = Image.open(image_path)
    # image = transform(image_i).unsqueeze(0).to(device)

    # with torch.no_grad():
    #     pred_mask = model(image)[0]
    
    # mask_output = img_uint8(pred_mask[0].squeeze(0).cpu().numpy())
    # cv2.imwrite(f'{output_folder}{mask_save_name}', mask_output)
    # similarities = (image_feature @ text_feature.T).diag().cpu().numpy()
    
    # # Calculate CLIP score
    # clip_score = similarities.mean()
    # print('CLIP Score:', clip_score)
    # CLIP_score_list.append(clip_score)
    # dataset_info[i][-1] = clip_score

    
end_time = time.time()
total_cost = end_time-start_time
iter_speed = total_cost/len(dataset_info)
print(f'Total tine cost: {total_cost}, per iter: {iter_speed}')

for noise in noise_test:
    noise_folder = f'{output_folder}/FID_{noise}'
    file_list = os.listdir(noise_folder)
    file_size = len(file_list)
    print(f'noise: {noise} files: {file_size}')


# valid_data = np.array(valid_data, dtype=object)
save_folder = './CLIP_test/'
# np.save(f'{save_folder}CLIP_data.npy', dataset_info)


# plt.plot(CLIP_score_list)





















