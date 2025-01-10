# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:51:29 2024

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = './'
image_folder = path + '/test/samples_test/'

dataset_info = np.load(path + 'dataset_2024-09-06_02-17-51.npy', allow_pickle=True)
print(dataset_info.shape)

# Append the new column
new_column = np.zeros((len(dataset_info),2))  # Example: random values, replace with your actual data
dataset_info = np.column_stack((dataset_info, new_column))


CLIP_score_list = []
CLIP_similarity_list = []

start_time = time.time()
for i in range(len(dataset_info)):
    image_info = dataset_info[i]
    
    index,image_save_name,mask_save_name,date_time_string,time_cost,prompt,scale,strength,ddim_steps,noise_std,_,_ = image_info
    
    image_path = image_folder+image_save_name
    mask_path  = image_folder+mask_save_name 

    print(image_path)
    
    # image_rgb = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(mask_path, cv2.COLOR_GRAY2RGB)
    
    # plt.imshow(image_rgb)
    # plt.imshow(mask)
    
    image_i = Image.open(image_path)
    image = preprocess(image_i).unsqueeze(0).to(device)
    
    mask = Image.open(mask_path)
    mask = preprocess(mask).unsqueeze(0).to(device)

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
    print(f'{index} | CLIP Score: {clip_score} | Similarity Score: {similarity_score}')
    
    CLIP_score_list.append(clip_score)
    CLIP_similarity_list.append(similarity_score)
    
    dataset_info[i][-2] = clip_score
    dataset_info[i][-1] = similarity_score

    
end_time = time.time()
print(f'Total tine cost: {end_time-start_time}')


# valid_data = np.array(valid_data, dtype=object)
save_folder = './CLIP_test/'
# np.save(f'{save_folder}CLIP_data.npy', dataset_info)


plt.plot(CLIP_score_list)
plt.plot(CLIP_similarity_list)

plt.scatter(CLIP_similarity_list, CLIP_score_list)



test_path = './test/samples_clip_test/'
os.makedirs(f'{test_path}', exist_ok=True)

for i in range(len(dataset_info)):
    image_info = dataset_info[i]
    index,image_save_name,mask_save_name,date_time_string,time_cost,prompt,scale,strength,ddim_steps,noise_std,clip_score,similarity_score = image_info

    if similarity_score > 0.8:
        print(image_save_name, clip_score, similarity_score)
        image_path = image_folder+image_save_name
        image_i = Image.open(image_path)
        cv2.imwrite(f'{test_path}/{image_save_name}', np.array(image_i))













