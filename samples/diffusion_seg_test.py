# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:50:58 2024

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
# from torchsummary import summary
from torchsummaryX  import summary

import torch
import torch.nn as nn
import torch.nn.functional as F


print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
device = torch.device("cuda")
# print(torch.cuda.get_arch_list())


import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Transforms to be applied to the data
transform = transforms.Compose([
    # transforms.ToPILImage(),        # Convert to PIL Image
    transforms.Resize((512, 512)),  # Resize to (256, 256)
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]), # norm
])

# Transforms to be applied to the data
totensor = transforms.Compose([
    # transforms.ToPILImage(),        # Convert to PIL Image
    transforms.Resize((512, 512)),  # Resize to (256, 256)
    transforms.ToTensor(),           # Convert to tensor
    # transforms.Normalize(mean=[0.5], std=[0.5]), # norm
])




def norm_image(x):
    min_value = np.min(x)
    max_value = np.max(x)
    normalized_array = (x - min_value) / (max_value - min_value) *255
    return normalized_array.astype(np.uint8)


def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image

def display_result(images, masks, outputs):
    images = images[:4]
    masks = masks[:4]
    outputs = outputs[:4]

    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    masks = masks.detach().cpu().permute(0, 2, 3, 1).numpy()
    outputs = outputs.detach().cpu().permute(0, 2, 3, 1).numpy()

    for i in range(4):
        image = images[i, :, :]
        mask = masks[i, :, :, 0]
        output = outputs[i, :, :, 0]
        
        heatmap_rgb = plt.get_cmap('jet')(output)[:, :, :3]  # RGB channels only
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

        image = norm_image(image)
        mask = norm_image(mask)
        output = norm_image(output)
        
        # heatmap_rgb = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        heatmap_rgb = norm_image(heatmap_rgb)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

        combined_mask_output = cv2.hconcat([image, mask, output, heatmap_rgb])

        if i == 0:
            combined_image = combined_mask_output
        else:
            combined_image = cv2.vconcat([combined_image, combined_mask_output])
    
    return combined_image




from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from models import AE, UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AE().to(device)
# model.load_state_dict(torch.load(f'./weights/{model_name}_last.pt'))
# model.load_state_dict(torch.load(f'./weights/smp_last_25_1e4_128.pt'))
model.load_state_dict(torch.load(f'./weights/AE_last.pt'))
model.eval()


# test_path = './FID_test/FID_prism/FID_0.1'
# test_path = './FID_test/FID_noise/FID_0.1'
# test_path = './FID_test/controlnet_10_10_0.9'
test_path = './FID_test/unicontrolnet_output'

# test_path = './FID_test/Micro_Organism'

test_list = os.listdir(test_path)

output_path = './output'
os.makedirs(output_path, exist_ok=True)



mask = Image.open('./FID_test/75.jpg').convert('L')
mask = np.array(mask)

ssim_list = []
with torch.no_grad():
    pbar = tqdm(test_list)
    for image in pbar:
        name = image.split('.')[0]
        # print(name)
        image = Image.open(os.path.join(test_path, image)).convert('RGB')
        image = transform(image)
        image = image.to(device).unsqueeze(0)
        
        pred = model(image)
 
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    
        image = norm_image(image[0])
        pred = norm_image(pred[0])[:,:,0]
        # pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

        pred[pred>127] = 255
        pred[pred<=127] = 0
        
        # mask_name = name.split('_')[0]
        # mask = Image.open(os.path.join('./samples_mask/', f'{mask_name}.jpg')).convert('L')
        # mask = np.array(mask)
        
        ssim_score = ssim(mask, pred, data_range=pred.max() - pred.min())
        ssim_list.append(ssim_score)
        pbar.set_description(f'SSIM: {ssim_score:.2f}')
        
        # combined_image = display_result(images, masks, outputs)
        combined_image = np.hstack((image[:,:,0],mask,pred))
        cv2.imwrite(f'./{output_path}/{name}_combined.png', combined_image)

mean_SSIM = np.mean(ssim_list)
print(f"Mean SSIM: {mean_SSIM}")


# ssim_results = []

# output_folder = './'

# noise_results = []
# # noise_test = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
# # noise_test = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# noise_test = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# noise_index = np.array(noise_test)

# for noise in noise_test: 
#     generated_images_dir = f'./FID_test/FID_noise/FID_{noise}'
#     # generated_images_dir = f'test/samples_noise_{noise}'
#     file_list = os.listdir(generated_images_dir)
#     print(generated_images_dir)
    
#     ssim_list = []
#     with torch.no_grad():
#         pbar = tqdm(file_list)
#         for image in pbar:
#             name = image.split('.')[0]
#             # print(name)
#             image = Image.open(os.path.join(generated_images_dir, image)).convert('RGB')
#             image = transform(image)
#             image = image.to(device).unsqueeze(0)
            
#             pred = model(image)
     
#             image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
#             pred = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
        
#             image = norm_image(image[0])
#             pred = norm_image(pred[0])[:,:,0]
#             # pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    
#             pred[pred>127] = 255
#             pred[pred<=127] = 0
            
#             mask_name = name.split('_')[0]
#             mask = Image.open(os.path.join('./samples_mask/', f'{mask_name}.jpg')).convert('L')
#             mask = np.array(mask)
            
#             ssim_score = ssim(mask, pred, data_range=pred.max() - pred.min())
#             ssim_list.append(ssim_score)
#             pbar.set_description(f'SSIM: {ssim_score:.2f}')
            
#             # combined_image = display_result(images, masks, outputs)
#             combined_image = np.hstack((image[:,:,0],mask,pred))
            
#             # plt.figure()
#             # plt.imshow(combined_image)
#             # plt.show()
#             # cv2.imwrite(f'./{output_path}/{name}.png', image)
#             # cv2.imwrite(f'./{output_path}/{name}_mask.png', pred)
#             # cv2.imwrite(f'./{output_path}/{name}_combined.png', combined_image)
    
#     ssim_results.append(ssim_list)
#     mean_SSIM = np.mean(ssim_list)
#     noise_results.append(mean_SSIM)
#     print(f"Mean SSIM: {mean_SSIM}")


# print("Testing complete. Results saved.")
# print('FID Score:', noise_results)














