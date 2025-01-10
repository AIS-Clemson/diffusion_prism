# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:51:29 2024

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import torch
# import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vit_b_16

def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image

# class ViT_Seg(nn.Module):
#     def __init__(self, num_classes=1):
#         super(ViT_Seg, self).__init__()
#         self.vit = vit_b_16(pretrained=True)
        
#         # self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
#         # self.vit.heads = nn.Identity()
#         self.layernorm_output = None
#         self.vit.encoder.ln.register_forward_hook(self.save_layernorm_output)

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28 -> 56x56
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56 -> 112x112
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)  # 112x112 -> 256x256
#         )
        
#     def save_layernorm_output(self, module, input, output):
#         self.layernorm_output = output
    
#     def forward(self, x, bbox=None):
#         _ = self.vit(x)  # activate hook
#         x = self.layernorm_output  # [batch_size, num_patches + 1, embedding_dim]

#         x = x[:, 1:, :] # remove class token
        
#         batch_size, num_patches, embedding_dim = x.size()
#         height = width = int(num_patches ** 0.5)  # Assuming square patches
#         x = x.transpose(1, 2).view(batch_size, embedding_dim, height, width)
        
#         x = self.decoder(x) 
        
#         return x



import segmentation_models_pytorch as smp

aux_params=dict(
    # pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    # activation='sigmoid',      # activation function, default is None
    classes=1,                 # define number of output labels
)

model = smp.Unet(
    # encoder_name="resnet50",        # backbone
    encoder_name="resnet34",        # backbone
    encoder_weights="imagenet",     # load ImageNet as weight
    in_channels=3,                  # input channel
    classes=1,                      # output channel
    # aux_params={"dropout": 0.1}  # Use auxiliary output with 3 classes
    aux_params=aux_params
).to(device)


# Test the model
# x = torch.randn(1, 3, 256, 256).to(device)
# model = UNet().to(device)
# model = ViT_Seg(num_classes=1).to(device)
model.load_state_dict(torch.load('./smp_last.pt'))

transform = transforms.Compose([
    # transforms.ToPILImage(),        # Convert to PIL Image
    transforms.Resize((256, 256)),  # Resize to (256, 256)
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]), # norm
])







import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = './'
image_folder = path + '/test/samples_test/'
output_folder = path + '/test/diffseg_smp_output/'
os.makedirs(output_folder, exist_ok=True)

dataset_info = np.load(path + 'dataset_2024-09-06_02-17-51.npy', allow_pickle=True)
print(dataset_info.shape)

# Append the new column
new_column = np.zeros(len(dataset_info))  # Example: random values, replace with your actual data
dataset_info = np.column_stack((dataset_info, new_column))


SSIM_score_list = []
start_time = time.time()
for i in range(len(dataset_info)):
    image_info = dataset_info[i]
    
    index,image_save_name,mask_save_name,date_time_string,time_cost,prompt,scale,strength,ddim_steps,noise_std,_ = image_info
    
    image_path = image_folder+image_save_name
    mask_path  = image_folder+mask_save_name 

    print(index, image_path)
    
    image_rgb = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # plt.imshow(image_rgb)
    # plt.imshow(mask)
    
    image_i = Image.open(image_path)
    image = transform(image_i).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(image)[0]
    
    mask_output = img_uint8(pred_mask[0].squeeze(0).cpu().numpy())
    cv2.imwrite(f'{output_folder}{mask_save_name}', mask_output)
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


# valid_data = np.array(valid_data, dtype=object)
save_folder = './CLIP_test/'
# np.save(f'{save_folder}CLIP_data.npy', dataset_info)


# plt.plot(CLIP_score_list)





















