# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 20:33:32 2024

@author: MaxGr
"""


import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Folder paths
path = './samples_20k/raw/'
image_path = path+'/sample/'
mask_path  = path+'/mask/' 

save_path = './samples_20k/augmentation/'
save_image = save_path+'/image_new/'
save_mask = save_path+'/mask_new/'

for folder in [save_path, save_image, save_mask]:
    os.makedirs(folder, exist_ok=True)


def combine_images_and_masks(image_paths, mask_paths):
    combined_image = np.zeros((output_rows * image_size, output_cols * image_size, 3), dtype=np.uint8)
    combined_mask = np.zeros((output_rows * image_size, output_cols * image_size), dtype=np.uint8)
        
    ID = ''
    for i in range(output_rows):
        for j in range(output_cols):
            random_index = random.randint(0, len(image_paths) - 1)
            image = cv2.imread(image_paths[random_index])
            image_name = image_paths[random_index].split('/')[-1].split('.')[0]
            # mask_index = [mask_path.split('/')[-1].split('_')[0] for mask_path in mask_paths].index(image_paths[random_index].split('/')[-1].split('_')[0])
            mask = cv2.imread(mask_path+image_name+'_mask.jpg', cv2.IMREAD_GRAYSCALE)
            
            ID = ID + str(random_index) + '_'
            
            row_start = i * image_size
            row_end = (i + 1) * image_size
            col_start = j * image_size
            col_end = (j + 1) * image_size
            
            # Apply the mask to the combined image
            combined_image[row_start:row_end, col_start:col_end] = image
            combined_mask[row_start:row_end, col_start:col_end] = mask
            
    return combined_image, combined_mask, ID


# List all image and mask files
image_paths = [os.path.join(image_path, filename) for filename in os.listdir(image_path)]
mask_paths = [os.path.join(mask_path, filename) for filename in os.listdir(mask_path)]
# Shuffle the image paths
random.shuffle(image_paths)


# Output image dimensions
output_rows = 3
output_cols = 3
image_size = 512

for i in range(5000): 
    print(i)
    # Combine images and masks
    combined_image, combined_mask, ID = combine_images_and_masks(image_paths, mask_paths)
    combined_image = cv2.resize(combined_image, (512,512))
    combined_mask = cv2.resize(combined_mask, (512,512))
    # Save the combined image and mask
    cv2.imwrite(save_image+f'{ID}.jpg', combined_image)
    cv2.imwrite(save_mask+f'{ID}mask.jpg', combined_mask)

    # # Display or save the combined result
    # cv2.imshow('Combined Result', combined_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # plt.figure()
    # plt.imshow(combined_image)
    
    # plt.figure()
    # plt.imshow(combined_mask)





import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import random

def random_crop_segmentation(image, mask, crop_size):
    width, height = image.size
    crop_height = crop_size
    crop_width = crop_size

    # Generate random top-left corner for the crop
    top = random.randint(0, height - crop_height)
    left = random.randint(0, width - crop_width)

    # Crop both image and mask
    cropped_image = image.crop((left, top, left + crop_width, top + crop_height))
    cropped_mask = mask.crop((left, top, left + crop_width, top + crop_height))

    return np.array(cropped_image), np.array(cropped_mask)


# for i in range(10000): 
#     print(i)
#     # random_index = random.randint(0, len(image_paths) - 1)
#     random_index = i
#     image = cv2.imread(image_paths[random_index])
#     image = Image.open(image_paths[random_index])
#     image_name = image_paths[random_index].split('/')[-1].split('.')[0]
#     mask = Image.open(mask_path+image_name+'_mask.jpg').convert('L')
    
#     # Crop images and masks
#     crop_size = random.randint(128, 384)
#     cropped_image, cropped_mask = random_crop_segmentation(image, mask, crop_size)
#     cropped_image = cv2.resize(cropped_image, (512,512))
#     cropped_mask = cv2.resize(cropped_mask, (512,512))
#     # Save the combined image and mask
#     cv2.imwrite(save_image+f'{image_name}_crop_{crop_size}.jpg', cropped_image)
#     cv2.imwrite(save_mask+f'{image_name}_crop_{crop_size}_mask.jpg', cropped_mask)


    
    
    
    
    
    
    
    
    
    
    
    
    
    























