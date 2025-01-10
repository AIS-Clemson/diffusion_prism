# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 23:02:58 2023

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"


import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_image_augmentation(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Define the combined transformations
    combined_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((32, 32)),
        transforms.Resize((512, 512)),
        # transforms.GaussianBlur(kernel_size=7),
        # transforms.Lambda(lambda img: img.filter(ImageFilter.MedianFilter(size=5))),
        transforms.RandomApply([transforms.Lambda(lambda img: img.point(lambda x: x + torch.randint(0, 50, (1,))))], p=0.5),
        transforms.ToTensor()
    ])

    # Apply the combined transformation to the image
    augmented_image = combined_transform(image)
    
    img = augmented_image.permute(1, 2, 0).numpy() *255
    
    img = img.astype(np.uint8)

    # Display the augmented image
    plt.imshow(img)
    # plt.title("Augmented Image")
    # plt.show()
    return img

# Call the function with your image path
image_path = "dendrite_sample/eyeQ.png"  # Replace with the actual image path
img = apply_image_augmentation(image_path)

cv2.imwrite('style.png', img)
