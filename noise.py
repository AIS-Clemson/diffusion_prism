# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:03:09 2023

@author: MaxGr
"""

import os
import cv2
import numpy as np

# Input image path
image_path = 'Picture1.png'

# Output folder path
output_folder = './noise/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the input image
input_image = cv2.imread(image_path)
input_image = cv2.resize(input_image, (512,512))

# Generate noise and add it to the image
for strength in np.arange(0.1, 100, 1):
    noisy_image = cv2.add(input_image, np.random.normal(scale=strength, size=input_image.shape).astype(np.uint8))

    # Save the noisy image
    output_path = os.path.join(output_folder, f'noisy_image_{strength:.1f}.jpg')
    cv2.imwrite(output_path, noisy_image)

print("Noisy images with varying strengths saved successfully.")






# # Add noise
# mask = np.array(mask).astype(np.float32) / 255.0
# mask = mask + np.random.normal(0, noise_std, size=(w,h,3))
# mask = (mask - np.min(mask))/(np.max(mask)-np.min(mask)) *255
# # mask = mask*255
# mask = mask.astype(np.uint8)
