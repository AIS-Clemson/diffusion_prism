# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:52:32 2024

@author: MaxGr
"""

import os
import cv2
import numpy as np

from skimage import io, color, morphology


# Input and output directories
mask_folder = "samples_mask"
skeleton_folder = "samples_skeleton"
trimap_folder = "samples_trimap"

# Create the output directory if it doesn't exist
# if not os.path.exists(output_folder):
os.makedirs(trimap_folder, exist_ok=True)


import cv2
import numpy as np

def generate_trimap(mask, kernel_size=3, erosion_iterations=1, dilation_iterations=3):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    foreground = mask.copy()
    
    background = cv2.erode(mask, kernel, iterations=erosion_iterations)
    
    unknown = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    gray_area = np.full(mask.shape, 128, dtype=np.uint8)
    
    # gray_area[background > 200] = 255
    
    gray_area[unknown < 50] = 0

    # trimap[background == 0] = 0
    
    return gray_area



# Process images in the input folder
for filename in os.listdir(mask_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        mask_image_path = os.path.join(mask_folder, filename)
        skeleton_image_path = os.path.join(skeleton_folder, filename)
        trimap_image_path = os.path.join(trimap_folder, filename)

        mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        skeleton = cv2.imread(skeleton_image_path, cv2.IMREAD_GRAYSCALE)

        gray_area = generate_trimap(mask)
        trimap = cv2.add(gray_area, skeleton)
                
        # Save the skeletonized image
        cv2.imwrite(trimap_image_path, trimap)

        print(f"Processed: {trimap_image_path}")

print("Image skeletonization complete.")
















