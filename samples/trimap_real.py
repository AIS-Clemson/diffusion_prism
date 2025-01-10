# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:59:03 2024

@author: MaxGr
"""

import os
import cv2
import numpy as np

from skimage import io, color, morphology


def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image


# Input and output directories
# sample_folder = "samples_sample"
# skeleton_folder = "samples_skeleton"
# trimap_folder = "samples_trimap"
sample_folder = "dendrite_sample"

output_folder = "dendrite_sample_trimap"
# Create the output directory if it doesn't exist
# if not os.path.exists(output_folder):
os.makedirs(output_folder, exist_ok=True)


import utils


# Process images in the input folder
for filename in os.listdir(sample_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        sample_image_path = os.path.join(sample_folder, filename)
        trimap_image_path = os.path.join(output_folder, filename)

        image = cv2.imread(sample_image_path)
        # image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

        coordinates = utils.get_click_coordinates(image)
        point = coordinates[0]
        
        h, w = image.shape[:2]
        trimap = np.zeros((h, w), dtype=np.float32)
        
        # 获取特征像素的颜色
        feature_pixel = image[point[1], point[0]]
        
        # 计算每个像素与特征像素的欧氏距离
        distance_map = np.linalg.norm(image - feature_pixel, axis=2)
        
        # 归一化距离
        normalized_distance = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min())
        
        # plt.imshow(normalized_distance)
        
        # # 生成 trimap
        trimap[normalized_distance > 0.5] = 1  # 前景区域
        trimap[(normalized_distance <= 0.5) & (normalized_distance >= 0.2)] = 0.5  # 未知区域
        trimap[normalized_distance < 0.2] = 0  # 背景区域
        
        trimap = img_uint8(trimap)
        # Save the skeletonized image
        cv2.imwrite(trimap_image_path, trimap)

        print(f"Processed: {trimap_image_path}")

print("Image skeletonization complete.")


















