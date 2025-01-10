# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:52:32 2024

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from PIL import Image
import numpy as np

# Convert an image to grayscale
def convert_to_grayscale(image):
    if len(image.shape) == 3:  # RGB image
        return rgb2gray(image)
    return image  # Already grayscale

# Load a single image and convert to grayscale
def load_image(filepath):
    with Image.open(filepath) as img:
        img = img.convert("L")  # Convert to grayscale
        return np.array(img)

# Calculate SSIM between two images
def calculate_ssim(mask_path, generated_path):
    mask = load_image(mask_path)
    generated = load_image(generated_path)
    score = ssim(mask, generated, data_range=generated.max() - generated.min())
    return score

# Example usage
if __name__ == "__main__":
    # mask_path = "./test/samples_5/75_0000_mask.jpg"  # Path to the mask image
    mask_path = "./test/samples_5/75_0000.png"  # Path to the mask image

    generated_path = "./test/samples_5/75_0006.png"  # Path to the generated image
    # generated_path = "./test/samples_5/75_0002_mask.jpg"  # Path to the generated image


    score = calculate_ssim(mask_path, generated_path)
    print(f"SSIM Score: {score:.4f}")
