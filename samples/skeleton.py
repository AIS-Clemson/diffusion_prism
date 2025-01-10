# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:51:05 2023

@author: MaxGr
"""

import os
import cv2
import numpy as np

from skimage import io, color, morphology


# Input and output directories
input_folder = "samples_mask"
output_folder = "samples_skeleton"

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)




# Function to skeletonize an image using skimage
def sk_skeletonize_image(input_image_path):
    # Read the image using scikit-image
    img = io.imread(input_image_path)
    
    # Convert the image to grayscale
    # img_gray = color.rgb2gray(img)
    _, img_thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)


    # # Perform morphological closing operation
    # kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
    # img_closed = cv2.morphologyEx(img_thresholded, cv2.MORPH_CLOSE, kernel)
    
    # Apply a median filter to reduce noise
    for i in range(5):
        img_median_filtered = cv2.medianBlur(img_thresholded, 5)  # Adjust kernel size as needed

    # Skeletonize the image using skimage
    skeleton = morphology.skeletonize(img_median_filtered)
    
    # Convert the skeletonized image to uint8 format (0 or 255)
    skeleton = skeleton.astype(np.uint8) * 255
    
    return skeleton


# Function to skeletonize an image
def skeletonize_image(input_image):
    # Read the image in grayscale mode
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

    # Threshold the image
    _, img_thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Apply morphological operations for skeletonization
    size = np.size(img_thresholded)
    skel = np.zeros(img_thresholded.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img_thresholded, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_thresholded, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_thresholded = eroded.copy()

        zeros = size - cv2.countNonZero(img_thresholded)
        if zeros == size:
            break

    return skel

# Process images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename)

        # Skeletonize the image
        skeletonized_img = sk_skeletonize_image(input_image_path)

        # Save the skeletonized image
        cv2.imwrite(output_image_path, skeletonized_img)

        print(f"Processed: {input_image_path} -> {output_image_path}")

print("Image skeletonization complete.")
















