# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:02:20 2024

@author: MaxGr
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.metrics import structural_similarity as ssim

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Load images
def load_image(filepath):
    with Image.open(filepath) as img:
        return np.array(img)

# Prepare training data
def prepare_training_data(mask, generated_image):
    h, w = mask.shape
    mask_flat = mask.flatten()
    image_flat = generated_image.reshape(-1, 3)  # Flatten RGB image
    # image_flat = generated_image.flatten()

    # Extract labeled pixels
    X = image_flat[mask_flat > 0]  # Pixels labeled as 1 in mask
    y = mask_flat[mask_flat > 0]  # Corresponding labels (1 for foreground)

    # Sample background pixels
    bg_X = image_flat[mask_flat == 0]
    bg_y = np.zeros(bg_X.shape[0])

    # Combine foreground and background
    X = np.vstack((X, bg_X))
    y = np.hstack((y, bg_y))

    return X, y

# Train Random Forest
def train_random_forest(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

# Segment the image using trained Random Forest
def segment_image(rf, generated_image):
    h, w, c = generated_image.shape
    
    image_flat = generated_image.reshape(-1, 3)
    # image_flat = generated_image.flatten()

    predictions = rf.predict(image_flat)
    segmented = predictions.reshape(h, w)
    return segmented

# Calculate SSIM between two images
def calculate_ssim(mask, generated):
    score = ssim(mask, generated, data_range=generated.max() - generated.min())
    return score

# Main script
if __name__ == "__main__":
    # mask_path = "./test/samples_5/75_0000.png"  # Path to the mask image
    mask_path = "./75.jpg"  # Path to the generated image

    # generated_path = "./test/samples_5/75_0004.png"  # Path to the generated image
    # generated_path = "./test/samples_5/75_0002_mask.jpg"  # Path to the generated image
    # generated_path = "./FID_test/unicontrolnet_output/234_2.png"  # Path to the generated image
    # generated_path = "./FID_test/controlnet_10_10_0.9/01020-2602292205.png"  # Path to the generated image
    # generated_path = "./FID_test/SD_0.7/01020-2602292205.png"  # Path to the generated image
    # generated_path = "./color_test/75_0710.png"  # Path to the generated image
    generated_path = "./FID_test/00686-3480162588.png"  # Path to the generated image

    # mask_path = "./example_mask.png"  # Path to mask image (binary)
    # generated_path = "./example_generated.png"  # Path to generated image (RGB)

    mask = load_image(mask_path)
    generated_image = load_image(generated_path)

    # Ensure mask is binary
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

    # mask = (mask > 128).astype(int)

    # Prepare training data
    X, y = prepare_training_data(mask, generated_image)

    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = train_random_forest(X_train, y_train)

    # Evaluate on test set
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Test Accuracy: {accuracy:.4f}")

    # Segment the image
    segmented_image = segment_image(rf, generated_image)
    
    ssim_score = calculate_ssim(mask, segmented_image)
    print(f"SSIM: {ssim_score:.4f}")


    # Visualize results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(generated_image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmented Image")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")
    
    cv2.imwrite('./exp/SSIM/generated_image.png', cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite('./exp/SSIM/mask.png', mask)
    cv2.imwrite('./exp/SSIM/segmented_image.png', segmented_image)

    plt.show()







