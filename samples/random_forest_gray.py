# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:33:50 2024

@author: MaxGr
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Load image and convert to grayscale
def load_image(filepath):
    with Image.open(filepath) as img:
        gray_img = img.convert("L")  # Convert to grayscale
        return np.array(gray_img)

# Prepare training data (single channel)
def prepare_training_data(mask, generated_image):
    h, w = mask.shape
    mask_flat = mask.flatten()
    image_flat = generated_image.flatten()  # Flatten grayscale image

    # Extract labeled pixels
    X = image_flat[mask_flat > 0].reshape(-1, 1)  # Foreground pixels
    y = mask_flat[mask_flat > 0]  # Foreground labels

    # Sample background pixels
    bg_X = image_flat[mask_flat == 0].reshape(-1, 1)
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
    h, w = generated_image.shape
    image_flat = generated_image.flatten().reshape(-1, 1)  # Flatten grayscale
    predictions = rf.predict(image_flat)
    segmented = predictions.reshape(h, w)
    return segmented

# Main script
if __name__ == "__main__":
    mask_path = "./test/samples_5/75_0000_mask.jpg"  # Path to the mask image
    # mask_path = "./test/samples_5/75_0000.png"  # Path to the mask image

    generated_path = "./test/samples_5/75_0004.png"  # Path to the generated image
    # generated_path = "./test/samples_5/75_0002_mask.jpg"  # Path to the generated image

    # Load mask and generated image
    mask = load_image(mask_path)
    generated_image = load_image(generated_path)

    # Ensure mask is binary
    mask = (mask > 128).astype(int)

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

    # Visualize results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Grayscale Image")
    plt.imshow(generated_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")

    plt.show()
