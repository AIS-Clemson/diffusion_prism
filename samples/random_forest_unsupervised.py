# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:12:41 2024

@author: MaxGr
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# Load image
def load_image(filepath):
    with Image.open(filepath) as img:
        return np.array(img)

# Randomly sample pixels from the image
def sample_pixels(image, sample_size=1000):
    h, w, c = image.shape
    pixels = image.reshape(-1, 3)
    sampled_indices = random.sample(range(len(pixels)), sample_size)
    sampled_pixels = pixels[sampled_indices]
    return sampled_pixels, sampled_indices

# Train K-means for unsupervised clustering
def generate_labels_with_kmeans(pixels, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return labels, kmeans

# Train Random Forest
def train_random_forest(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

# Segment the image using trained Random Forest
def segment_image(rf, image):
    h, w, c = image.shape
    image_flat = image.reshape(-1, 3)
    predictions = rf.predict(image_flat)
    segmented = predictions.reshape(h, w)
    return segmented

# Main script
if __name__ == "__main__":
    # Load generated image
    generated_path = "./test/samples_5/75_0002.png"  # Replace with your image path
    
    generated_image = load_image(generated_path)

    # Step 1: Randomly sample pixels from the image
    sampled_pixels, sampled_indices = sample_pixels(generated_image)

    # Step 2: Cluster sampled pixels with K-means
    labels, kmeans = generate_labels_with_kmeans(sampled_pixels)

    # Step 3: Train Random Forest using sampled pixels and their cluster labels
    rf = train_random_forest(sampled_pixels, labels)

    # Step 4: Use Random Forest to segment the entire image
    segmented_image = segment_image(rf, generated_image)

    # Visualize results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(generated_image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Clustered Pixels (K-means)")
    clustered_pixels = np.zeros_like(generated_image.reshape(-1, 3))
    clustered_pixels[sampled_indices] = kmeans.cluster_centers_[labels]
    clustered_pixels = clustered_pixels.reshape(generated_image.shape)
    plt.imshow(clustered_pixels.astype(np.uint8))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmented Image (Random Forest)")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")

    plt.show()
