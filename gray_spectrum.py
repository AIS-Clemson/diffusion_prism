# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 01:54:19 2024

@author: MaxGr
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale_to_spectrum(grayscale_value):
    """Maps a grayscale value (0-255) to a color in the RGB spectrum."""

    hue = (grayscale_value / 255) * 180  # Hue range 0-180 (half the HSV circle)
    saturation = 255  # Full saturation
    value = 255       # Full value (brightness)

    # Convert HSV to BGR (OpenCV format)
    bgr = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]

    return tuple(bgr)  # Convert to tuple (B, G, R)

# Example usage
gray_values = np.arange(0, 256, 1)  # Grayscale values from 0 to 255
color_band = np.array([grayscale_to_spectrum(v) for v in gray_values], dtype=np.uint8)
color_band = color_band.reshape((1, -1, 3))  # Reshape for display

#Display colors
plt.figure(figsize=(10, 5))
plt.imshow(color_band)
plt.axis('off')
plt.show()

