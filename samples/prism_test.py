# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:55:36 2024

@author: MaxGr
"""

import cv2

def pixel_rgb_and_stats(image_path):
    """
    Displays RGB value of a selected pixel and calculates mean and std for each channel.

    Args:
        image_path (str): Path to the image file.
    """

    img = cv2.imread(image_path)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get RGB value at clicked pixel
            b, g, r = img[y, x]
            print(f"RGB at ({x}, {y}): [{r}, {g}, {b}]")

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    while True:
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Calculate mean and std for each channel
    b, g, r = cv2.split(img)
    stats = {
        'r': [r.mean(),r.std(),], 
        'g': [g.mean(),g.std(),], 
        'b': [b.mean(),b.std(),], 
        }

    print("\nChannel Statistics:")
    print(stats)

    cv2.destroyAllWindows()

# Example usage
path = './dendrite_sample/'
image_path = path + 'eyeQ.png'  # Replace with your image path
pixel_rgb_and_stats(image_path)





























