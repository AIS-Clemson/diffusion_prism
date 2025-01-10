# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:02:42 2024

@author: MaxGr
"""

import cv2

def get_click_coordinates(img):
    """
    Opens an image in a cv2 window, waits for mouse clicks, and returns the coordinates.

    Args:
        image_path: The path to the image file.

    Returns:
        A list of tuples, each representing the (x, y) coordinates of a click.
    """
    
    # img = cv2.imread(image_path)
    click_coordinates = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
            click_coordinates.append([x, y])
            print(f"Clicked at: ({x}, {y})")

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.imshow('Image', img)

    while True:
        key = cv2.waitKey(1)  # Wait for a key press
        if key == ord('q'):  # Press 'q' to quit
            break
        # if click_coordinates != []:
        #     break

    cv2.destroyAllWindows()
    return click_coordinates

# Example usage
# image_path = 'path/to/your/image.jpg'
# label_image = np.array(sample_image)
# coordinates = get_click_coordinates(label_image)
# print("All click coordinates:", coordinates)

import os
import shutil

def train_file_divider(folder):
    # sample_folder = './samples_20k/'
    sample_folder = folder
    train_folder = sample_folder + '/sample/'
    gt_folder = sample_folder + '/mask/'
    
    os.makedirs(train_folder, exist_ok = True)
    os.makedirs(gt_folder, exist_ok = True)
    
    file_list = os.listdir(sample_folder)
    
    for file in file_list:
        print(file)
        name = file.replace('.','_').split('_')
        file_loc = sample_folder+file
        if 'png' in name:
            shutil.move(file_loc, train_folder + file)
        if 'mask' in name:
            shutil.move(file_loc, gt_folder + file)
    
    
# train_file_divider('./samples_20k/')




# import os

# folder = './samples_20k/sample/'
# file_list = os.listdir(folder)
# for file in file_list:
#     names = file.split('.')
#     name = names[0]
#     if name[-1] == '_':
#         print(name)
#         old_name = folder + file
#         new_name = folder + name[:-1] + names[1]
#         os.rename(old_name, new_name)
    

# folder = './samples_20k/sample/'
# file_list = os.listdir(folder)
# for file in file_list:
#     names = file.split('.')
#     if names[-1] not in ['jpg', 'png']:
#         old_name = folder + file
#         new_name = folder + file[:-3] + '.jpg'
#         print(new_name)
#         os.rename(old_name, new_name)
    




# import os
# import random
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Folder paths
# output_path = './FID_test/Micro_Organism/'

# input_path = 'D://Data/Micro_Organism/'

# folder_list = os.listdir(input_path)
    
# for folder_name in folder_list:
#     print(folder_name)
    
#     file_list = os.listdir(input_path+folder_name)
    
#     for file_name in file_list:
#         file = folder_name+'_'+file_name
#         print(file)
#         file_loc = f'{input_path}/{folder_name}/{file_name}'
#         target_loc = output_path+file
#         shutil.move(file_loc, target_loc)



import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Folder paths
output_path = './FID_test/'
# input_path = './test/'

# noise_test = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
noise_test = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for noise in noise_test:
    os.makedirs(f'{output_path}/FID_{noise}', exist_ok=True)

    print(noise)
    # generated_images_dir = f'./test/samples_noise/samples_noise_{noise}/'
    generated_images_dir = f'./test/samples_prism/samples_noise_prism_{noise}/'

    file_list = os.listdir(generated_images_dir)
    
    for file_name in file_list:
        names = file_name.split('.')
        if 'png' in names:
            print(noise, file_name)
            file_loc = f'{generated_images_dir}/{file_name}'
            target_loc = f'{output_path}/FID_{noise}/{names[0]}.jpg'
            shutil.move(file_loc, target_loc)
    




import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Folder paths
output_path = './FID_test/'
# input_path = './test/'

# noise_test = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# noise_test = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for noise in noise_test:
    os.makedirs(f'{output_path}/FID_{noise}', exist_ok=True)

    print(noise)
    # generated_images_dir = f'./test/samples_noise/samples_noise_{noise}/'
    generated_images_dir = f'./test/samples_prism/samples_noise_prism_{noise}/'

    file_list = os.listdir(generated_images_dir)
    
    for file_name in file_list:
        names = file_name.split('.')
        if 'png' in names:
            print(noise, file_name)
            file_loc = f'{generated_images_dir}/{file_name}'
            target_loc = f'{output_path}/FID_{noise}/{names[0]}.jpg'
            shutil.move(file_loc, target_loc)
    























