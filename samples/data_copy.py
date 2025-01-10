# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:27:18 2024

@author: MaxGr
"""



import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Folder paths
output_path = './FID_test/EMDS_6/'
os.makedirs(output_path, exist_ok=True)

input_path = 'D://Data/EMDS-6/EMDS-6/EMDS5-Original/'

folder_list = os.listdir(input_path)
    
for folder_name in folder_list:
    print(folder_name)
    
    file_list = os.listdir(input_path+folder_name)
    
    for file_name in file_list:
        file = folder_name+'_'+file_name
        print(file)
        file_loc = f'{input_path}/{folder_name}/{file_name}'
        target_loc = output_path
        shutil.copy(file_loc, target_loc)
