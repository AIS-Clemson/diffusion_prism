# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:09:03 2024

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# path = './'
# image_folder = path + '/test/samples_test/'

dataset_info = np.load('./CLIP_test/CLIP_data.npy', allow_pickle=True)
print(dataset_info.shape)

scale_list = dataset_info[:,6]
strength_list = dataset_info[:,7]
ddim_steps_list = dataset_info[:,8]
noise_list = dataset_info[:,9]
CLIP_score_list = dataset_info[:,-2]
CLIP_similarity_list = dataset_info[:,-1]

# plt.plot(CLIP_score_list)
# plt.plot(strength_list)

# plt.scatter(strength_list, CLIP_score_list)
# plt.scatter(scale_list, CLIP_score_list)
# plt.scatter(noise_list, CLIP_score_list)


'''
CLIP - scale test
'''
scale_min = np.min(scale_list)
scale_max = np.max(scale_list)
scale_step = 10
scale_index = np.arange(scale_min, scale_max+scale_step, scale_step)
scale_length = len(scale_index)

CLIP_scale_list = [[] for _ in range(scale_length)]
for data_i in dataset_info:
    clip_score = data_i[-1]
    scale_i = data_i[6]
    
    index = np.argmin(abs(scale_index-scale_i))
    # index = np.where(scale_i == scale_index)[0][0]
    CLIP_scale_list[index].append(clip_score)
    
for i in range(len(CLIP_scale_list)):
    CLIP_scale_list[i] = np.mean(CLIP_scale_list[i])

plt.scatter(scale_index, CLIP_scale_list)



'''
CLIP - strength test
'''
strength_min = np.min(strength_list)
strength_max = np.max(strength_list)
strength_step = 0.1
strength_index = np.arange(strength_min, strength_max+strength_step, 0.1)
strength_length = len(strength_index)

CLIP_strength_list = [[] for _ in range(strength_length)]
for data_i in dataset_info:
    clip_score = data_i[-1]
    strength_i = data_i[7]
    
    index = int(round(strength_i,1)*10 -1)
    CLIP_strength_list[index].append(clip_score)
    
for i in range(len(CLIP_strength_list)):
    CLIP_strength_list[i] = np.mean(CLIP_strength_list[i])

plt.scatter(strength_index, CLIP_strength_list)




'''
CLIP - noise test
'''
noise_loc = []

plt.figure()
noise_min = np.min(noise_list)
noise_max = np.max(noise_list)
noise_step = 0.1
noise_index = np.round(np.arange(noise_min, noise_max+noise_step, noise_step), 2)

# noise_test = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
noise_test = [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1]

noise_index = np.array(noise_test)

noise_length = len(noise_index)

CLIP_noise_list = [[] for _ in range(noise_length)]
SSIM_noise_list = [[] for _ in range(noise_length)]


for data_i in dataset_info:
    clip_score = data_i[-2]
    ssim_score = data_i[-1]

    noise_i = data_i[9]
    
    index = np.argmin(abs(noise_index-noise_i))
    # index = np.where(scale_i == scale_index)[0][0]
    CLIP_noise_list[index].append(clip_score)
    SSIM_noise_list[index].append(ssim_score)
    
for i in range(len(CLIP_noise_list)):
    CLIP_noise_list[i] = np.mean(CLIP_noise_list[i])
    SSIM_noise_list[i] = np.mean(SSIM_noise_list[i])

plt.plot(noise_index, CLIP_noise_list)
plt.plot(noise_index, SSIM_noise_list)

plt.scatter(SSIM_noise_list, CLIP_noise_list)
for i, txt in enumerate(zip(SSIM_noise_list, CLIP_noise_list)):
    plt.text(txt[0]+0.001, txt[1]+0.01, f'{noise_test[i]}', fontsize=12)  # Format values as needed



# '''
# CLIP - strength - noise test
# '''
# plt.figure()
# noise_test = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
# CLIP_strength_noise_list = []
# for noise in noise_test:
#     temp_noise_list = np.round(noise_list.astype(float), 2)
#     new_df = dataset_info[np.where(temp_noise_list==noise)]
#     new_strength_list = new_df[:,7]
    
#     data_min = np.min(new_strength_list)
#     data_max = np.max(new_strength_list)
#     data_step = 0.1
#     data_index = np.arange(data_min, data_max+data_step, 0.1)
#     data_length = len(data_index)

#     temp_list = [[] for _ in range(data_length)]
#     for data_i in new_df:
#         clip_score = data_i[-1]
#         strength_i = data_i[7]
        
#         index = np.argmin(abs(data_index-strength_i))
#         # index = int(round(strength_i,1)*10 -1)
#         temp_list[index].append(clip_score)
        
#     for i in range(len(temp_list)):
#         temp_list[i] = np.mean(temp_list[i])

#     plt.plot(data_index, temp_list, label=str(noise))
#     CLIP_strength_noise_list.append(temp_list)
# plt.legend()



'''
FID 
'''
FID_noise_list = [905.9038674832452, 
                  799.5856,
                  776.1199,
                  748.5967, 
                  742.1110419058718, 
                   # 752.6305368478531, 
                   # 756.8794090234603, 
                   # 756.4478359943434, 
                  765.151981669017, 
                   # 767.0741833255396, 
                   # 753.1882310848806, 
                   # 768.6514153965131, 
                   # 776.3219741154298, 
                  801.3551562528703]


FID_prism_list = [859.10335914653, 
                  785.8339565828779, 
                  752.8221634619833, 
                  689.2095161175585, 
                  666.7083423309739, 
                  # 639.1605983595989, 
                  # 624.0216125495193, 
                  # 629.384067196893, 
                  636.2317680380098,
                  # 672.3838568224235, 
                  # 703.7692507779453, 
                  # 717.6205117201521, 
                  # 725.6868400448736, 
                  733.0761927624376]

# FID_noise = [9.059038674832452509e-01,
# 7.421110419058718488e-01,
# 7.526305368478530822e-01,
# 7.568794090234602923e-01,
# 7.564478359943433983e-01,
# 7.651519816690169895e-01,
# 7.670741833255395603e-01,
# 7.531882310848806439e-01,
# 7.686514153965131602e-01,
# 7.763219741154298026e-01,
# 8.013551562528703487e-01]

FID_noise = np.array(FID_noise_list)/1500
FID_prism = [859.1034,
             785.8340,
             752.8222,
             689.2095,
             666.7083,
              # 639.1606,
              # 624.0216,
              # 629.3841,
             636.2318,
              # 672.3839,
              # 703.7693,
              # 717.6205,
              # 725.6868,
             733.0762]
FID_prism = np.array(FID_prism)/1500

control_clip_score, control_similarity_score, control_fid_score = 27.97, 0.8633, 1279.7968029402127/1500
SD_similarity_score, SD_clip_score = SSIM_noise_list[0], CLIP_noise_list[0]


random_clip_score, random_similarity_score, random_fid_score = 27.77, 0.6666, 1243.9515230583354/1500


plt.figure()
plt.scatter(noise_test[0], FID_noise[0], color='C0', marker='^', s=100, label='SD', zorder=3)
plt.scatter(noise_test[0], FID_prism[0], color='C1', marker='^', s=100, label='SD (w/ chroma)', zorder=3)

plt.scatter(noise_test[1:], FID_noise[1:], color='C2', marker='^', s=100, label='Prism (noise only)', zorder=3)
plt.scatter(noise_test[1:], FID_prism[1:], color='C4', marker='^', s=100, label='Prism (noise + chroma)', zorder=2)
plt.plot(noise_test[1:], FID_noise[1:], zorder=3, color='C2')
plt.plot(noise_test[1:], FID_prism[1:], zorder=2, color='C4')
plt.xlabel('Noise Level', fontsize=14)
plt.ylabel('FID', fontsize=14)
# for i, txt in enumerate(zip(SSIM_noise_list, FID_noise_list)):
#     plt.text(txt[0]+0.001, txt[1]+0.01, f'{noise_test[i]}', fontsize=12)  # Format values as needed
plt.grid(True, zorder=1)  # Set grid zorder to 1 (behind markers)
plt.legend(fontsize=12)
plt.show()



SSIM_noise_list = [0.9692,
0.9697,
0.9680,
0.9636,
0.9566,
# 0.9409,
# 0.9373,
# 0.9338,
0.9254,
# 0.9266,
# 0.9247,
# 0.9304,
# 0.9265,
0.9254
]

control_similarity_score = 0.9801

plt.figure()
plt.scatter(SSIM_noise_list[0], FID_prism[0], color='C0', marker='*', s=200, label='SD1.5', zorder=2) # Plot first two points as stars
plt.scatter(SSIM_noise_list[1:], FID_prism[1:], color='C2', marker='^', s=100, label='Prism', zorder=2)
plt.plot(SSIM_noise_list[1:], FID_prism[1:], color='C2', zorder=2)
plt.scatter(control_similarity_score, control_fid_score, color='C1', marker='^', s=100, label='ControlNet', zorder=2)

plt.scatter(0.9378, 0.8311, color='C3', marker='s', s=100, label='Uni-ControlNet', zorder=2)

plt.xlabel('Similarity', fontsize=14)
plt.ylabel('FID', fontsize=14)
# for i, txt in enumerate(zip(SSIM_noise_list, FID_prism)):
    # plt.text(txt[0]-0, txt[1]+0.02, f'{noise_test[i]}', fontsize=14)  # Format values as needed
plt.grid(True, zorder=1)  # Set grid zorder to 1 (behind markers)
plt.legend(fontsize=14)
plt.show()


plt.figure()
plt.scatter(CLIP_noise_list[0], FID_prism[0], color='C0', marker='*', s=200, label='SD', zorder=2) # Plot first two points as stars
plt.scatter(CLIP_noise_list[1:], FID_prism[1:], color='C2', marker='^', s=100, label='Prism', zorder=2)
plt.plot(CLIP_noise_list[1:], FID_prism[1:], color='C2', zorder=2)
plt.scatter(control_clip_score, control_fid_score, color='C1', marker='^', s=100, label='ControlNet', zorder=2)

plt.scatter(28.01, 0.8311, color='C3', marker='s', s=100, label='Uni-ControlNet', zorder=2)

plt.xlabel('CLIP Score', fontsize=14)
plt.ylabel('FID', fontsize=14)
# for i, txt in enumerate(zip(CLIP_noise_list, FID_prism)):
    # plt.text(txt[0]+0.001, txt[1]+0.01, f'{noise_test[i]}', fontsize=12)  # Format values as needed
plt.grid(True, zorder=1)  # Set grid zorder to 1 (behind markers)
# plt.xlim(0.9, 1)  # Set x-axis limits
plt.xlim(27.5, 31)  # Set y-axis limits
plt.legend(fontsize=14)
plt.show()


# plt.scatter(control_similarity_score, control_clip_score)






# plt.figure()
# plt.scatter(SSIM_noise_list, CLIP_noise_list)
# plt.scatter(control_similarity_score, control_clip_score)



plt.figure()
plt.scatter(SSIM_noise_list[1:], CLIP_noise_list[1:], color='C2', marker='^', s=100, label='Prism', zorder=2) # Plot first two points as stars
plt.scatter(SSIM_noise_list[0], CLIP_noise_list[0], color='C0', marker='*', s=200, label='SD', zorder=2) # Plot first two points as stars
plt.plot(SSIM_noise_list[1:], CLIP_noise_list[1:], color='C2', zorder=2)
plt.scatter(control_similarity_score, control_clip_score, color='C1', marker='o', s=100, label='ControlNet', zorder=2) # Plot last point as triangle
plt.xlabel('Similarity Score', fontsize=14)
plt.ylabel('CLIP Score', fontsize=14)
# plt.title('SSIM Noise vs CLIP Noise')
plt.grid(True, zorder=1)  # Set grid zorder to 1 (behind markers)
plt.xlim(0.9, 1)  # Set x-axis limits
plt.ylim(27.5, 30)  # Set y-axis limits

plt.legend(fontsize=14, loc=3)
plt.show()





