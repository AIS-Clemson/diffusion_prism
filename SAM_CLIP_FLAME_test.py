# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:02:48 2023

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import sys
from utils import *

# !nvidia-smi
HOME = './'
CHECKPOINT_PATH = os.path.join(HOME, "segment_any", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))



import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"




import cv2
import matplotlib.pyplot as plt
import supervision as sv
print(sv.__version__)

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)





import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)




def read_bounding_boxes_file(file_path):
    bounding_boxes = []
    
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            values = [int(values[0])] + [float(val) for val in values[1:]]
            bounding_boxes.append(values)
    
    return bounding_boxes



import time
import glob
import random


# folder = 'Diffusion/train_2/image/'
# folder = 'Diffusion/wildfire_5/sample_2/'
# folder = 'Diffusion/wildfire_5/real_sample/'
# folder = 'Diffusion/wildfire_5/FLAME1/'
# folder = 'outputs/img2img-samples/samples/'
image_folder = 'CLIP/image/'
mask_folder = 'CLIP/mask/'

# image_path = glob.glob(HOME+folder+'*.png')
# image_names = [os.path.basename(image) for image in image_path]  # Extract file names

start_time = time.time()
# sampled_names = random.sample(image_names, 1)

# IMAGE_NAME = sampled_names[0]

# image_name = '131_103_17_279__png.rf.844c7f751ebf158a29cd7d4aba718a94'

# folder = 'Flame-3-Diffusion-10/test/images/'
IMAGE_NAME = image_name+'.jpg'
IMAGE_PATH = os.path.join(HOME, folder, IMAGE_NAME)
print(IMAGE_PATH)

bounding_boxes = read_bounding_boxes_file('Flame-3-Diffusion-10/test/labels/'+
                                 image_name+'.txt')




image_bgr = cv2.imread(IMAGE_PATH)
image_bgr = resize_image(image_bgr, 1024)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)


sam_result = mask_generator.generate(image_rgb)
print(sam_result[0].keys())

# mask_annotator = sv.MaskAnnotator()
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=sam_result)
annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

end_time = time.time()
print(f'Total tine cost: {end_time-start_time}')


sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)



# CLASSES = ["fire"]
CLASSES = ["fire","smoke", "tree", "rock", "people", "building", "car", "cloud", "snow"]


# bounding_boxes = detections.xyxy

bboxes = []
classification_results = []
for box in bounding_boxes:
    # x1, y1, x2, y2 = box.astype(int)
    
    _,x,y,w,h = box
    
    w, h = int(w * image_rgb.shape[1]), int(h * image_rgb.shape[0])
    x1, y1 = int((x * image_rgb.shape[1])-w//2), int((y * image_rgb.shape[0])-h//2)
    x2, y2 = int(x1+w), int(y1+h)

    bboxes.append([x1, y1, x2, y2])
    cropped_area = image_rgb[y1:y2, x1:x2]
    # plt.imshow(cropped_area)

    
    image = preprocess(Image.fromarray(cropped_area)).unsqueeze(0).to(device)
    text = clip.tokenize(CLASSES).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    
    class_id = probs[0].argmax()
    confidence = probs[0][class_id]

    classification_results.append(f"{CLASSES[class_id]} {confidence:0.2f}")
    
import numpy as np


box_annotator = sv.BoxAnnotator()
labels = classification_results

detections.xyxy = np.array(bboxes)


annotated_frame = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, labels=labels)
sv.plot_image(annotated_frame, (32, 32))


cv2.imwrite(f'./plot/SAM_{IMAGE_NAME}', annotated_image)
cv2.imwrite(f'./plot/CLIP_{IMAGE_NAME}', annotated_frame)
cv2.imwrite(f'./plot/{IMAGE_NAME}', image_bgr)











