# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 23:56:27 2023

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import sys
from utils import *



import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"





# !nvidia-smi
HOME = './'
SAM_CHECKPOINT_PATH = os.path.join(HOME, "segment_any", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))




GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

import sys
sys.path.insert(0, './GroundingDINO')
# !pip install -e .


from groundingdino.util.inference import Model, load_model, load_image, predict, annotate
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)



# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# SAM_ENCODER_VERSION = "vit_h"
# sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
# sam_predictor = SamPredictor(sam)
# mask_generator = SamAutomaticMaskGenerator(sam)











import glob
import random
# image_folder = 'wildfire_images/wildfire/'
# image_folder = 'YOLO/wildfire_images/'
# image_folder = 'YOLO/test/'
# image_folder = 'Diffusion/train_2/image/'
image_folder = ''



image_path = glob.glob(HOME+image_folder+'*.png')
image_names = [os.path.basename(image) for image in image_path]  # Extract file names
sampled_names = random.sample(image_names, 1)


IMAGE_NAME = sampled_names[0]
IMAGE_PATH = os.path.join(HOME, image_folder, IMAGE_NAME)
# IMAGE_PATH = 'bus.jpg'
# IMAGE_PATH = './example/images/000000000308.jpg'
# IMAGE_PATH = 'game.jpg'

IMAGE_PATH = 'image_5.png'
IMAGE_NAME = IMAGE_PATH

print(IMAGE_PATH)



SOURCE_IMAGE_PATH = IMAGE_PATH
# CLASSES = ['wildfire', 'fire', 'tree', 'snow', 'soil', 'ground', 'sky', 'people', 'car', 'smoke', 
#             'wine', 'glasses', 'table', 'hat', 'animal', 'windows', 'sign', 'shoes', 'hands', 
#             'yellow light', 'light', 'red light', 'orange light']

# CLASSES = ['hero', 'people', 'car', 'signal', 'light', 'tree']

CLASSES = [
    'car', 'tree', 'person', 'bus', 'bicycle', 'motorcycle', 'traffic light', 'stop sign',
    'fountain',
    'crosswalk', 'sidewalk', 'door', 'stair', 'escalator', 'elevator', 'ramp',
    'bench', 'trash can', 'pole', 'fence', 'tree', 'dog', 'cat', 'bird', 'parking meter',
    'mailbox', 'manhole', 'puddle', 'construction sign', 'construction barrier',
    'scaffolding', 'hole', 'crack', 'speed bump', 'curb', 'guardrail', 'traffic cone',
    'traffic barrel', 'pedestrian signal', 'street sign', 'fire hydrant', 'lamp post',
    'bench', 'picnic table', 'public restroom', 'fountain', 'statue', 'monument',
    'directional sign', 'information sign', 'map', 'emergency exit', 'no smoking sign',
    'wet floor sign', 'closed sign', 'open sign', 'entrance sign', 'exit sign',
    'stairs sign', 'escalator sign', 'elevator sign', 'restroom sign', 'men restroom sign',
    'women restroom sign', 'unisex restroom sign', 'baby changing station',
    'wheelchair accessible sign', 'braille sign', 'audio signal device', 'tactile paving',
    'detectable warning surface', 'guide rail', 'handrail', 'turnstile', 'gate',
    'ticket barrier', 'security checkpoint', 'metal detector', 'baggage claim',
    'lost and found', 'information desk', 'meeting point', 'waiting area', 'seating area',
    'boarding area', 'disembarking area', 'charging station', 'water dispenser',
    'vending machine', 'ATM', 'kiosk', 'public telephone', 'public Wi-Fi hotspot',
    'emergency phone', 'first aid station', 'defibrillator',
    'tree', 'pole', 'lamp post', 'staff', 'road hazard'
]

BOX_TRESHOLD = 0.2
TEXT_TRESHOLD = 0.15


from typing import List

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


import cv2
import supervision as sv

# load image
image = cv2.imread(SOURCE_IMAGE_PATH)
# image = resize_image(image, 720)


# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=enhance_class_name(class_names=CLASSES),
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _ 
    in detections]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

sv.plot_image(annotated_frame, (16, 16))


cv2.imwrite(f'./plot/{IMAGE_NAME}', annotated_frame)










import numpy as np
# from segment_anything import SamPredictor


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)




import cv2

# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _ 
    in detections]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# %matplotlib inline
sv.plot_image(annotated_image, (16, 16))

cv2.imwrite(f'./plot/GSAM_{IMAGE_NAME}', annotated_image)




# import cv2
# import matplotlib.pyplot as plt
# import supervision as sv
# print(sv.__version__)

# image_bgr = cv2.imread(IMAGE_PATH)
# # image_bgr = resize_image(image_bgr, 1024)
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# plt.imshow(image_rgb)


# sam_result = mask_generator.generate(image_rgb)
# print(sam_result[0].keys())


# mask_annotator = sv.MaskAnnotator()
# detections = sv.Detections.from_sam(sam_result=sam_result)
# annotated_image = sam_predictor.annotate(scene=image_bgr.copy(), detections=detections)

# sv.plot_images_grid(
#     images=[image_bgr, annotated_image],
#     grid_size=(1, 2),
#     titles=['source image', 'segmented image']
# )


# cv2.imwrite('test.jpg', annotated_image)
























































