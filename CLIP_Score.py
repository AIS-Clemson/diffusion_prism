# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 02:47:10 2024

@author: MaxGr
"""

import os
import clip
import torch
from PIL import Image
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model and preprocess function
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to load image-text pairs from directories
def load_image_text_pairs(image_dir, text_dir):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                   if fname.endswith(".jpg") or fname.endswith(".png")]
    text_paths = [os.path.join(text_dir, fname.replace('.jpg', '.txt').replace('.png', '.txt')) for fname in os.listdir(image_dir)
                  if fname.endswith(".jpg") or fname.endswith(".png")]
    return image_paths, text_paths

# # Function to load images from a directory and close files properly
# def load_images_from_directory(directory, max_images=None):
#     images = []
#     file_list = os.listdir(directory)
#     random.shuffle(file_list)

#     for filename in tqdm(file_list, desc=f"Load images from {directory}"):
#         if max_images and len(images) >= max_images: break
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             with Image.open(os.path.join(directory, filename)) as img:
#                 images.append(img.convert("RGB"))
                
#     return images

# Function to calculate CLIP score
def calculate_clip_score(image_paths, text_paths, model, preprocess):
    image_features = []
    text_features = []
    
    for img_path, txt_path in tqdm(zip(image_paths, text_paths), desc="Calculating CLIP scores", total=len(image_paths)):
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with open(txt_path, 'r') as file:
            text = file.read().strip()
        
        text = clip.tokenize([text]).to(device)
        
        with torch.no_grad():
            image_feature = model.encode_image(image)
            text_feature = model.encode_text(text)
        
        image_features.append(image_feature)
        text_features.append(text_feature)
    
    # Stack all features and calculate cosine similarity
    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarities = (image_features @ text_features.T).diag().cpu().numpy()
    
    # Return average similarity score as the CLIP score
    return similarities.mean()

# Example usage
if __name__ == "__main__":
    # Directory paths for generated images and corresponding text descriptions
    generated_images_dir = 'path/to/generated/images'
    text_descriptions_dir = 'path/to/text/descriptions'
    
    # Load image-text pairs
    image_paths, text_paths = load_image_text_pairs(generated_images_dir, text_descriptions_dir)
    
    # Calculate CLIP score
    clip_score = calculate_clip_score(image_paths, text_paths, model, preprocess)
    print('CLIP Score:', clip_score)












