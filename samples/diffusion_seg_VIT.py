# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 01:25:16 2023

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
# from torchsummary import summary
from torchsummaryX  import summary

import torch
import torch.nn as nn
import torch.nn.functional as F


print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
device = torch.device("cuda")
# print(torch.cuda.get_arch_list())



from torchvision.models import vit_b_16

class ViT_Seg(nn.Module):
    def __init__(self, num_classes=1):
        super(ViT_Seg, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        
        # self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        # self.vit.heads = nn.Identity()
        self.layernorm_output = None
        self.vit.encoder.ln.register_forward_hook(self.save_layernorm_output)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)  # 112x112 -> 224x224
        )
        
    def save_layernorm_output(self, module, input, output):
        self.layernorm_output = output
    
    def forward(self, x, bbox=None):
        _ = self.vit(x)  # activate hook
        x = self.layernorm_output  # [batch_size, num_patches + 1, embedding_dim]

        x = x[:, 1:, :] # remove class token
        
        batch_size, num_patches, embedding_dim = x.size()
        height = width = int(num_patches ** 0.5)  # Assuming square patches
        x = x.transpose(1, 2).view(batch_size, embedding_dim, height, width)
        
        x = self.decoder(x) 
        
        return x


# Test the model
# if __name__ == "__main__":
x = torch.randn(1, 3, 224, 224).to(device)
# model = UNet().to(device)
model = ViT_Seg(num_classes=1).to(device)
# model = vit_b_16().to(device)
output = model(x)
# print(output.size())  # Output size: torch.Size([1, 1, 512, 512])

device = torch.device("cuda")
table = summary(model, x)









import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_list = os.listdir(image_paths)
        self.mask_list = os.listdir(mask_paths)
        self.transform = True

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_paths, self.image_list[idx])
        
        file_name = self.image_list[idx]
        # image_name = file_name.split('_')[0] + '.jpg'
        # mask_path  = os.path.join(self.mask_paths, image_name)
        mask_name = file_name.split('.')[0] + '_mask.jpg'
        mask_path  = os.path.join(self.mask_paths, mask_name)
            
        # image_path = self.image_paths[idx]
        # mask_path = self.mask_paths[idx]

        # print(image_path)
        # print(mask_path)
        
        # Load image and mask using PIL (you can use other libraries if needed)
        image = Image.open(image_path).convert('RGB')
        # image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale (single channel)

        # Apply transformations if provided
        if self.transform:
            image = transform(image)
            mask = totensor(mask)

        return image, mask



# Example data and targets (dummy data, replace with your actual data)
image_paths = './samples_20k/sample/'
# mask_paths  = './samples_mask/' 
# mask_paths  = './samples_skeleton/' 
mask_paths  = './samples_20k/mask/' 


# Transforms to be applied to the data
transform = transforms.Compose([
    # transforms.ToPILImage(),        # Convert to PIL Image
    transforms.Resize((224, 224)),  # Resize to (256, 256)
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]), # norm
])

# Transforms to be applied to the data
totensor = transforms.Compose([
    # transforms.ToPILImage(),        # Convert to PIL Image
    transforms.Resize((224, 224)),  # Resize to (256, 256)
    transforms.ToTensor(),           # Convert to tensor
    # transforms.Normalize(mean=[0.5], std=[0.5]), # norm
])


# Create custom dataset and dataloader
custom_dataset = CustomDataset(image_paths, mask_paths, transform)
train_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

# Test the data loader
for batch in train_loader:
    images, masks = batch
    print(images.size())    # torch.Size([32, 3, 256, 256]) (batch_size=32, 3 channels, 256x256 size)
    print(masks.size())   # torch.Size([32, 1, 256, 256]) (batch_size=32, single-channel masks, 256x256 size)
    break


# for i in range(5):
#     plt.figure()
#     img = images[i]
#     img = img.permute(1,2,0)
#     plt.imshow(img)
#     plt.figure()
#     mask = masks[i]
#     mask = mask.permute(1,2,0)
#     plt.imshow(mask)
    
# temp = mask.data.numpy()[:,:,0]




def norm_image(x):
    # Get the minimum and maximum values in the tensor
    min_value = np.min(x)
    max_value = np.max(x)

    # Normalize the tensor to the range [0, 1]
    normalized_array = (x - min_value) / (max_value - min_value) *255

    return normalized_array.astype(np.uint8)


def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image


import cv2

def display_result(images, masks, outputs):
    # Select the first 3 images from the batch
    images = images[:4]
    masks = masks[:4]
    outputs = outputs[:4]

    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    masks = masks.detach().cpu().permute(0, 2, 3, 1).numpy()
    outputs = outputs.detach().cpu().permute(0, 2, 3, 1).numpy()

    for i in range(4):
        image = images[i, :, :]
        mask = masks[i, :, :, 0]
        output = outputs[i, :, :, 0]
        
        heatmap_rgb = plt.get_cmap('jet')(output)[:, :, :3]  # RGB channels only

        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

        # mask = cv2.merge((mask, mask, mask))
        # output = cv2.merge((output, output, output))

        # Normalize and convert to uint8
        image = norm_image(image)
        mask = norm_image(mask)
        output = norm_image(output)
        
        # heatmap_rgb = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        heatmap_rgb = norm_image(heatmap_rgb)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

        # Stack the mask and output horizontally
        combined_mask_output = cv2.hconcat([image, mask, output, heatmap_rgb])

        if i == 0:
            combined_image = combined_mask_output
        else:
            combined_image = cv2.vconcat([combined_image, combined_mask_output])
    
    return combined_image




class TestDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.image_list = os.listdir(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_paths, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        # image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)
        
        return image


test_path = './dendrite_sample/'
test_list = os.listdir(test_path)

test_dataset = TestDataset(test_path, transform)
test_loader = DataLoader(test_dataset, batch_size=len(test_list), shuffle=False)




import torch.optim as optim

# Move the model and loss function to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and loss to GPU
# model = UNet().to(device)
# model = AE().to(device)

# Define loss function (using Mean Squared Error loss)
criterion = nn.MSELoss().to(device)
# criterion = nn.L1Loss().to(device)

# Define optimizer (you can adjust learning rate and other hyperparameters)
optimizer = optim.Adam(model.parameters(), lr=1e-6)



import time
from tqdm import tqdm


# Define the training loop
load_weight = False

weight_file_path = './last.pt'
if os.path.exists(weight_file_path) and load_weight==True:
    # Load the weights and resume training
    model.load_state_dict(torch.load(weight_file_path))
    print("Weight file exists. Resuming training...")

    
os.makedirs(os.path.join('./', "train_results"), exist_ok=True)
os.makedirs(os.path.join('./', "test_results"), exist_ok=True)
os.makedirs(os.path.join('./', "weights"), exist_ok=True)


num_epochs = 200
loss_list = []


start_time = time.time()  # Record the start time


for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    pbar = tqdm(train_loader)
    
    epoch_start_time = time.time()  # Record the start time

    for i, (images, masks) in enumerate(pbar):
    # for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        
        # image_256 = F.interpolate(outputs, scale_factor=0.125, mode="bilinear")   
        # mask_256 = F.interpolate(masks, scale_factor=0.125, mode="bilinear")   

        # loss = criterion(image_256, mask_256)
        area_loss = -torch.sum(masks) * 1e-7
        mask_loss = criterion(outputs, masks)
        
        loss = mask_loss #+area_loss
        
        loss.backward()
        optimizer.step()

        # Calculate elapsed time for the current epoch
        epoch_time = time.time() - epoch_start_time
        remaining_epochs = num_epochs - epoch - 1
        estimated_remaining_time = epoch_time * remaining_epochs
    
        # Update running loss
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item(), Epochs=epoch)


    # Calculate average loss for the epoch
    # epoch_loss = running_loss / len(train_loader.dataset)
    epoch_loss = running_loss / i
    loss_list.append(epoch_loss)

    estimated_remaining_hour = estimated_remaining_time/3600
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
          f'Epoch Time: {epoch_time:.2f}s, '
          f'Estimated Remaining Time: {estimated_remaining_time:.2f}s ({estimated_remaining_hour:.2f}h)')
    
    # Save model weights after every epoch
    torch.save(model.state_dict(), f'./last.pt')
    # torch.save(model.state_dict(), f'./weights/{epoch}.pt')
    
    
    # Save some visualizations of the input images, predicted masks, and ground truth masks
    with torch.no_grad():
        outputs = model(images)
        
        combined_image = display_result(images, masks, outputs)

        # plt.figure()
        # plt.imshow(combined_image)
        # plt.show()
        cv2.imwrite(f'./train_results/{epoch}.png', combined_image)
        
        
        
        for j, (images) in enumerate(test_loader):
            images = images.to(device)
            
            outputs = model(images)
        
            images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            outputs = outputs.detach().cpu().permute(0, 2, 3, 1).numpy()
        
            for k in range(len(images)):
                image = images[k, :, :]
                output = outputs[k, :, :, 0]
                heatmap_rgb = plt.get_cmap('jet')(output)[:, :, :3]  # RGB channels only
                
                output = cv2.merge((output, output, output))
        
                # Normalize and convert to uint8
                image = norm_image(image)
                output = norm_image(output)
                heatmap_rgb = norm_image(heatmap_rgb)
                heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
                # Stack the mask and output horizontally
                combined_mask_output = cv2.hconcat([image, output, heatmap_rgb])
        
                if k == 0:
                    combined_image = combined_mask_output
                else:
                    combined_image = cv2.vconcat([combined_image, combined_mask_output])
                    
            # plt.figure()
            # plt.imshow(combined_image)
            # plt.show()
            cv2.imwrite(f'./test_results/{epoch}.png', combined_image)



# Calculate the total training time
total_training_time = (time.time() - start_time)/3600
print('Finished Training')
print(f'Total Training Time: {total_training_time:.2f} h')


# Plot the loss curve
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()















# # Load the trained model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = AE().to(device)
# model.load_state_dict(torch.load('./weights/last.pt'))
# model.eval()


# test_path = './dendrite_sample'
# test_list = os.listdir(test_path)

# # Save some visualizations of the input images, predicted masks, and ground truth masks
# with torch.no_grad():
#     for image in test_list:
#         name = image
#         print(name)
#         image = Image.open(os.path.join(test_path, image)).convert('RGB')
#         image = transform(image)
#         image = image.to(device).unsqueeze(0)
        
#         output = model(image)
 
#         images = image.detach().cpu().permute(0, 2, 3, 1).numpy()
#         outputs = output.detach().cpu().permute(0, 2, 3, 1).numpy()
    
#         for i in range(1):
#             image = images[i, :, :]
#             output = outputs[i, :, :, 0]
#             heatmap_rgb = plt.get_cmap('jet')(output)[:, :, :3]  # RGB channels only
    
#             output = cv2.merge((output, output, output))
    
#             # Normalize and convert to uint8
#             image = norm_image(image)
#             output = norm_image(output)
#             heatmap_rgb = norm_image(heatmap_rgb)
#             heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            
#             h,w,z = image.shape
    
#             output = cv2.resize(output, (w,h))
#             heatmap_rgb = cv2.resize(heatmap_rgb, (w,h))

#             # Stack the mask and output horizontally
#             combined_mask_output = cv2.hconcat([image, output, heatmap_rgb])
    
#             if i == 0:
#                 combined_image = combined_mask_output
#             else:
#                 combined_image = cv2.vconcat([combined_image, combined_mask_output])
                
#         plt.figure()
#         plt.imshow(combined_image)
#         plt.show()
#         cv2.imwrite('./'+name, combined_image)
    
    
# print("Testing complete. Results saved.")

























