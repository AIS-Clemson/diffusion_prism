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

# the architecture of the gan
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
        )
        
    
    def forward(self,x):
        x = self.conv_init(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        return x
    
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Contracting Path (Encoder)
        self.conv1 = self.conv_block(3, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)

        # Expansive Path (Decoder)
        self.upconv3 = self.upconv_block(256, 128)
        self.cat_conv3 = self.conv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)
        self.cat_conv2 = self.conv_block(128, 64)

        # Output layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
    #     # Apply weight init
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
    #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #         if module.bias is not None:
    #             nn.init.constant_(module.bias, 0)
    #     elif isinstance(module, nn.BatchNorm2d):
    #         nn.init.constant_(module.weight, 1)
    #         nn.init.constant_(module.bias, 0)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting Path
        x1 = self.conv1(x)
        
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2) #/2
        x2 = self.conv2(x2)
        
        x3 = F.max_pool2d(x2, kernel_size=2, stride=2) #/2
        x3 = self.conv3(x3)

        # Expansive Path
        x2_R = self.upconv3(x3)
        x2 = torch.cat([x2, x2_R], dim=1)
        x2 = self.cat_conv3(x2)

        x1_R = self.upconv2(x2)
        x1 = torch.cat([x1, x1_R], dim=1)
        x1 = self.cat_conv2(x1)

        # Output layer
        x = self.output_layer(x1)

        return x


# from math import sqrt
# class UNet(nn.Module):
#     # class UNet(torch.jit.ScriptModule):
#     def __init__(self, colordim=3):
#         super(UNet, self).__init__()
#         self.conv1_1 = nn.Conv2d(colordim, 64, 3, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.bn1_1 = nn.BatchNorm2d(64)
#         self.bn1_2 = nn.BatchNorm2d(64)
#         self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
#         self.bn2_1 = nn.BatchNorm2d(128)
#         self.bn2_2 = nn.BatchNorm2d(128)

#         self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
#         self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
#         self.upconv4 = nn.Conv2d(256, 128, 1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.bn4_1 = nn.BatchNorm2d(256)
#         self.bn4_2 = nn.BatchNorm2d(256)
#         self.bn4_out = nn.BatchNorm2d(256)

#         self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1)
#         self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
#         self.upconv7 = nn.Conv2d(128, 64, 1)
#         self.bn7 = nn.BatchNorm2d(64)
#         self.bn7_1 = nn.BatchNorm2d(128)
#         self.bn7_2 = nn.BatchNorm2d(128)
#         self.bn7_out = nn.BatchNorm2d(128)

#         self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
#         self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.bn9_1 = nn.BatchNorm2d(64)
#         self.bn9_2 = nn.BatchNorm2d(64)
#         self.conv9_3 = nn.Conv2d(64, colordim, 1)
#         self.bn9_3 = nn.BatchNorm2d(colordim)
#         self.bn9 = nn.BatchNorm2d(colordim)
#         self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
#         self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
#         #self._initialize_weights()

#         # self.input_layer = nn.Sequential(self.conv1_1, self.bn1_1, nn.ReLU(),self.conv1_2, self.bn1_2,  nn.ReLU())
#         # self.down1 = nn.Sequential(self.conv2_1, self.bn2_1, nn.ReLU(), self.conv2_2, self.bn2_2, nn.ReLU())
#         # self.down2 = nn.Sequential(self.conv4_1, self.bn4_1, nn.ReLU(), self.conv4_2, self.bn4_2, nn.ReLU())
#         # self.up1 = nn.Sequential(self.upconv4, self.bn4)
#         # self.up2 = nn.Sequential(self.bn4_out, self.conv7_1,self.bn7_1 , nn.ReLU(), self.conv7_2, self.bn7_2, nn.ReLU())
#         # self.output = nn.Sequential(self.conv4_1, self.bn4_1, nn.ReLU(), self.conv4_2, self.bn4_2, nn.ReLU())

#     def forward(self, x1):
#         x1 = F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x1))))))
#         x2 = F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(self.maxpool(x1)))))))
#         xup = F.relu(self.bn4_2(self.conv4_2(F.relu(self.bn4_1(self.conv4_1(self.maxpool(x2)))))))
#         xup = self.bn4(self.upconv4(self.upsample(xup)))
#         xup = self.bn4_out(torch.cat((x2, xup), 1))
#         xup = F.relu(self.bn7_2(self.conv7_2(F.relu(self.bn7_1(self.conv7_1(xup))))))

#         xup = self.bn7(self.upconv7(self.upsample(xup)))
#         xup = self.bn7_out(torch.cat((x1, xup), 1))

#         xup = F.relu(self.conv9_3(F.relu(self.bn9_2(self.conv9_2(F.relu(self.bn9_1(self.conv9_1(xup))))))))

#         return torch.sigmoid(self.bn9(xup))

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


model_name = 'AE'

# Test the model
# if __name__ == "__main__":
x = torch.randn(1, 3, 512, 512).to(device)
# model = UNet().to(device)
# model = ViT_Seg(num_classes=1).to(device)
# model = SMP(num_classes=1).to(device)
model = AE().to(device)

output = model(x)
# print(output.size())  # Output size: torch.Size([1, 1, 512, 512])

device = torch.device("cuda")
table = summary(model, x)









import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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
        mask_path = os.path.join(self.mask_paths, mask_name)
            
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
    transforms.Resize((512, 512)),  # Resize to (256, 256)
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]), # norm
])

# Transforms to be applied to the data
totensor = transforms.Compose([
    # transforms.ToPILImage(),        # Convert to PIL Image
    transforms.Resize((512, 512)),  # Resize to (256, 256)
    transforms.ToTensor(),           # Convert to tensor
    # transforms.Normalize(mean=[0.5], std=[0.5]), # norm
])



# Create custom dataset and dataloader
custom_dataset = CustomDataset(image_paths, mask_paths, transform)

# Random sampling: select 10% of the dataset
dataset_size = len(custom_dataset)
sample_size = int(dataset_size * 0.1)  # 10% of the dataset
indices = random.sample(range(dataset_size), sample_size)
sampler = SubsetRandomSampler(indices)

# DataLoader with sampler
train_loader = DataLoader(custom_dataset, batch_size=32, sampler=sampler, pin_memory=True)
# train_loader = DataLoader(custom_dataset, batch_size=8, shuffle=True, pin_memory=True)

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
    
    length = len(images)

    for i in range(length):
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
test_loader = DataLoader(test_dataset, batch_size=len(test_list), shuffle=False, pin_memory=True)




import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = UNet().to(device)
# model = AE().to(device)

# criterion = nn.BCEWithLogitsLoss().to(device)
criterion = nn.MSELoss().to(device)
# criterion = nn.L1Loss().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)



import time
from tqdm import tqdm


# Define the training loop
load_weight = False

weight_file_path = f'./{model_name}_last.pt'
if os.path.exists(weight_file_path) and load_weight==True:
    # Load the weights and resume training
    model.load_state_dict(torch.load(weight_file_path))
    print("Weight loaded. Resuming training...")

    
os.makedirs(os.path.join('./', "train_results"), exist_ok=True)
os.makedirs(os.path.join('./', "test_results"), exist_ok=True)
os.makedirs(os.path.join('./', "weights"), exist_ok=True)


start_epoch = 0
num_epochs = 100
loss_list = []


start_time = time.time()  # Record the start time


for epoch in range(start_epoch, num_epochs):
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
        # area_loss = -torch.sum(masks) * 1e-7
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
    torch.save(model.state_dict(), f'./{model_name}_last.pt')
    # torch.save(model.state_dict(), f'./weights/{epoch}.pt')
    
    
    # Save some visualizations of the input images, predicted masks, and ground truth masks
    with torch.no_grad():
        # outputs = model(images)
        
        combined_image = display_result(images, masks, outputs)

        # plt.figure()
        # plt.imshow(combined_image)
        # plt.show()
        cv2.imwrite(f'./train_results/{epoch}.png', combined_image)
        
        
        for j, (images) in enumerate(test_loader):
            test_images = images.to(device)
            
            test_outputs = model(test_images)
        
            test_images = test_images.detach().cpu().permute(0, 2, 3, 1).numpy()
            test_outputs = test_outputs.detach().cpu().permute(0, 2, 3, 1).numpy()
        
            for k in range(len(images)):
                test_image = test_images[k, :, :]
                test_output = test_outputs[k, :, :, 0]
                heatmap_rgb = plt.get_cmap('jet')(test_output)[:, :, :3]  # RGB channels only
                
                test_output = cv2.merge((test_output, test_output, test_output))
        
                # Normalize and convert to uint8
                test_image = norm_image(test_image)
                test_output = norm_image(test_output)
                heatmap_rgb = norm_image(heatmap_rgb)
                heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
                # Stack the mask and output horizontally
                combined_mask_output = cv2.hconcat([test_image, test_output, heatmap_rgb])
        
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

























