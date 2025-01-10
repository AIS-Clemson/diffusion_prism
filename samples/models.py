# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:12:23 2024

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


# import segmentation_models_pytorch as smp

# aux_params=dict(
#     # pooling='avg',             # one of 'avg', 'max'
#     dropout=0.5,               # dropout ratio, default is None
#     # activation='sigmoid',      # activation function, default is None
#     classes=1,                 # define number of output labels
# )

# model = smp.Unet(
#     # encoder_name="resnet50",        # backbone
#     encoder_name="resnet34",        # backbone
#     encoder_weights="imagenet",     # load ImageNet as weight
#     in_channels=3,                  # input channel
#     classes=1,                      # output channel
#     # aux_params={"dropout": 0.1}  # Use auxiliary output with 3 classes
#     aux_params=aux_params
# ).to(device)


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


from math import sqrt
class UNet_tiny(nn.Module):
    # class UNet(torch.jit.ScriptModule):
    def __init__(self, colordim=3):
        super(UNet_tiny, self).__init__()
        self.conv1_1 = nn.Conv2d(colordim, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.upconv4 = nn.Conv2d(256, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.bn4_out = nn.BatchNorm2d(256)

        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.upconv7 = nn.Conv2d(128, 64, 1)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.bn7_2 = nn.BatchNorm2d(128)
        self.bn7_out = nn.BatchNorm2d(128)

        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.bn9_2 = nn.BatchNorm2d(64)
        self.conv9_3 = nn.Conv2d(64, colordim, 1)
        self.bn9_3 = nn.BatchNorm2d(colordim)
        self.bn9 = nn.BatchNorm2d(colordim)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        #self._initialize_weights()

        # self.input_layer = nn.Sequential(self.conv1_1, self.bn1_1, nn.ReLU(),self.conv1_2, self.bn1_2,  nn.ReLU())
        # self.down1 = nn.Sequential(self.conv2_1, self.bn2_1, nn.ReLU(), self.conv2_2, self.bn2_2, nn.ReLU())
        # self.down2 = nn.Sequential(self.conv4_1, self.bn4_1, nn.ReLU(), self.conv4_2, self.bn4_2, nn.ReLU())
        # self.up1 = nn.Sequential(self.upconv4, self.bn4)
        # self.up2 = nn.Sequential(self.bn4_out, self.conv7_1,self.bn7_1 , nn.ReLU(), self.conv7_2, self.bn7_2, nn.ReLU())
        # self.output = nn.Sequential(self.conv4_1, self.bn4_1, nn.ReLU(), self.conv4_2, self.bn4_2, nn.ReLU())

    def forward(self, x1):
        x1 = F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x1))))))
        x2 = F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(self.maxpool(x1)))))))
        xup = F.relu(self.bn4_2(self.conv4_2(F.relu(self.bn4_1(self.conv4_1(self.maxpool(x2)))))))
        xup = self.bn4(self.upconv4(self.upsample(xup)))
        xup = self.bn4_out(torch.cat((x2, xup), 1))
        xup = F.relu(self.bn7_2(self.conv7_2(F.relu(self.bn7_1(self.conv7_1(xup))))))

        xup = self.bn7(self.upconv7(self.upsample(xup)))
        xup = self.bn7_out(torch.cat((x1, xup), 1))

        xup = F.relu(self.conv9_3(F.relu(self.bn9_2(self.conv9_2(F.relu(self.bn9_1(self.conv9_1(xup))))))))

        return torch.sigmoid(self.bn9(xup))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# model_name = 'AE'

# # Test the model
# # if __name__ == "__main__":
# x = torch.randn(1, 3, 512, 512).to(device)
# # model = UNet().to(device)
# # model = ViT_Seg(num_classes=1).to(device)
# # model = SMP(num_classes=1).to(device)
# model = AE().to(device)

# output = model(x)
# # print(output.size())  # Output size: torch.Size([1, 1, 512, 512])

# device = torch.device("cuda")
# table = summary(model, x)

