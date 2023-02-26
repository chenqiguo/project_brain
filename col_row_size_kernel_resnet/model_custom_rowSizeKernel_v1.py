#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:56:41 2021

@author: guo.1648
"""

# version 1: NOT used! The test_acc does not improve!

# referenced from model_resnet34_myTest2.py

# instead of using resnet, here we construct our own model architecture:
# using row-size kernels for convolution.

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.resnet import resnet34

from typing import Type, Any, Callable, Union, List, Optional



def conv3x3(in_planes, out_planes, kernelSize, stride, groups, padding) -> nn.Conv2d:
    """3x3 convolution with or without padding"""
    return nn.Conv2d(in_planes, out_planes, kernelSize, stride,
                     padding=padding, groups=groups, bias=False, dilation=1)



class Model(nn.Module):
    def __init__(self,
                 row_size: int, # num of rows of the input x
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(Model, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.row_size = row_size
        self.stride = 1
        self.groups = 1
        self.planes = 64 # m
        self.padding = 0 # 1?
        
        self.conv1 = conv3x3(1, self.planes, (1,self.row_size), self.stride, self.groups, self.padding) # row-size kernel
        self.bn1 = norm_layer(self.planes)
        self.relu = nn.ReLU(inplace=True) # should output 64 number of (2V x 1) tensors!
        # nn.MaxPool2d ???
        
        # then concatenate these 64 num of tensors horizontally to get one (2V x 64) tensor!
        
        self.conv2 = conv3x3(1, self.planes, (1,self.planes), self.stride, self.groups, self.padding) # row-size kernel
        self.bn2 = norm_layer(self.planes) # should output 64 number of (2V x 1) tensors!
        #self.relu = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() # should output torch.Size([batch_size, 64])
        
        # a bunch of fully connected layers!!!:
        #self.linear = nn.Linear(self.planes, 1) # original code
        self.fc1 = nn.Linear(self.planes, 32)
        self.fc2 = nn.Linear(32, 1)
        #self.fc3 = nn.Linear(16, 8)
        #self.fc4 = nn.Linear(8, 1)
        

    def forward(self, x):
        #out = self.f(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # torch.Size([batch_size, 64, 64984, 1])
        
        # then concatenate these 64 num of tensors horizontally to get one (2V x 64) tensor:
        #out = torch.hstack(out) # deprecated!
        # use permutation to achieve this?!
        out = out.permute(0, 3, 2, 1) # torch.Size([batch_size, 1, 64984, 64])
        
        out = self.conv2(out)
        out = self.bn2(out) # torch.Size([batch_size, 64, 64984, 1])
        
        out = self.avgpool(out) # torch.Size([batch_size, 64, 1, 1])
        out = self.flatten(out) # torch.Size([batch_size, 64])
        
        # bunch of fc layers:
        #out = self.linear(out) # torch.Size([batch_size, 1]) # original code
        out = self.fc1(out)
        out = self.fc2(out)
        #out = self.fc3(out)
        #out = self.fc4(out) # torch.Size([batch_size, 1])
        
        return out #F.normalize(out, dim=-1)




"""
# just for debug:
if __name__ == '__main__':
    input_width = 379
    model = Model(input_width).cuda()
    
    print(model)
    
    connData = torch.from_numpy(np.array([connArray]))
    connData_ = connData.float().unsqueeze(1).cuda(non_blocking=True) # connData_.shape: torch.Size([batch_size, 1, 64984, 379])
    
    out = model(connData_)
    
    print(out.shape)
"""

