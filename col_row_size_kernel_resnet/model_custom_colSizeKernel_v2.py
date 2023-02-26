#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 07:41:51 2021

@author: guo.1648
"""

# referenced from model_custom_rowSizeKernel_v2.py

# using residual blocks & column-size kernels for convolution!


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


class BasicBlock(nn.Module):
    expansion: int = 1
    
    def __init__(self,
                 kernel_size: int, # column-size of the input x, e.g., 64, 128, 256...
                 planes: int, # output channels, e.g., 64, 128, 256...
                 inplanes: int = 1, # input channels: since we concatenate them
                 stride: int = 1,
                 groups: int = 1,
                 padding: int = 0, # 1?
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.kernel_size = kernel_size # input dimension2
        self.planes = planes # output dimension2
        
        self.conv1 = conv3x3(1, planes, (kernel_size,1), stride, groups, padding) # column-size kernel
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) # should output 64 number of (1 x p) tensors!
        # nn.MaxPool2d ???
        
        # then concatenate these 64 num of tensors vertically to get one (64 x p) tensor! --> permute
        
        self.conv2 = conv3x3(1, planes, (planes,1), stride, groups, padding) # row-size kernel
        self.bn2 = norm_layer(planes) # should output 64 number of (1 x p) tensors!
        #self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x # torch.Size([batch_size, 1, self.kernel_size, 379]) --> take planes = 64 for example
        # newly modified:
        # ??? Is this correct ???:
        if self.kernel_size != self.planes:
            assert(2*self.kernel_size == self.planes)
            identity = torch.cat((x,x), 2)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # torch.Size([batch_size, 64, 1, 379])
        # then concatenate these 64 num of tensors vertically to get one (64 x p) tensor:
        out = out.permute(0, 2, 1, 3) # torch.Size([batch_size, 1, 64, 379])
        
        out = self.conv2(out)
        out = self.bn2(out) # torch.Size([batch_size, 128, 1, 379])
        out = out.permute(0, 2, 1, 3) # torch.Size([batch_size, 1, 128, 379])
        
        out += identity
        out = self.relu(out) # torch.Size([batch_size, 1, 128, 379])
        
        return out #F.normalize(out, dim=-1)


class Model(nn.Module):
    def __init__(self,
                 block: Type[BasicBlock], #Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 col_size: int, # num of rows of the input x
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(Model, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.stride = 1
        self.padding = 0 # 1?
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.col_size = col_size
        
        self.conv1 = conv3x3(1, self.inplanes, (self.col_size,1), self.stride, self.groups, self.padding) # col-size kernel
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True) # should output 64 number of (1 x p) tensors!
        # NO nn.MaxPool2d
        self.layer1 = self._make_layer(block, 64, self.inplanes, layers[0]) # (block, planes, kernel_size, num_blocks)
        self.layer2 = self._make_layer(block, 128, 64, layers[1])
        self.layer3 = self._make_layer(block, 256, 128, layers[2])
        self.layer4 = self._make_layer(block, 512, 256, layers[3]) # modification: How about using 64 channels kernels all the time???
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() # check: should output torch.Size([batch_size, 512])
        self.fc = nn.Linear(512 * block.expansion, 1) # for regression!!!
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
               
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                """
                elif isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                """
                
    def _make_layer(self, block: Type[BasicBlock], planes: int, kernel_size: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential: #block: Type[Union[BasicBlock, Bottleneck]]
        norm_layer = self._norm_layer
        
        layers = []
        layers.append(block(kernel_size, planes, 1, stride, self.groups, self.padding, norm_layer))
        
        for _ in range(1, blocks):
            layers.append(block(planes, planes, 1, stride, self.groups, self.padding, norm_layer))
            
        return nn.Sequential(*layers)
        
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # check: torch.Size([batch_size, 64, 1, 379])
        
        # then concatenate these 64 num of tensors vvertically to get one (64 x p) tensor:
        out = out.permute(0, 2, 1, 3) # check: torch.Size([batch_size, 1, 64, 379])
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) # check: torch.Size([batch_size, 1, 512, 379])
        
        out = out.permute(0, 2, 1, 3) # check: torch.Size([batch_size, 512, 1, 379])
        
        out = self.avgpool(out) # check: torch.Size([batch_size, 512, 1, 1])
        out = self.flatten(out) # check: torch.Size([batch_size, 512])
        
        # ???bunch??? of fc layers:
        out = self.fc(out) # torch.Size([batch_size, 1])
        
        return out #F.normalize(out, dim=-1)
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _model(
    block: Type[BasicBlock], #Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    col_size
    #**kwargs: Any
    ) -> Model:
    
    model = Model(block, layers, col_size) #, **kwargs
    
    return model

        
def Model_modi_resnet18(col_size) -> Model: #**kwargs: Any
    
    return _model(BasicBlock, [2, 2, 2, 2], col_size) #,**kwargs



"""
# just for debug:
if __name__ == '__main__':
    input_height = 64984
    model = Model_modi_resnet18(input_height).cuda()
    
    print(model)
    
    import numpy as np
    import scipy.io as sio
    
    fullFileName = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/100206.mat' # full name of the mat file
    mat_contents = sio.loadmat(fullFileName)
    connL = mat_contents['connL']
    connR = mat_contents['connR']
    connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
    
    connData = torch.from_numpy(np.array([connArray]))
    connData_ = connData.float().unsqueeze(1).cuda(non_blocking=True) # connData_.shape: torch.Size([batch_size, 1, 64984, 379])
    
    out = model(connData_)
    
    print(out.shape)
    print(out)
"""


