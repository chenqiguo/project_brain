#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:52:32 2021

@author: guo.1648
"""

# version 2: based on version 1, using more conv layers & fc layers;
# besides, using residual blocks!
# ??? also adding bottleneck ???

# referenced from model_custom_rowSizeKernel_v1.py

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


class BasicBlock(nn.Module):
    expansion: int = 1
    
    def __init__(self,
                 kernel_size: int, # row-size of the input x, e.g., 64, 128, 256...
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
        
        self.kernel_size = kernel_size # input dimension3
        self.planes = planes # output dimension3
        
        self.conv1 = conv3x3(1, planes, (1,kernel_size), stride, groups, padding) # row-size kernel
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) # should output 64 number of (2V x 1) tensors!
        # nn.MaxPool2d ???
        
        # then concatenate these 64 num of tensors horizontally to get one (2V x 64) tensor! --> permute
        
        self.conv2 = conv3x3(1, planes, (1,planes), stride, groups, padding) # row-size kernel
        self.bn2 = norm_layer(planes) # should output 64 number of (2V x 1) tensors!
        #self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x # torch.Size([batch_size, 1, 64984, self.kernel_size]) --> take planes = 64 for example
        # newly modified:
        # ??? Is this correct ???:
        if self.kernel_size != self.planes:
            assert(2*self.kernel_size == self.planes)
            identity = torch.cat((x,x), 3)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # torch.Size([batch_size, 64, 64984, 1])
        # then concatenate these 64 num of tensors horizontally to get one (2V x 64) tensor:
        out = out.permute(0, 3, 2, 1) # torch.Size([batch_size, 1, 64984, 64])
        
        out = self.conv2(out)
        out = self.bn2(out) # torch.Size([batch_size, 128, 64984, 1])
        out = out.permute(0, 3, 2, 1) # torch.Size([batch_size, 1, 64984, 128])
        
        out += identity
        out = self.relu(out) # torch.Size([batch_size, 1, 64984, 128])
        
        return out #F.normalize(out, dim=-1)
        
"""
class Bottleneck(nn.Module):
    def __init__(self,
                 inplanes: int = 1, # input channels: since we concatenate them
                 planes: int, # output channels, e.g., 64, 128, 256...
                 stride: int = 1,
                 groups: int = 1,
                 padding: int = 0, # 1?
                 kernel_size: int, # num of rows of the input x
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # add...
        
    def forward(self, x):
        identity = x # torch.Size([batch_size, 1, 64984, 64]) --> take planes = 64 for example
        
        # add...
        
        return
"""
    

    
class Model(nn.Module):
    def __init__(self,
                 block: Type[BasicBlock], #Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 row_size: int, # num of columns of the input x
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
        self.row_size = row_size
        
        self.conv1 = conv3x3(1, self.inplanes, (1,self.row_size), self.stride, self.groups, self.padding) # row-size kernel
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True) # should output 64 number of (2V x 1) tensors!
        # NO nn.MaxPool2d
        self.layer1 = self._make_layer(block, 64, self.inplanes, layers[0]) # (block, planes, kernel_size, num_blocks)
        self.layer2 = self._make_layer(block, 128, 64, layers[1])
        self.layer3 = self._make_layer(block, 256, 128, layers[2])
        self.layer4 = self._make_layer(block, 512, 256, layers[3]) # modification: How about using 64 channels kernels all the time???
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # should be avg pool on a col!!! -> or change to max pool!
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
        out = self.relu(out) # check: torch.Size([batch_size, 64, 64984, 1])
        
        # then concatenate these 64 num of tensors horizontally to get one (2V x 64) tensor:
        #out = torch.hstack(out) # deprecated!
        # use permutation to achieve this?!
        out = out.permute(0, 3, 2, 1) # check: torch.Size([batch_size, 1, 64984, 64])
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) # check: torch.Size([batch_size, 1, 64984, 512])
        
        out = out.permute(0, 3, 2, 1) # check: torch.Size([batch_size, 512, 64984, 1])
        
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
    row_size
    #**kwargs: Any
    ) -> Model:
    
    model = Model(block, layers, row_size) #, **kwargs
    
    return model


def Model_modi_resnet18(row_size) -> Model: #**kwargs: Any
    
    return _model(BasicBlock, [2, 2, 2, 2], row_size) #,**kwargs


"""
# just for debug:
if __name__ == '__main__':
    input_width = 379
    model = Model_modi_resnet18(input_width).cuda()
    
    print(model)
    
    connData = torch.from_numpy(np.array([connArray]))
    connData_ = connData.float().unsqueeze(1).cuda(non_blocking=True) # connData_.shape: torch.Size([batch_size, 1, 64984, 379])
    
    out = model(connData_)
    
    print(out.shape)
    print(out)
"""




