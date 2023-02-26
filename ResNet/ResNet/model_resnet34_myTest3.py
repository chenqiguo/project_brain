#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:42:19 2021

@author: guo.1648
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        #model = resnet50(pretrained=False)
        #model.fc = nn.Linear(512, 1) # assuming that the fc7 layer has 512 neurons, otherwise change it 
        #model.cuda()
        
        self.f = []
        for name, module in resnet34(pretrained=False).named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if not isinstance(module, nn.Linear): #and not isinstance(module, nn.MaxPool2d)
                self.f.append(module)
        
        # use our own (regression) top layer:
        self.f.append(nn.Flatten()) #ouput here will get torch.Size([batch_size, 512])
        self.f.append(nn.Linear(512, 1))
        
        self.f = nn.Sequential(*self.f)
        

    def forward(self, x):
        out = self.f(x)
        return out #F.normalize(out, dim=-1)






