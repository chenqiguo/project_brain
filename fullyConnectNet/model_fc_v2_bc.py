#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:09:10 2021

@author: guo.1648
"""

# use deep fully connected network with flatten input.
# v2: only 3 layers instead of 6 layers (in v1)

# binary classification version.

# referenced from model_fc_v1_bc.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, num_cls, input_dim):
        super(Model, self).__init__()
        
        # num_cls: ==1 for regression; ==2 for binary classification.
        # input_dim == 32*32 = 1024
        k = 32
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, k) # 128 ???
        self.fc3 = nn.Linear(k, num_cls)
        self.dropout = nn.Dropout(p=0.4) #0.4
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(k)

    def forward(self, x):
                
        out = F.relu(self.bn1(self.fc1(x).squeeze(1)))      # torch.Size([B, 512])
        out = F.relu(self.bn2(self.dropout(self.fc2(out)))) # torch.Size([B, k])
        
        out = F.relu(self.fc3(out)) # torch.Size([B, 2]) ??? How about F.log_softmax(out, dim=1) ???
        
        return out






