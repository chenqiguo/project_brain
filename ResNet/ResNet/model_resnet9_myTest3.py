#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:42:19 2021

@author: guo.1648
"""

# use resnet9 architecture with larger kernel size (17).
# also use larger batch size (10) , add drop-out (???not yet???), use maxpool (???not yet???).

# referenced from https://medium.com/swlh/natural-image-classification-using-resnet9-model-6f9dc924cd6d


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        #model = resnet50(pretrained=False)
        #model.fc = nn.Linear(512, 1) # assuming that the fc7 layer has 512 neurons, otherwise change it 
        #model.cuda()
        
        model_resnet9 = ResNet(BasicBlock, [1, 1, 1, 1])
        in_features = model_resnet9.fc.in_features
        
        self.f = []
        for name, module in model_resnet9.named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=17, stride=2, padding=3, bias=False) #kernel_size=17 for others; = 7 for PCA 25
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.AdaptiveAvgPool2d):
                self.f.append(module) # but here in each layer, still kernel_size=(3, 3) <-- modify ???
        
        # our.size() here is torch.Size([bs, 512, 2031, 12])
        
        # 1st try: using AdaptiveAvgPool2d: --> USE this !!!
        self.f.append(nn.AdaptiveAvgPool2d((1, 1))) # torch.Size([bs, 512, 1, 1])
        
        # 2nd try: using MaxPool2d:
        #self.f.append(nn.MaxPool2d(kernel_size=(2031,12))) # torch.Size([bs, 512, 1, 1])
        
        # use our own (regression) top layer:
        self.f.append(nn.Flatten()) #ouput here will get torch.Size([batch_size, 512])
        self.f.append(nn.Linear(in_features, 1)) # in_features == 512
        
        self.f = nn.Sequential(*self.f)
        

    def forward(self, x):
        out = self.f(x)
        return out #F.normalize(out, dim=-1)



"""
def conv_block(in_channels, out_channels, pool=False, pool_no=2): # pool_no=12
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # kernel_size=18
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)


class Model(nn.Module): # ResNet9 # CUDA out of memory!!!
    def __init__(self):
        super(Model, self).__init__()
        
        in_channels = 1
        num_classes = 1 # for regression!!!
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True, pool_no=3) # pool_no=18
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True, pool_no=5) # pool_no=30
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.regresser = nn.Sequential(nn.MaxPool2d(5), # 30
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.regresser(out)
        return out
"""


# for debug:
#model = Model()


# just for debug:
if __name__ == '__main__':
    #input_width = 379
    #input_height = 64984
    model = Model().cuda()
    
    print(model)
    
    import scipy.io as sio
    import numpy as np
    
    mat_contents = sio.loadmat('/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/100206.mat')
    connL = mat_contents['connL'] # (32492, 379)
    connR = mat_contents['connR'] # (32492, 379)
    connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
    connData = torch.from_numpy(np.array([connArray]))
    connData_ = connData.float().unsqueeze(1).cuda(non_blocking=True) # connData_.shape: torch.Size([batch_size, 1, 64984, 379])
    
    out = model(connData_)
    
    print(out.shape)
    print(out)
    

