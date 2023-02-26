#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 20:01:43 2021

@author: guo.1648
"""

# version 1: write my custom loss function for fc regression:
# Pearson correlation between preds and labels


import torch


def myLoss_PearsonCorr(output, target):
    x = output
    y = target.unsqueeze(1)
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    
    # newly added: in case that vx or vy are all 0:
    eps=1e-6
    if torch.sum(vx ** 2) == 0:
        vx = torch.empty(output.size()).fill_(eps).cuda()
    if torch.sum(vy ** 2) == 0:
        vy = torch.empty(target.size()).fill_(eps).cuda()
    
    PearsonCorr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    loss = 1.0 - PearsonCorr
    
    
    """
    print('*******debug*******')
    print(vx)
    print(vy)
    print(PearsonCorr)
    """
    
    
    return loss




