#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:37:13 2021

@author: guo.1648
"""

# for testing:
# brain research project: deep fc network version2:
# deep fully connected network with flatten 2PCA connData as input.


import argparse
import os
import numpy as np
import math
import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils_PCAflat.ModelNetDataLoader_2PCAflat_bc import ModelNetDataLoader_PCAflat
from model_fc_v2_bc import Model

#from PIL import Image
from torchvision import transforms, datasets


@torch.no_grad()
def evaluate(model, val_loader, tolerance):
    model.eval()
    outputs = [validation_step(batch, tolerance, model) for batch in val_loader]
    return validation_epoch_end(outputs)


def accuracy(outputs, labels): # newly modified by Chenqi:
    
    # Fabian's ver for classification:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels.cuda()).item() / len(preds))
    """
    # Chenqi's ver for regression:
    difference = (outputs.detach().cpu() - labels.unsqueeze(1).detach().cpu()).numpy() # difference.shape: (batch_size, 1)
    difference = np.absolute(difference)
    return torch.tensor(np.count_nonzero(difference <= tolerance) / len(outputs))
    """


def validation_step(batch, tolerance, model):
    images, labels, _ = batch 
    images = images.float().unsqueeze(1).cuda() # torch.Size([batch_size, 1, 32, 32])
    
    #labels = labels.cuda()
    out = model(images)                    # Generate predictions  torch.Size([batch_size, 1])
    
    #"""
    print('********* preds:') # only for debug in testing
    _, preds = torch.max(out, dim=1)
    print(preds)
    #"""
    
    loss = F.cross_entropy(out, labels.cuda())   # Calculate loss <-- Fabian's ver for classification
    
    acc = accuracy(out, labels)           # Calculate accuracy
    
    return {'val_loss': loss.detach(), 'val_acc': acc}


def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet9 Regress w. PCA Data, FabianVer')
    parser.add_argument('--model_path', type=str, default='results/fc_v1_bc_allSubjects_thresh50/model_weights_bestValidLoss.pth',help='The pretrained model path')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl', type=str, help='Dir of the PCA dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    #parser.add_argument('--train_split', default=0.7, type=float, help='Percent of training set')
    #parser.add_argument('--tolerance', default=2, type=float, help='tolerance of difference between abs(gt-pred)')
    parser.add_argument('--result_dir', default='results/fc_v1_bc_allSubjects_thresh50/', type=str, help='Dir of the results')
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    
    # args parse
    args = parser.parse_args()
    model_path = args.model_path
    batch_size = args.batch_size
    data_dir = args.data_dir
    label_name = args.label_name
    #train_split = args.train_split
    #tolerance = args.tolerance
    result_dir = args.result_dir
    train_test_root = args.train_test_root
    
    # NO transforms!
    
    # data prepare: newly modified:
    test_dataset = ModelNetDataLoader_PCAflat(args=args, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    # load model:
    model = Model(num_cls=2, input_dim=1024).cuda() # for binary classification!
    model.load_state_dict(torch.load(model_path))
    """
    for param in model.f.parameters():
        param.requires_grad = False
    """
    
    tolerance = None # dummy val: do NOT need it in bc !!!
    result = evaluate(model, test_loader, tolerance)
    print('@@@@@@@@@@@@@@@@@@@@@@ result:')
    print(result)
    
    # save testing result to pk:
    f_pkl = open(result_dir+'test_result.pkl', 'wb')
    pickle.dump(result,f_pkl)
    f_pkl.close()
    









