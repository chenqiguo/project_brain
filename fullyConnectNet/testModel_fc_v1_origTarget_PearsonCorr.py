#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:21:17 2021

@author: guo.1648
"""

# for testing:
# brain research project: deep fc network version1:
# deep fully connected network with flatten 2PCA connData as input.


import argparse
import os
import numpy as np
import math
import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils_PCAflat.ModelNetDataLoader_2PCAflat_origTarget import ModelNetDataLoader_PCAflat
from model_fc_v1 import Model
from myLoss_PearsonCorr_v1 import myLoss_PearsonCorr

#from PIL import Image
from torchvision import transforms, datasets





@torch.no_grad()
def evaluate(model, val_loader, tolerance):
    model.eval()
    outputs = [validation_step(batch, tolerance, model) for batch in val_loader]
    return validation_epoch_end(outputs)


"""
# No need to compute:
def accuracy(outputs, labels, tolerance): # newly modified by Chenqi:
    
    # Fabian's ver for classification:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
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
    print('********* out:') # only for debug in testing
    print(out)
    #"""
    
    loss = myLoss_PearsonCorr(out, labels.cuda())
    
    # newly modified: also get training PearsonCorr:
    PearsonCorr = 1.0 - loss
    
    return {'val_loss': loss.detach(), 'val_corr': PearsonCorr}


def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_corrs = [x['val_corr'] for x in outputs]
    epoch_corr = torch.stack(batch_corrs).mean()      # Combine correlations
    return {'val_loss': epoch_loss.item(), 'val_corr': epoch_corr.item()}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet9 Regress w. PCA Data, FabianVer')
    parser.add_argument('--model_path', type=str, default='results/fc_v1_origTarget_allSubjects_lossPearsonCorr_v1/model_weights_bestValidLoss.pth',help='The pretrained model path')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl', type=str, help='Dir of the PCA dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    #parser.add_argument('--train_split', default=0.7, type=float, help='Percent of training set')
    #parser.add_argument('--tolerance', default=2, type=float, help='tolerance of difference between abs(gt-pred)')
    parser.add_argument('--result_dir', default='results/fc_v1_origTarget_allSubjects_lossPearsonCorr_v1/', type=str, help='Dir of the results')
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
    model = Model(num_cls=1, input_dim=1024).cuda() # for regression!
    model.load_state_dict(torch.load(model_path))
    """
    for param in model.f.parameters():
        param.requires_grad = False
    """
    
    tolerance = None # NOT used dummy value!
    result = evaluate(model, test_loader, tolerance)
    print('@@@@@@@@@@@@@@@@@@@@@@ result:')
    print(result)
    
    # save testing result to pk:
    f_pkl = open(result_dir+'test_result.pkl', 'wb')
    pickle.dump(result,f_pkl)
    f_pkl.close()
    
    
