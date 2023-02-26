#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:18:14 2021

@author: guo.1648
"""


# referenced from resnetPred_myTest2.py

# instead of using resnet, here we construct our own model architecture:
# using (1) row-size kernels or (2) column-size kernels for convolution.

# also, for the target DSM_Anxi_T values, since they are very biased to 50,
# we use log(x-49) instead of the original values to do regression!


import argparse
import os
import numpy as np
import math

import pandas as pd
import torch
import torch.optim as optim
#from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

#import utils_myTest2_origTarget
from data_utils.ModelNetDataLoader_v2_origTarget import ModelNetDataLoader
#from model_custom_rowSizeKernel_v1 import Model # NOT USE!!!
#from model_custom_rowSizeKernel_v2 import Model_modi_resnet18
#from model_custom_colSizeKernel_v2 import Model_modi_resnet18

#from PIL import Image
from torchvision import transforms, datasets

import importlib



# train for one epoch
def train(net, data_loader, train_optimizer, criterion):
    net.train()
    total_loss, total_num, total_correctNum, train_bar = 0.0, 0, 0, tqdm(data_loader)
    
    for connData, target, target_mat_name in train_bar: # target.shape: torch.Size([batch_size])
        """
        # for batch_size ==1: check if target exists (not Nan):
        if batch_size ==1 and math.isnan(target.detach().numpy()[0]):
            #print('Mat file target is Nan: ' + target_mat_name[0])
            continue
        """
        connData_ = connData.float().unsqueeze(1).cuda(non_blocking=True) # connData_.shape: torch.Size([batch_size, 1, 64984, 379])
        out = net(connData_) #torch.Size([batch_size, 1])
        target_ = target.float().unsqueeze(1) # target_.shape: torch.Size([batch_size, 1])
        
        # compute loss:
        loss = criterion(out, target_.cuda())
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
        total_num += batch_size
        total_loss += loss.item() * batch_size
        
        # for acc:
        difference = (out.detach().cpu() - target_).numpy() # difference.shape: (batch_size, 1)
        difference = np.absolute(difference)
        total_correctNum += np.count_nonzero(difference <= tolerance)
        
        train_bar.set_description('Train Epoch: [{}/{}] Train Loss: {:.4f} Train Acc: {:.4f}%'.format(epoch, epochs, total_loss / total_num, total_correctNum*100 / total_num))

    train_loss = total_loss / total_num
    train_acc = total_correctNum / total_num
    
    return (train_loss, train_acc)


def test(net, data_loader, criterion):
    net.eval()
    total_loss, total_num, total_correctNum, test_bar = 0.0, 0, 0, tqdm(data_loader, desc='testing on valid set')
    
    with torch.no_grad():
        
        for connData, target, target_mat_name in test_bar:
            """
            # for batch_size ==1: check if target exists (not Nan):
            if batch_size ==1 and math.isnan(target.detach().numpy()[0]):
                #print('Mat file target is Nan: ' + target_mat_name[0])
                continue
            """
            connData_ = connData.float().unsqueeze(1).cuda(non_blocking=True) # connData_.shape: torch.Size([batch_size, 1, 64984, 379])
            out = net(connData_) #torch.Size([batch_size, 1])
            target_ = target.float().unsqueeze(1) # target_.shape: torch.Size([batch_size, 1])
            
            # compute loss:
            loss = criterion(out, target_.cuda())
            total_num += connData_.size(0)
            total_loss += loss.item() * connData_.size(0)
            # for acc:
            difference = (out.detach().cpu() - target_).numpy() # difference.shape: (batch_size, 1)
            difference = np.absolute(difference)
            total_correctNum += np.count_nonzero(difference <= tolerance)
            
            test_bar.set_description('Test Epoch: [{}/{}] Test Loss: {:.4f} Test Acc: {:.4f}%'.format(epoch, epochs, total_loss / total_num, total_correctNum*100 / total_num))
            
    valid_loss = total_loss / total_num
    valid_acc = total_correctNum / total_num
    
    return (valid_loss, valid_acc)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Custom Model Regress v1')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/', type=str, help='Dir of the mat files dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    #parser.add_argument('--train_split', default=0.7, type=float, help='Percent of training set')
    parser.add_argument('--tolerance', default=2, type=float, help='tolerance of difference between abs(gt-pred)') # 2 for target_list_orig
    parser.add_argument('--result_rootDir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/col_row_size_kernel_resnet/results_customModel_origTarget/', type=str, help='Root dir of the results')
    parser.add_argument('--input_height', default=64984, type=int, help='Height of input image. Used in col-size kernels. Default is for conn')
    parser.add_argument('--input_width', default=379, type=int, help='Width of input image. Used in row-size kernels. Default is for conn')
    parser.add_argument('--model_type', default='rowKernel', type=str, help='Which kind of custom model to use: rowKernel or colKernel or rowcolKernel')
    parser.add_argument('--model_version', default='v2', type=str, help='Which version of custom model to use')
    
    # newly added:
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    
    # args parse
    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epochs
    data_dir = args.data_dir
    label_name = args.label_name
    #train_split = args.train_split
    tolerance = args.tolerance
    result_rootDir = args.result_rootDir
    input_height = args.input_height
    input_width = args.input_width
    model_type = args.model_type
    train_test_root = args.train_test_root
    model_version = args.model_version
    
    # model setup: newly modified:
    if model_type == 'rowKernel':
        kernel_len = input_width
        model_custom_import = importlib.import_module('model_custom_rowSizeKernel_'+model_version)
        model = model_custom_import.Model_modi_resnet9(kernel_len).cuda() #Model_modi_resnet18
        verFolder = '/'+model_version+'/'
    elif model_type == 'colKernel':
        kernel_len = input_height
        model_custom_import = importlib.import_module('model_custom_colSizeKernel_'+model_version)
        model = model_custom_import.Model_modi_resnet18(kernel_len).cuda()
        verFolder = '/'+model_version+'/'
    elif model_type == 'rowcolKernel':
        model_custom_import = importlib.import_module('model_custom_rowcolSizeKernel_'+model_version)
        model = model_custom_import.Model_modi_resnet9(input_width, input_height).cuda() #Model_modi_resnet18
        verFolder = '/'+model_version+'/'
    
    # NO transforms!
    
    # data prepare: newly modified:
    train_dataset = ModelNetDataLoader(root=data_dir, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=data_dir, args=args, split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    # optimizer config
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    
    criterion = torch.nn.L1Loss() #torch.nn.MSELoss()
    
    # training loop:
    results = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    save_name_pre = model_type + '_{}_{}'.format(batch_size, epochs) # only save the three: latest & the best (lowest valid_loss & highest valid_acc) models!
    best_validLoss = np.float('inf')
    best_validAcc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        valid_loss, valid_acc = test(model, valid_loader, criterion)
        results['valid_loss'].append(valid_loss)
        results['valid_acc'].append(valid_acc)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # newly modified:
        data_frame.to_csv(result_rootDir+model_type+verFolder+'{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        
        
        # save the "best" model:
        if valid_loss <= best_validLoss:
            best_validLoss = valid_loss
            torch.save(model.state_dict(), result_rootDir+model_type+verFolder+'bestLoss_{}_model.pth'.format(save_name_pre))
        if valid_acc >= best_validAcc:
            best_validAcc = valid_acc
            torch.save(model.state_dict(), result_rootDir+model_type+verFolder+'bestAcc_{}_model.pth'.format(save_name_pre))
        
        # keep saving the latest mode:
        torch.save(model.state_dict(), result_rootDir+model_type+verFolder+'latest_{}_model.pth'.format(save_name_pre))
        
        # NOT USED (to save space): save all the models during training:
        #torch.save(model.state_dict(), 'results_v2/' + dataset + '/' + 'epoch{}'.format(epoch) + '_{}_model.pth'.format(save_name_pre))
        

