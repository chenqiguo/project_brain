#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:46:02 2021

@author: guo.1648
"""

# brain research project: my 3rd try: based on /NotUsed/ 2nd try ,
# using concatenated connL&connR data as input ('image'), DSM_Anxi_T as output,
# AND pre-defined train(70%) and test(30%) a ResNet regression (modify the top fc layer),
# AND log target metric.

# Q: should I treat connL & connR seperately using two networks????


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

#from data_utils.ModelNetDataLoader_v2_origTarget import ModelNetDataLoader
from data_utils.ModelNetDataLoader_PCA_origTarget import ModelNetDataLoader
#from model_resnet34_myTest3 import Model
from model_resnet9_myTest3 import Model

#from PIL import Image
from torchvision import transforms, datasets



# train for one epoch
def train(net, data_loader, train_optimizer, criterion):
    net.train()
    total_loss, total_num, total_correctNum, train_bar = 0.0, 0, 0, tqdm(data_loader)
    
    for connData, target, target_mat_name in train_bar: # target.size: torch.Size([batch_size])
        # for batch_size ==1: check if target exists (not Nan):
        
        """
        # already moved those nanTarget mat files to another dir!!!
        if batch_size ==1 and math.isnan(target.detach().numpy()[0]):
            #print('Mat file target is Nan: ' + target_mat_name[0])
            continue
        """
        
        connData_ = connData.float().unsqueeze(1).cuda(non_blocking=True) # connData_.shape: torch.Size([batch_size, 1, 64984, new_featDim])
        
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
    parser = argparse.ArgumentParser(description='Train ResNet Regress myTest3')
    parser.add_argument('--batch_size', default=2, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/', type=str, help='Dir of the mat files dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    #parser.add_argument('--train_split', default=0.7, type=float, help='Percent of training set')
    parser.add_argument('--tolerance', default=2, type=float, help='tolerance of difference between abs(gt-pred)')
    #parser.add_argument('--result_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/', type=str, help='Dir of the mat files dataset')
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    
    # newly added for PCA versions: 50 for doing PCA; 379 for NOT doing PCA
    parser.add_argument('--new_featDim', default=50, type=int,  help='Number of feature dimensions after PCA reduction')
    
    # args parse
    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epochs
    data_dir = args.data_dir
    label_name = args.label_name
    #train_split = args.train_split
    tolerance = args.tolerance
    #result_dir = args.result_dir
    train_test_root = args.train_test_root
    new_featDim = args.new_featDim
    
    # NO transforms!
    
    # data prepare: newly modified:
    train_dataset = ModelNetDataLoader(root=data_dir, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=data_dir, args=args, split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    # model setup and optimizer config
    model = Model().cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6) # how about weight_decay=1e-4 ???
    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.L1Loss() #torch.nn.MSELoss()
    
    # training loop:
    results = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    save_name_pre = '{}_{}'.format(batch_size, epochs) # only save the two: latest & the best (lowest valid_loss) models!
    best_validLoss = np.float('inf')
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
        data_frame.to_csv('results_myTest3_origTarget/model_resnet9_myTest3_origTarget/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        
        #"""
        # USED code: only save the "best" model:
        if valid_loss <= best_validLoss:
            best_validLoss = valid_loss
            torch.save(model.state_dict(), 'results_myTest3_origTarget/model_resnet9_myTest3_origTarget/best_{}_model.pth'.format(save_name_pre))
        #"""
        """
        # NOT USED: save all the models!!! (while also keep track on the "best" model):
        torch.save(model.state_dict(), 'results_v2/' + dataset + '/' + 'epoch{}'.format(epoch) + '_{}_model.pth'.format(save_name_pre))
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            # newly modified:
            torch.save(model.state_dict(), 'results_v2/' + dataset + '/' + 'best_{}_model.pth'.format(save_name_pre))
        """
    
    

