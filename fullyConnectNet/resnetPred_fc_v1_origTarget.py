#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:46:02 2021

@author: guo.1648
"""

# for training:
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

#from PIL import Image
from torchvision import transforms, datasets




@torch.no_grad()
def evaluate(model, val_loader, tolerance):
    model.eval()
    outputs = [validation_step(batch, tolerance, model) for batch in val_loader]
    return validation_epoch_end(outputs)



def accuracy(outputs, labels, tolerance): # newly modified by Chenqi:
    """
    # Fabian's ver for classification:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    """
    # Chenqi's ver for regression:
    difference = (outputs.detach().cpu() - labels.unsqueeze(1).detach().cpu()).numpy() # difference.shape: (batch_size, 1)
    difference = np.absolute(difference)
        
    return torch.tensor(np.count_nonzero(difference <= tolerance) / len(outputs))


def training_step(batch, tolerance, model):
    images, labels, _ = batch 
    images = images.float().unsqueeze(1).cuda() # torch.Size([batch_size, 1024])
    
    out = model(images) #torch.Size([batch_size, 1])
    
    criterion = torch.nn.L1Loss() # ??? is it suitable ???
    loss = criterion(out, labels.cuda())
    
    acc = accuracy(out, labels, tolerance)           # Calculate accuracy
    
    return (loss, acc)


def validation_step(batch, tolerance, model):
    images, labels, _ = batch 
    images = images.float().unsqueeze(1).cuda() # torch.Size([batch_size, 1, 32, 32])
    
    #labels = labels.cuda()
    out = model(images)                    # Generate predictions  torch.Size([batch_size, 1])
    
    """
    print('********* out:') # only for debug in testing
    print(out)
    """
    
    #loss = F.cross_entropy(out, labels)   # Calculate loss <-- Fabian's ver for classification
    criterion = torch.nn.L1Loss() # <-- Chenqi's ver for regression ??? is it suitable ???
    loss = criterion(out, labels.cuda())
    
    acc = accuracy(out, labels, tolerance)           # Calculate accuracy
    
    return {'val_loss': loss.detach(), 'val_acc': acc}


def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


def epoch_end(epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def fit_one_cycle(epochs, tolerance, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    best_validLoss = np.float('inf')
    best_validAcc = 0
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        train_accs = []
        lrs = []
        k1 = 0
        
        train_bar = tqdm(train_loader)
        
        for batch in train_bar:
            # Note: batch = (dataMat_feat_flat, target, target_mat_name)
            loss, acc = training_step(batch, tolerance, model)
            train_losses.append(loss)
            train_accs.append(acc)
            
            loss.backward()
            
            if k1 % 100 == 0:
                print(f'batch {k1}')  
            k1 += 1
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader, tolerance)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_accs).mean().item()
        result['lrs'] = lrs
        epoch_end(epoch, result)
        history.append(result)
        
        # save "best" models:
        if result['val_loss'] <= best_validLoss:
            best_validLoss = result['val_loss']
            torch.save(model.state_dict(), result_dir+'model_weights_bestValidLoss.pth')
        if result['val_acc'] >= best_validAcc:
            best_validAcc = result['val_acc']
            torch.save(model.state_dict(), result_dir+'model_weights_bestValidAcc.pth')
        
    return history



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deep FC Regress w. PCA Flatten Data, v1')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl', type=str, help='Dir of the PCA dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    #parser.add_argument('--train_split', default=0.7, type=float, help='Percent of training set')
    parser.add_argument('--tolerance', default=2, type=float, help='tolerance of difference between abs(gt-pred)')
    parser.add_argument('--result_dir', default='results/fc_v1_origTarget/', type=str, help='Dir of the results')
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    
    # args parse
    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epochs
    data_dir = args.data_dir
    label_name = args.label_name
    #train_split = args.train_split
    tolerance = args.tolerance
    result_dir = args.result_dir
    train_test_root = args.train_test_root
    
    # NO transforms!
    
    # data prepare: newly modified:
    train_dataset = ModelNetDataLoader_PCAflat(args=args, split='train')
    test_dataset = ModelNetDataLoader_PCAflat(args=args, split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    # model setup:
    model = Model(num_cls=1, input_dim=1024).cuda() # for regression!
    
    # referenced from resnetPred_origTarget_FabianVer.py:
    # training loop:
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    
    history = fit_one_cycle(epochs, tolerance, max_lr, model, train_loader, valid_loader, 
                            grad_clip=grad_clip, 
                            weight_decay=weight_decay, 
                            opt_func=opt_func)
    
    torch.save(model.state_dict(), result_dir+'model_weights_latest.pth')
    
    # save training history to pk:
    f_pkl = open(result_dir+'history.pkl', 'wb')
    pickle.dump(history,f_pkl)
    f_pkl.close()
    
    #print()
    
    
    

