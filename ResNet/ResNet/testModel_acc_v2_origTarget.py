#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:21:17 2021

@author: guo.1648
"""

# version 1: code to test accuracy with the trained model.
# for checking: just use the whole dataset to test! <-- need modification!


import argparse
import os
import numpy as np
import math

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# for model trained in resnetPred_myTest3.py:
#from data_utils.ModelNetDataLoader_v2_origTarget import ModelNetDataLoader
from data_utils.ModelNetDataLoader_PCA_origTarget import ModelNetDataLoader
#from model_resnet34_myTest3 import Model
from model_resnet9_myTest3 import Model

#from PIL import Image
from torchvision import transforms, datasets
import torch.nn as nn




class testModel_Net(nn.Module):
    def __init__(self, pretrained_path):
        super(testModel_Net, self).__init__()
        
        self.f = Model().f
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        out = self.f(x)
        return out


def testAcc_func(net, data_loader, criterion, tolerance):
    net.eval()
    
    total_loss, total_num, total_correctNum, test_bar = 0.0, 0, 0, tqdm(data_loader, desc='testing on test set')
    
    with torch.no_grad():
        
        for connData, target, target_mat_name in test_bar:
            """
            # already moved those nanTarget mat files to another dir!!!
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
            
            # for debug:
            print('**************out:')
            print(out.size())
            print(out)
            
            test_bar.set_description('Test Acc: {:.4f}% Test Loss: {:.4f}'.format(total_correctNum*100 / total_num, total_loss / total_num))
            
    test_loss = total_loss / total_num
    test_acc = total_correctNum / total_num
    
    return (test_loss, test_acc)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Version 2 Test Accuracy with Trained Model')
    parser.add_argument('--model_path', type=str, default='results_myTest3_origTarget/model_resnet9_myTest3_origTarget/avgpool/connData_PCA_50/best_10_50_model.pth',help='The pretrained model path')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/', type=str, help='Dir of the mat files dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    parser.add_argument('--batch_size', default=10, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--result_dir', default='results_myTest3_origTarget/model_resnet9_myTest3_origTarget/avgpool/connData_PCA_50/acc_v2.txt', type=str, help='Dir to save the acc result')
    parser.add_argument('--tolerance', default=2, type=float, help='tolerance of difference between abs(gt-pred)')
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    
    # newly added for PCA versions: 50 for doing PCA; 379 for NOT doing PCA
    parser.add_argument('--new_featDim', default=50, type=int,  help='Number of feature dimensions after PCA reduction')
    
    # args parse:
    args = parser.parse_args()
    model_path = args.model_path
    data_dir = args.data_dir
    label_name = args.label_name
    batch_size = args.batch_size
    result_dir = args.result_dir
    tolerance = args.tolerance
    train_test_root = args.train_test_root
    new_featDim = args.new_featDim
    
    # NO transforms!
    
    # data prepare: newly modified:
    test_dataset = ModelNetDataLoader(root=data_dir, args=args, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    # load model:
    model = testModel_Net(pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
    
    # do our test: compute accuracy (and also loss) using the output from the model:
    criterion = torch.nn.L1Loss() # SAME as the one used in trainig!
    test_loss, test_acc = testAcc_func(model, test_loader, criterion, tolerance)
    
    # write result to file:
    file1 = open(result_dir,"w")
    text_str = 'test_loss = ' + str(test_loss) + '\ntest_acc = ' + str(test_acc*100) + '%'
    file1.write(text_str)
    file1.close()
    

