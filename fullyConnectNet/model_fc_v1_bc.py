#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:42:19 2021

@author: guo.1648
"""

# use deep fully connected network with flatten input.

# binary classification version.

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_cls, input_dim):
        super(Model, self).__init__()
        
        # num_cls: ==1 for regression; ==2 for binary classification.
        # input_dim == 32*32 = 1024
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 8)
        self.fc6 = nn.Linear(8, num_cls)
        self.dropout = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(8)

    def forward(self, x):
                
        out = F.relu(self.bn1(self.fc1(x).squeeze(1)))      # torch.Size([B, 512])
        out = F.relu(self.bn2(self.dropout(self.fc2(out)))) # torch.Size([B, 256])
        out = F.relu(self.bn3(self.dropout(self.fc3(out)))) # torch.Size([B, 128])
        out = F.relu(self.bn4(self.dropout(self.fc4(out)))) # torch.Size([B, 64])
        out = F.relu(self.bn5(self.dropout(self.fc5(out)))) # torch.Size([B, 8])
        
        out = F.relu(self.fc6(out)) # torch.Size([B, 2]) ??? How about F.log_softmax(out, dim=1) ???
        
        return out



# for debug:
#model = Model()


# just for debug:
if __name__ == '__main__':
    #input_width = 379
    #input_height = 64984
    model = Model(num_cls=2, input_dim=1024).cuda()
    
    print(model)
    
    import argparse
    from data_utils_PCAflat.ModelNetDataLoader_2PCAflat_origTarget import ModelNetDataLoader_PCAflat
    
    parser = argparse.ArgumentParser(description='Train Deep FC Regress w. PCA Flatten Data, v1')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=20, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl', type=str, help='Dir of the PCA dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    #parser.add_argument('--train_split', default=0.7, type=float, help='Percent of training set')
    #parser.add_argument('--tolerance', default=2, type=float, help='tolerance of difference between abs(gt-pred)')
    parser.add_argument('--result_dir', default='results/fc_v1_origTarget/', type=str, help='Dir of the results')
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    
    # args parse
    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epochs
    data_dir = args.data_dir
    label_name = args.label_name
    #train_split = args.train_split
    #tolerance = args.tolerance
    result_dir = args.result_dir
    train_test_root = args.train_test_root
    
    train_dataset = ModelNetDataLoader_PCAflat(args=args, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    
    for batch in train_loader:
        
        images, labels, _ = batch 
        images = images.float().unsqueeze(1).cuda() # torch.Size([batch_size, 1024])
        
        out = model(images) #torch.Size([batch_size, 2])
        
        print(out.shape)
        print(out)
                    

