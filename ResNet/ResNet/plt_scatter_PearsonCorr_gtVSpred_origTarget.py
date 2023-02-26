#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 19:40:06 2021

@author: guo.1648
"""

# A new metric: try correlations:
# my 1st try, Pearson correlation between ground truth and model prediction.

# referenced from testModel_acc_v2_origTarget.py and scoresCorr_pearson_v1.py.


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
from data_utils.ModelNetDataLoader_v2_origTarget import ModelNetDataLoader
#from model_resnet34_myTest3 import Model
from model_resnet9_myTest3 import Model

#from PIL import Image
from torchvision import transforms, datasets
import torch.nn as nn

import pickle
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm



class testModel_Net(nn.Module):
    def __init__(self, pretrained_path):
        super(testModel_Net, self).__init__()
        
        self.f = Model().f
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        out = self.f(x)
        return out


def get_gt_pred_target(net, data_loader):
    # store the gt & pred label into lists:
    target_gt_list = []
    target_pred_list = []
    
    net.eval()
    test_bar = tqdm(data_loader, desc='testing on test set')
    
    with torch.no_grad():
        for connData, target, target_mat_name in test_bar:
            # already moved those nanTarget mat files to another dir!!!
            
            connData_ = connData.float().unsqueeze(1).cuda(non_blocking=True) # connData_.shape: torch.Size([batch_size, 1, 64984, 379])
            out = net(connData_) #torch.Size([batch_size, 1])
            
            target_pred = out.detach().cpu().squeeze(1).tolist()
            target_gt = target.float().cpu().tolist()
            
            target_gt_list += target_gt
            target_pred_list += target_pred
            
            
    return (target_gt_list, target_pred_list)


def plotScatter(X_arr, Y_arr, Pearson_r, Pearson_pval, X_str, Y_str, graph_filepath):
    # plot the 2D scatter plot, also label its corresponding Pearson r and pval:
    
    XY_arr = np.hstack((X_arr.reshape(-1,1), Y_arr.reshape(-1,1)))
    colors = np.zeros(len(Y_arr))
    for i in range(len(Y_arr)):
        colors[i] = np.count_nonzero(np.logical_and(XY_arr[:,0]==XY_arr[i,0],XY_arr[:,1]==XY_arr[i,1]))
    plt.scatter(X_arr, Y_arr, c=colors, alpha=0.3, cmap='viridis') # s=sizes,
    plt.colorbar()
    # also fit a line to X and Y:
    line_fit_1 = sm.OLS(Y_arr, sm.add_constant(X_arr)).fit()
    X_plot = np.linspace(45,85,10000)
    plt.plot(X_plot, X_plot*line_fit_1.params[0] + line_fit_1.params[1])
    plt.xticks(range(50,81,5), range(50,81,5))
    plt.yticks(range(50,56,5), range(50,56,5))
    plt.ylim(50,55)
    plt.title('Scatter plot of ' + X_str + ' v.s. ' + Y_str)
    plt.xlabel(X_str)
    plt.ylabel(Y_str)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    textstr = 'Pearson:\nr = %s' % float('%.5g' % Pearson_r) + '\np-value = %s' % float('%.5g' % Pearson_pval)
    plt.text(65, 54, textstr, verticalalignment='top', bbox=props)
    plt.savefig(graph_filepath, dpi=300, format='png', bbox_inches='tight')
    
    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ResNet Regress myTest3 plt scatter and compute Pearson')
    parser.add_argument('--model_path', type=str, default='results_myTest3_origTarget/model_resnet9_myTest3_origTarget/best_2_50_model.pth',help='The pretrained model path')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/', type=str, help='Dir of the mat files dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    parser.add_argument('--batch_size', default=2, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--result_dir', default='results_myTest3_origTarget/model_resnet9_myTest3_origTarget/', type=str, help='Dir to save the acc result')
    parser.add_argument('--tolerance', default=2, type=float, help='tolerance of difference between abs(gt-pred)')
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    
    # args parse:
    args = parser.parse_args()
    model_path = args.model_path
    data_dir = args.data_dir
    label_name = args.label_name
    batch_size = args.batch_size
    result_dir = args.result_dir
    tolerance = args.tolerance
    train_test_root = args.train_test_root
    
    # data prepare:
    test_dataset = ModelNetDataLoader(root=data_dir, args=args, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    
    # load model:
    model = testModel_Net(pretrained_path=model_path).cuda()
    for param in model.f.parameters():
        param.requires_grad = False
    
    target_gt_list, target_pred_list = get_gt_pred_target(model, test_loader)
    
    # save target_gt_list and target_pred_list to pkl file:
    target_gt_pred_list = {'target_gt_list': target_gt_list,
                           'target_pred_list': target_pred_list}
    f_pkl = open(result_dir+'target_gt_pred_list.pkl', 'wb')
    pickle.dump(target_gt_pred_list,f_pkl)
    
    # compute Pearson correlation between target_gt_list and target_pred_list:
    Pearson_r, Pearson_pval = stats.pearsonr(target_gt_list, target_pred_list)
    # write result to file:
    file1 = open(result_dir+'Pearson.txt',"w")
    text_str = 'Pearson_r = ' + str(Pearson_r) + '\nPearson_pval = ' + str(Pearson_pval)
    file1.write(text_str)
    file1.close()
    
    # also plot Scatter plot between target_gt_list and target_pred_list:
    X_str = label_name + ' gt'
    Y_str = label_name + ' pred'
    graph_filepath = result_dir + 'Pearson_scatter_gtVSpred.png'
    plotScatter(np.array(target_gt_list), np.array(target_pred_list), Pearson_r, Pearson_pval, X_str, Y_str, graph_filepath)

