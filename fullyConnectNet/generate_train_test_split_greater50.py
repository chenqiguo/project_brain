#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:53:42 2021

@author: guo.1648
"""

# based on ResNet/train_test_split/train7test3/ ,
# generate new train_test_split s.t. only contains subjects whose target values > 50

import os
#import scipy.io as sio
import pickle

target_behavior_name = 'DSM_Anxi_T'
train_test_root = '/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/'
matData_dir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl'

dstDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/fullyConnectNet/train_test_split/only_targetGreater50/'


if __name__ == '__main__':
    
    mat_ids = {}
    mat_ids['train'] = [line.rstrip() for line in open(os.path.join(train_test_root, 'train.txt'))]
    mat_ids['test'] = [line.rstrip() for line in open(os.path.join(train_test_root, 'test.txt'))]
    
    f_pkl = open(matData_dir,'rb')
    dataMat_subject_all_dict = pickle.load(f_pkl)
    f_pkl.close()
    
    # (1) for training:
    matF_train = []
    for matFileName in mat_ids['train']:
        dict_ = dataMat_subject_all_dict[matFileName]
        target_val = dict_[target_behavior_name]
        if target_val > 50:
            matF_train.append(matFileName)
        
    # (2) for test:
    matF_test = []
    for matFileName in mat_ids['test']:
        dict_ = dataMat_subject_all_dict[matFileName]
        target_val = dict_[target_behavior_name]
        if target_val > 50:
            matF_test.append(matFileName)
    
    # save file names to txt files:
    f_train = open(dstDir + 'train.txt', 'w')
    for line in matF_train:
        f_train.write(line+'\n')
    f_train.close()
    
    f_test = open(dstDir + 'test.txt', 'w')
    for line in matF_test:
        f_test.write(line+'\n')
    f_test.close()
        
        
    

