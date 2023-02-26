#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:05:00 2021

@author: guo.1648
"""

# version 1:
# NOT use this! --> cannot fit data into CPU memory!

# binary classification (regression) version:
# thresh50: 0 (target==50) v.s. 1 (target>50)
# thresh51: 0 (target==51) v.s. 1 (target>51) <-- NOT did! Target is biased: lots of 1

# This is for sparse linear regression (as baseline);
# threshold the connData to produce a binary adjacency matrix as input.


import os
import scipy.io as sio
import numpy as np
import math

from sklearn.linear_model import Lasso
from scipy import sparse


srcDir = '/scratch/Chenqi/project_brain/MMPconnMesh/'
train_test_root = '/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/'
target_behavior_name = 'DSM_Anxi_T'

threshold = 0.05 # threshold for connData


def loadDataTarget(flag):
    # flag = 'train' or 'test'
    
    conn_list = []
    behavior_val_list = []
    
    mat_ids = [line.rstrip() for line in open(os.path.join(train_test_root, flag+'.txt'))]
    for filename in mat_ids:
        print("------------------deal with---------------------")
        print(filename)
        fullFileName = srcDir + filename # full name of the mat file
        mat_contents = sio.loadmat(fullFileName)
        
        #print()
        # get mat fields:
        connL = mat_contents['connL']
        connR = mat_contents['connR']
        connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
        # threshold the connData to produce a binary adjacency matrix as input:
        dataMat_feat = connArray >= threshold # bool array
        
        behavior_val = None
        behavior_names = mat_contents['behavior_names']
        for i,element in enumerate(behavior_names[0]):
            this_behavior_name = element[0]
            if this_behavior_name == target_behavior_name:
                behavior_val = mat_contents['behavior'][0][i]
                break
        assert(behavior_val is not None)
        
        # binary classification target:
        if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
            if behavior_val == 50: # 51
                behavior_val = 0
            else:
                behavior_val = 1
        
        conn_list.append(dataMat_feat.flatten())
        behavior_val_list.append(behavior_val)
        
        #print()
    
    return (conn_list, behavior_val_list)



if __name__ == '__main__':
    
    conn_list_train, behavior_val_list_train = loadDataTarget(flag = 'train')
    conn_list_test, behavior_val_list_test = loadDataTarget(flag = 'test')
    
    #"""
    # we may need to search for the penalty!!!
    penalty = 1 # 0.1
    sparse_lasso = Lasso(alpha=penalty,max_iter=10e5)
    
    X_train = np.array(conn_list_train[:100]).astype(int) # for debug: only 5 samp
    X_train_sp = sparse.coo_matrix(X_train)
    y_train = behavior_val_list_train[:100]
    
    sparse_lasso.fit(X_train_sp,y_train)
    #"""

