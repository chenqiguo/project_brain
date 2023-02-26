#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:07:18 2021

@author: guo.1648
"""

# binary classification version:
# thresh50: 0 (target==50) v.s. 1 (target>50)
# thresh51: 0 (target==51) v.s. 1 (target>51) <-- NOT did! Target is biased: lots of 1

# This is for sparse linear regression (as baseline);
# threshold the connData to produce a binary adjacency matrix as input.


import os
import scipy.io as sio
import pickle

import numpy as np
import math



def loadTargetMatFile_sparse(data_dir, target_behavior_name, target_mat_name, threshold):
    # e.g., data_dir = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/'
    
    fullFileName = data_dir + target_mat_name # full name of the mat file
    mat_contents = sio.loadmat(fullFileName)
    
    # get mat fields:
    connL = mat_contents['connL'] # (32492, 379)
    connR = mat_contents['connR'] # (32492, 379)
    
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
    
    return (dataMat_feat, behavior_val)

