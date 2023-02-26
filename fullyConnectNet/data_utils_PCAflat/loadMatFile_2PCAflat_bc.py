#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:07:18 2021

@author: guo.1648
"""

# binary classification version:
# thresh50: 0 (target==50) v.s. 1 (target>50)
# thresh51: 0 (target==51) v.s. 1 (target>51) <-- NOT did! Target is biased: lots of 1


import os
#import scipy.io as sio
import pickle

import numpy as np
import math



def loadTargetMatFile_PCAflat(data_dir, target_behavior_name, target_mat_name):
    # e.g., data_dir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v3/connData_dimReduc_PCA_dict_v3.pkl'
    
    f_pkl = open(data_dir,'rb')
    dataMat_subject_all_dict = pickle.load(f_pkl)
    f_pkl.close()
    
    target_dict = dataMat_subject_all_dict[target_mat_name]
    dataMat_feat = target_dict['dataMat_feat']
    behavior_val = target_dict[target_behavior_name]
    
    assert(behavior_val is not None)
    
    # binary classification target:
    if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
        if behavior_val == 50: # 51
            behavior_val = 0
        else:
            behavior_val = 1
    
    # newly added: flatten dataMat_feat:
    dataMat_feat_flat = dataMat_feat.flatten() # dim: (1024,)
    
    return (dataMat_feat_flat, behavior_val)

