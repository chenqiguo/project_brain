#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:07:18 2021

@author: guo.1648
"""

import os
import scipy.io as sio

import numpy as np
import math


def loadTargetMatFile_xyz(target_behavior_name, target_mat_fullname):
    fullFileName = target_mat_fullname # full name of the mat file
    mat_contents = sio.loadmat(fullFileName)
    
    # get mat fields:
    connL = mat_contents['connL'] # (32492, 379)
    connR = mat_contents['connR'] # (32492, 379)
    
    verticesL = mat_contents['verticesL'] # (32492, 3)
    verticesR = mat_contents['verticesR'] # (32492, 3)
    
    connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
    verticesArray = np.concatenate((verticesL, verticesR)) # (64984, 3)
    
    connData_xyz = np.hstack((verticesArray, connArray)) # (64984, 382)
        
    behavior_val = None
    behavior_names = mat_contents['behavior_names']
    for i,element in enumerate(behavior_names[0]):
        this_behavior_name = element[0]
        if this_behavior_name == target_behavior_name:
            behavior_val = mat_contents['behavior'][0][i]
            break
    assert(behavior_val is not None)
    
    # newly added:
    if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
        behavior_val = math.log(behavior_val-49) # min of target_list_origin is 50!
    
    return (connData_xyz, behavior_val)

#loadTargetMatFile_xyz('DSM_Anxi_T', '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/100206.mat') # for debug



def loadTargetMatFile(target_behavior_name, target_mat_fullname):
    fullFileName = target_mat_fullname # full name of the mat file
    mat_contents = sio.loadmat(fullFileName)
    
    # get mat fields:
    connL = mat_contents['connL'] # (32492, 379)
    connR = mat_contents['connR'] # (32492, 379)
    
    connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
    #connArray = np.array(connL) # only for debug
    
    behavior_val = None
    behavior_names = mat_contents['behavior_names']
    for i,element in enumerate(behavior_names[0]):
        this_behavior_name = element[0]
        if this_behavior_name == target_behavior_name:
            behavior_val = mat_contents['behavior'][0][i]
            break
    assert(behavior_val is not None)
    
    # newly added:
    if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
        behavior_val = math.log(behavior_val-49) # min of target_list_origin is 50!
    
    return (connArray, behavior_val)

