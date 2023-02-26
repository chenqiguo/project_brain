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


def loadTargetMatFile(srcDir, target_behavior_name, target_mat_name):
    fullFileName = srcDir + target_mat_name # full name of the mat file
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


#connData, target = loadTargetMatFile(srcDir, target_behavior_name, target_mat_name)



def loadMatFile(srcDir, target_behavior_name):
    #connL_list = []
    #connR_list = []
    conn_list = []
    behavior_val_list = []
    
    for (dirpath, dirnames, filenames) in os.walk(srcDir):
        for filename in filenames:
            if ".mat" in filename:
                #print("------------------deal with---------------------")
                #print(filename)
                fullFileName = srcDir + filename # full name of the mat file
                mat_contents = sio.loadmat(fullFileName)
                
                # get mat fields:
                connL = mat_contents['connL']
                connR = mat_contents['connR']
                
                behavior_val = None
                behavior_names = mat_contents['behavior_names']
                for i,element in enumerate(behavior_names[0]):
                    this_behavior_name = element[0]
                    if this_behavior_name == target_behavior_name:
                        behavior_val = mat_contents['behavior'][0][i]
                        break
                assert(behavior_val is not None)
                
                #connL_list.append(np.array(connL))
                #connR_list.append(np.array(connR))
                conn_list.append(np.concatenate((np.array(connL), np.array(connR))))
                behavior_val_list.append(behavior_val)
                
                #print()
    
    return (conn_list, behavior_val_list)



#conn_list, behavior_val_list = loadMatFile(srcDir, target_behavior_name)
