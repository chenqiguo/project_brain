#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:07:18 2021

@author: guo.1648
"""

# my 1st try: dimensionality reduction on connData: PCA:
# do PCA independently on each connData(i.e., connL vertically concatenate connR),
# to reduce feature dim from 379 -> 100? 50? 25?
# finally using the eigenVal or eigenVec as new features.

# Note: eigenVals are in descending orders!

# referenced from connData_dimReduc_PCA_v1.py


import os
import scipy.io as sio

import numpy as np
import math

"""
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
"""


def myPCA(dataMat, new_featDim):
    # connArray: (64984, 379) dim
    
    # calculate the mean of each column
    M = np.mean(dataMat.T,axis=1)
    # center columns by subtracting column means
    C = dataMat - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T) # (379, 379) dim
    # eigendecomposition of covariance matrix
    eigenValues, eigenVectors = np.linalg.eig(V) # each column of eigenVectors is an eigenvector
    
    # order the eigenVectors based on eigenValues
    sort_order = np.argsort(-eigenValues)
    listToAppend = []
    for i in range(379):
        veci = eigenVectors[:,sort_order[i]].reshape(-1,1)
        listToAppend.append(veci)
    eigenVectors = np.hstack(listToAppend)
    # order the eigenValues in descending orders
    eigenValues = -np.sort(-eigenValues)
    
    # only take the first new_featDim num of eigenVals and eigenVecs:
    eigenVectors = eigenVectors[:,:new_featDim]
    eigenValues = eigenValues[:new_featDim]
    
    # note: the normalized eigenVectors s.t. the column v[:,i] is the eigenvec
    # corresponding to the eigenval w[i]
    
    # newly added: project the original dataMat onto the eigenVectors:
    dataMat_feat = np.matmul(dataMat, eigenVectors) # dim (64984, new_featDim)
    
    return (dataMat_feat, eigenValues, eigenVectors)


# PCA version 1:
def loadTargetMatFile(target_behavior_name, target_mat_fullname, new_featDim):
    fullFileName = target_mat_fullname # full name of the mat file
    mat_contents = sio.loadmat(fullFileName)
    
    # get mat fields:
    connL = mat_contents['connL'] # (32492, 379)
    connR = mat_contents['connR'] # (32492, 379)
    
    connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
    #connArray = np.array(connL) # only for debug
    
    # do the PCA on connArray:
    connArray_feat, eigenValues, eigenVectors = myPCA(connArray, new_featDim)
    
    # version 1: only use the connArray_feat(i.e., projection of connArray onto eigenVecs) as new features:
    connData_feat = connArray_feat
    
    behavior_val = None
    behavior_names = mat_contents['behavior_names']
    for i,element in enumerate(behavior_names[0]):
        this_behavior_name = element[0]
        if this_behavior_name == target_behavior_name:
            behavior_val = mat_contents['behavior'][0][i]
            break
    assert(behavior_val is not None)
    
    """# NOT USE: here we use original target instead of log target !!!
    if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
        behavior_val = math.log(behavior_val-49) # min of target_list_origin is 50!
    """
    
    return (connData_feat, behavior_val)

