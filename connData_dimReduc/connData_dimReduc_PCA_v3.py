#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 15:14:38 2021

@author: guo.1648
"""

# My 3rd try: Fabian's version: PCA:
# do PCA independently on each connData(i.e., connL vertically concatenate connR),
# to reduce feature dim from 379 -> 32.
# Then the resulting 32x32 matrix will be sent to resnet9 (Fabian's ver).

# Note: here we do PCA twice:
# 1) do PCA in connData's rows <==> do PCA in connData.T
# 2) do PCA in connData's cols <==> do PCA in 1)'s result
# So that we get a final data matrix with dim (32,32) for each subject's connData.

# For more details: see my notes.

# referenced from connData_dimReduc_PCA_v1.py


import numpy as np
import os
import pickle
import scipy.io as sio
from scipy.sparse.linalg import eigsh


new_featDim = 32 # decided from code connData_dimReduc_PCA_v3_decideNewFeatDim.py
target_behavior_name = 'DSM_Anxi_T'

srcRootDir = '/scratch/Chenqi/project_brain/MMPconnMesh_targetNan/' #'/scratch/Chenqi/project_brain/MMPconnMesh/'

dstRootDir = 'results/PCA_v3/'
dstPklName = 'connData_dimReduc_PCA_v3_remaining.pkl' #'connData_dimReduc_PCA_v3.pkl'

connData_dimReduc_PCA = []

# featureDict = {
#        'filename': '208125.mat',
#        'dataMat_feat': dataMat_feat, # dim (32, 32) --> input
#        'DSM_Anxi_T': 50 # the original target value --> output
#        }


def myPCA_newVer(dataMat, new_featDim):
    # for example: dataMat = connArray: (64984, 379) dim
    
    # calculate the mean of each column:
    M = np.mean(dataMat.T,axis=1)
    # center columns by subtracting column means:
    C = dataMat - M
    # calculate covariance matrix of centered matrix:
    V = np.cov(C.T) # (379, 379) dim
    # eigendecomposition of covariance matrix:
    #eigenValues, eigenVectors = np.linalg.eig(V) # NOT use this!!! <-- gives zero eigVecs!!!
    eigenValues, eigenVectors = eigsh(V, k=new_featDim) # each column of eigenVectors is an eigenvector
    
    # order the eigenVectors based on eigenValues:
    sort_order = np.argsort(-eigenValues)
    listToAppend = []
    for i in range(new_featDim):
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


if __name__ == '__main__':
    for (dirpath, dirnames, filenames) in os.walk(srcRootDir):
        #print(filenames)
        for filename in filenames:
            #print(filename)
            if ".mat" in filename:
                print("------------------deal with---------------------")
                print(filename)
                fullFileName = srcRootDir + filename
                mat_contents = sio.loadmat(fullFileName)
                
                # get mat fields:
                connL = mat_contents['connL'] # (32492, 379)
                connR = mat_contents['connR'] # (32492, 379)
                
                connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
                
                behavior_val = None
                behavior_names = mat_contents['behavior_names']
                for i,element in enumerate(behavior_names[0]):
                    this_behavior_name = element[0]
                    if this_behavior_name == target_behavior_name:
                        behavior_val = mat_contents['behavior'][0][i]
                        break
                assert(behavior_val is not None)
                
                #"""
                # (1) do the PCA on connArray.T:
                dataMat_feat_B2, eigenValues, eigenVectors = myPCA_newVer(np.transpose(connArray), new_featDim)
                dataMat_feat_B = dataMat_feat_B2.T # dim (32,379)
                
                # (2) do the PCA on dataMat_feat_B:
                dataMat_feat_C, eigenValues, eigenVectors = myPCA_newVer(dataMat_feat_B, new_featDim)
                #"""
                """
                ### NOT USE ###
                # (1) do the PCA on connArray:
                dataMat_feat, eigenValues, eigenVectors = myPCA(connArray, new_featDim)
                # dataMat_feat with dim (64984, 32)
                
                # (2) do the PCA on dataMat_feat.T:
                dataMat_feat_, eigenValues_, eigenVectors_ = myPCA(dataMat_feat.T, new_featDim)
                """
                
                featureDict = {
                        'filename': filename,
                        'dataMat_feat': dataMat_feat_C, # dim (32,32)
                        'DSM_Anxi_T': behavior_val
                        }
                
                connData_dimReduc_PCA.append(featureDict)
                
                #print()
                
    # save the dicts into pickle files
    f_pkl = open(dstRootDir+dstPklName, 'wb')
    pickle.dump(connData_dimReduc_PCA,f_pkl)
    f_pkl.close()

