#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:37:50 2021

@author: guo.1648
"""

# my 1st try: dimensionality reduction on connData: PCA:
# do PCA independently on each connData(i.e., connL vertically concatenate connR),
# to reduce feature dim from 379 -> 100? 50? 25?
# finally using the eigenVal or eigenVec or projections as new features (into CNN).

# Note: eigenVals are in descending orders!
# To save space, NOT saved the pkl file! Instead, I modified the load py file
# to compute the PCA while running --> this py file is just for debug & reference.

# referenced from feature4_PCA_lbRb_ab2d.py and loadMatFile_origTarget.py.


import numpy as np
import os
import pickle
import scipy.io as sio


new_featDim = 50 #100? 50? 25?

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/'
target_behavior_name = 'DSM_Anxi_T'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/PCA_v1/'
dstPklName = 'connData_dimReduc_PCA_v1.pkl'

connData_dimReduc_PCA = []

# featureDict = {
#        'filename': '208125.mat',
#        'eigenValues': eigenValues,
#        'eigenVectors': eigenVectors,
#        'dataMat_feat': dataMat_feat,
#        'DSM_Anxi_T': 50 # the original target value
#        }


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
                                
                # do the PCA on connArray:
                dataMat_feat, eigenValues, eigenVectors = myPCA(connArray, new_featDim)
                
                featureDict = {
                        'filename': filename,
                        'eigenValues': eigenValues,
                        'eigenVectors': eigenVectors,
                        'dataMat_feat': dataMat_feat,
                        'DSM_Anxi_T': behavior_val
                        }
                
                connData_dimReduc_PCA.append(featureDict)
                
                #print()
    
    # save the dicts into pickle files
    f_pkl = open(dstRootDir+dstPklName, 'wb')
    pickle.dump(connData_dimReduc_PCA,f_pkl)
    f_pkl.close()

    
    



