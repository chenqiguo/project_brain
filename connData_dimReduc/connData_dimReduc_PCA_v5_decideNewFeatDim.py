#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:50:02 2021

@author: guo.1648
"""

# for PCA v5:
# code to decide new_featDim (i.e., the num of new_feat) based on eigenvalues:
# select the new_featDim that captures the large varicance (i.e., eigenvalue) of fc data.

# referenced from connData_dimReduc_PCA_v3_decideNewFeatDim.py

import numpy as np
import os
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnSparse/'


def myPCA(dataMat, new_featDim):
    # fc: (379, 379) dim
    
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


def compute_eigValPercent_arr():
    global eigValPercent_arr
    
    eigValPercent_allList = []
    
    for (dirpath, dirnames, filenames) in os.walk(srcRootDir):
        #print(filenames)
        for filename in filenames:
            #print(filename)
            if ".mat" in filename:
                print("------------------deal with---------------------")
                print(filename)
                fullFileName = srcRootDir + filename
                mat_contents = sio.loadmat(fullFileName)
                
                fc = mat_contents['fc'] # (379, 379)
                
                # do the PCA on fc: Note: here we only use eigenValues (i.e., variances) !!!
                dataMat_feat, eigenValues, eigenVectors = myPCA(fc, 379)
                
                eigValPercent = eigenValues / sum(eigenValues)
                eigValPercent_allList.append(eigValPercent)
    
    eigValPercent_arr = np.array(eigValPercent_allList)
    
    return eigValPercent_arr




if __name__ == '__main__':
    global eigValPercent_arr
    
    eigValPercent_arr = compute_eigValPercent_arr()
    
    eigValPercent_avg = np.mean(eigValPercent_arr,axis=0)
    
    # plot the eigValPercent_avg:
    fig = plt.figure()
    x = np.arange(1,380)
    plt.plot(x, eigValPercent_avg)
    plt.scatter(x, eigValPercent_avg)
    plt.title('eigValPercent_avg')
    fig.savefig('results/PCA_v5_CV/PCA_v5_eigValPercent_avg.png')
    
    # combined this result with Fabian's idea,
    # we decide: new_featDim = 32
    

