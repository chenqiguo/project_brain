#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:58:05 2021

@author: guo.1648
"""


# My 4th try: Jeff's version: PCA:
# do PCA each time on a stack of connData cols or rows across all (training) subjects,
# to reduce feature dim from 379 -> 32.
# Then the resulting 32x32 matrix will be sent to resnet9 (Fabian's ver).

# Note1: here we do PCA twice:
# 1) do PCA for each region (total num 379) acrross all subjects (709 for training)
# 2) do PCA for each new "vertex" feat (total num 32) across all subjects (709 for training)
# So that we get a final data matrix with dim (32,32) for each subject's connData.

# Note2: in this pipeline, we compute each eigenVecs(Vals) only using the training dataset,
# and then compute dataMat_feat for the testing dataset using the eigenVecs computed from training dataset!

# Note3: to save memory, we saved some tmp files at dir results/PCA_v4/tmp/ .
# To save space, delete these tmp files later!!!

# For more details: see my notes.

# referenced from connData_dimReduc_PCA_v3.py.


import numpy as np
import os
import pickle
import scipy.io as sio
from scipy.sparse.linalg import eigsh


new_featDim = 32 # decided from code connData_dimReduc_PCA_v3_decideNewFeatDim.py
target_behavior_name = 'DSM_Anxi_T'

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/' #'/scratch/Chenqi/project_brain/MMPconnMesh/'
train_test_root = '/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/'

dstRootDir = 'results/PCA_v4/'
dstPklName_trainData = 'connData_train_dimReduc_PCA_v4.pkl' # to store connData_train_dimReduc_PCA
dstPklName_testData = 'connData_test_dimReduc_PCA_v4.pkl' # to store connData_test_dimReduc_PCA

connData_train_dimReduc_PCA = []
connData_test_dimReduc_PCA = []

# featureDict = {
#        'filename': '208125.mat',
#        'train_or_test': 'train',
#        'dataMat_feat': dataMat_feat, # dim (32, 32) --> input
#        'DSM_Anxi_T': 50 # the original target value --> output
#        }

tmpFilePath = 'results/PCA_v4/tmp/'


def get_dataMat_D(mat_list, i):
    dataMat_D_list = []
    
    for filename in mat_list:
        
        print("------------------deal with---------------------")
        print(filename)
        
        fullFileName = srcRootDir + filename
        mat_contents = sio.loadmat(fullFileName)
        
        # (1) for connData:
        connL = mat_contents['connL'] # (32492, 379)
        connR = mat_contents['connR'] # (32492, 379)
        connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
        
        # only get the i-th column:
        connArray_coli = connArray[:,i]
        dataMat_D_list.append(connArray_coli)
            
    dataMat_D = np.array(dataMat_D_list) # shape (709, 64984) or (306, 64984)
    
    return dataMat_D


def get_dataMat_F(tflag, mat_list, k):
    # tflag = 'train' or 'test'
    
    dataMat_F_list = []
    
    for j, filename in enumerate(mat_list): # for each subject
        print("&&&&&&&&&&&&&&&&&&deal with&&&&&&&&&&&&&&&&&&&&&")
        print(filename)
        
        fullFileName = tmpFilePath+'setp15/dataMat_'+tflag+'_subject'+str(j)+'.pkl'
        f_pkl = open(fullFileName,'rb')
        dataMat_subjectj = pickle.load(f_pkl)
        f_pkl.close()
        
        # only get the k-th row:
        connArray_rowk = dataMat_subjectj[k,:]
        dataMat_F_list.append(connArray_rowk)
        
    dataMat_F = np.array(dataMat_F_list) # dim (709, 379) or (306, 379) <-- check shape!!!
    
    return dataMat_F



def myPCA_trainVer(dataMat, new_featDim):
    # for example: dataMat = dataMat_D for train: (709, 64984) dim
    
    # calculate the mean of each column:
    M = np.mean(dataMat.T,axis=1)
    # center columns by subtracting column means:
    C = dataMat - M
    # calculate covariance matrix of centered matrix:
    V = np.cov(C.T) # (64984, 64984) dim
    # eigendecomposition of covariance matrix:
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
    dataMat_E = np.matmul(dataMat, eigenVectors) # dim (709, new_featDim)
    
    return (eigenVectors, dataMat_E)


def step15_func(tflag, mat_list):
    # tflag = 'train' or 'test'
    
    for j, filename in enumerate(mat_list): # for each subject
        print("******************deal with*********************")
        print(filename)
        
        all_regions_list = []
        for i in range(379): # for each region
            f_pkl = open(tmpFilePath+'setp1/dataMat_E_'+tflag+'_region'+str(i)+'.pkl','rb')
            dataMat_Ei = pickle.load(f_pkl)
            f_pkl.close()
            
            all_regions_list.append(dataMat_Ei[j,:])
            
        dataMat_subject = np.array(all_regions_list).T # dim (32,379) <-- check shape!!!
        # save dataMat_subject to tmp file:
        f_pkl = open(tmpFilePath+'setp15/dataMat_'+tflag+'_subject'+str(j)+'.pkl', 'wb')
        pickle.dump(dataMat_subject,f_pkl)
        f_pkl.close()
    
    return


def step25_func(tflag, mat_list):
    # tflag = 'train' or 'test'
    
    for j, filename in enumerate(mat_list): # for each subject
        print("^^^^^^^^^^^^^^^^^^deal with^^^^^^^^^^^^^^^^^^^^^")
        print(filename)
        
        all_vertices_list = []
        for k in range(new_featDim): # for each new "vertex" feat
            f_pkl = open(tmpFilePath+'setp2/dataMat_G_'+tflag+'_vertex'+str(k)+'.pkl','rb')
            dataMat_Gk = pickle.load(f_pkl)
            f_pkl.close()
            
            all_vertices_list.append(dataMat_Gk[j,:])
            
        dataMat_subject = np.array(all_vertices_list) # dim (32,32) <-- check shape!!!
        
        featureDict = {
                      'filename': filename,
                      'train_or_test': tflag,
                      'dataMat_feat': dataMat_subject, # dim (32, 32) --> input
                      #'DSM_Anxi_T': ??? # the original target value --> output
                      }

        if tflag == 'train':
            connData_train_dimReduc_PCA.append(featureDict)
        else:
            connData_test_dimReduc_PCA.append(featureDict)
    
    return




if __name__ == '__main__':
    
    # get mat file names from train_test split root:
    mat_ids = {}
    mat_ids['train'] = [line.rstrip() for line in open(os.path.join(train_test_root, 'train.txt'))]
    mat_ids['test'] = [line.rstrip() for line in open(os.path.join(train_test_root, 'test.txt'))]
    """
    # step 1) to save memory (but needs lots of time!) , here we deal with each region one by one and saved to tmp files:
    print('@@@@@@@@@@@@@@@@@@@@@@@@ step 1 @@@@@@@@@@@@@@@@@@@@@@@@')
    for i in range(379): # 379; for each region
        
        # (a) for train:
        dataMat_D = get_dataMat_D(mat_ids['train'], i) # for train
        assert(dataMat_D.shape == (len(mat_ids['train']), 64984))
        # (a1) do PCA on dataMat_D:
        eigenVecs_V3, dataMat_E = myPCA_trainVer(dataMat_D, new_featDim)
        # eigenVecs_V3.shape: (64984,32); dataMat_E.shape: (709, 32)
        # save dataMat_E to tmp file:
        f_pkl = open(tmpFilePath+'setp1/dataMat_E_train_region'+str(i)+'.pkl', 'wb')
        pickle.dump(dataMat_E,f_pkl)
        f_pkl.close()
        
        # (b) for test:
        dataMat_D = get_dataMat_D(mat_ids['test'], i) # for test
        assert(dataMat_D.shape == (len(mat_ids['test']), 64984))
        # (b1) do PCA on dataMat_D using eigenVecs_V3 computed from training:
        dataMat_E = np.matmul(dataMat_D, eigenVecs_V3) # dim (306, new_featDim)
        # save dataMat_E to tmp file:
        f_pkl = open(tmpFilePath+'setp1/dataMat_E_test_region'+str(i)+'.pkl', 'wb')
        pickle.dump(dataMat_E,f_pkl)
        f_pkl.close()
    """
    """
    # step 1.5) concate each region column of above, to get a (32,379) data mat for each subject:
    print('@@@@@@@@@@@@@@@@@@@@@@@@ step 1.5 @@@@@@@@@@@@@@@@@@@@@@@@')
    step15_func('train', mat_ids['train'])
    step15_func('test', mat_ids['test'])
    """
    """
    # step 2) need to load&concate the matrices above:
    print('@@@@@@@@@@@@@@@@@@@@@@@@ step 2 @@@@@@@@@@@@@@@@@@@@@@@@')
    for k in range(new_featDim): # ie 32: for each new "vertex" feat
        # (a) for train:
        dataMat_F = get_dataMat_F('train',mat_ids['train'], k) # for train
        assert(dataMat_F.shape == (len(mat_ids['train']), 379))
        # (a2) do PCA on dataMat_F:
        eigenVecs_V4, dataMat_G = myPCA_trainVer(dataMat_F, new_featDim) # <-- check shape!!!
        # save dataMat_G to tmp file:
        f_pkl = open(tmpFilePath+'setp2/dataMat_G_train_vertex'+str(k)+'.pkl', 'wb')
        pickle.dump(dataMat_G,f_pkl)
        f_pkl.close()
        
        # (b) for test:
        dataMat_F = get_dataMat_F('test',mat_ids['test'], k) # for test
        assert(dataMat_F.shape == (len(mat_ids['test']), 379))
        # (b2) do PCA on dataMat_F using eigenVecs_V4 computed from training:
        dataMat_G = np.matmul(dataMat_F, eigenVecs_V4) # dim (306, new_featDim) <-- check shape!!!
        # save dataMat_G to tmp file:
        f_pkl = open(tmpFilePath+'setp2/dataMat_G_test_vertex'+str(k)+'.pkl', 'wb')
        pickle.dump(dataMat_G,f_pkl)
        f_pkl.close()
    """
    #"""
    # step 2.5) get final result:
    # concate each new "region" row of above, to get a (32,32) data mat for each subject and save:
    print('@@@@@@@@@@@@@@@@@@@@@@@@ step 2.5 @@@@@@@@@@@@@@@@@@@@@@@@')
    step25_func('train', mat_ids['train'])
    step25_func('test', mat_ids['test'])
    
    # save final results (BUT WITHOUT target values!!! will add them later) to dst dir pkl:
    f_pkl = open(dstRootDir+dstPklName_trainData, 'wb')
    pickle.dump(connData_train_dimReduc_PCA,f_pkl)
    f_pkl.close()
    
    f_pkl = open(dstRootDir+dstPklName_testData, 'wb')
    pickle.dump(connData_test_dimReduc_PCA,f_pkl)
    f_pkl.close()
    #"""
