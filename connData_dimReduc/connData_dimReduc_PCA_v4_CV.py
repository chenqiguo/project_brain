#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:39:31 2021

@author: guo.1648
"""

# do 10-fold Cross Validation on connData when generating 2PCA_v4.
# (i.e., only change it to be CV; others remain the same as in connData_dimReduc_PCA_v4)

# referenced from connData_dimReduc_PCA_v4.py


import numpy as np
import os
import pickle
import scipy.io as sio
from scipy.sparse.linalg import eigsh


new_featDim = 32 # decided from code connData_dimReduc_PCA_v3_decideNewFeatDim.py
target_behavior_name = 'DSM_Anxi_T'

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/'
train_valid_root = srcRootDir + 'mat_names_CV/'
dataMat_D_rootDir = srcRootDir + 'dataMat_D/'
file_allMatNames = srcRootDir + 'dataMat_D_matNames.txt'

dstRootDir = 'results/PCA_v4_CV/'
dstPklName_trainData = 'connData_train_dimReduc_PCA_v4_CV.pkl' # to store connData_train_dimReduc_PCA
dstPklName_validData = 'connData_valid_dimReduc_PCA_v4_CV.pkl' # to store connData_valid_dimReduc_PCA

#connData_train_dimReduc_PCA = []
#connData_valid_dimReduc_PCA = []

# featureDict = {
#        'filename': '208125.mat',
#        'train_or_valid': 'train',
#        'dataMat_feat': dataMat_feat, # dim (32, 32) --> input
#        'DSM_Anxi_T': 50 # the original target value --> output
#        }

#tmpFilePath = 'results/PCA_v4_CV/tmp/'


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


def step15_func(tflag, mat_list, dstDir_CV):
    # tflag = 'train' or 'valid'
    
    for k, filename in enumerate(mat_list): # for each subject
        print("******************deal with*********************")
        print(filename)
        
        all_regions_list = []
        for l in range(379): # for each region
            f_pkl = open(dstDir_CV+'tmp/setp1/dataMat_E_'+tflag+'_region'+str(l)+'.pkl','rb')
            dataMat_El = pickle.load(f_pkl)
            f_pkl.close()
            
            all_regions_list.append(dataMat_El[k,:])
            
        dataMat_subject = np.array(all_regions_list).T # dim (32,379) <-- check shape!!!
        assert(dataMat_subject.shape == (32,379))
        # save dataMat_subject to tmp file:
        f_pkl = open(dstDir_CV+'tmp/setp15/dataMat_'+tflag+'_subject'+str(k)+'.pkl', 'wb')
        pickle.dump(dataMat_subject,f_pkl)
        f_pkl.close()
    
    return


def get_dataMat_F(tflag, mat_list, k, dstDir_CV):
    # tflag = 'train' or 'valid'
    
    dataMat_F_list = []
    
    for j, filename in enumerate(mat_list): # for each subject
        print("&&&&&&&&&&&&&&&&&&deal with&&&&&&&&&&&&&&&&&&&&&")
        print(filename)
        
        fullFileName = dstDir_CV+'tmp/setp15/dataMat_'+tflag+'_subject'+str(j)+'.pkl'
        f_pkl = open(fullFileName,'rb')
        dataMat_subjectj = pickle.load(f_pkl)
        f_pkl.close()
        
        # only get the k-th row:
        connArray_rowk = dataMat_subjectj[k,:]
        dataMat_F_list.append(connArray_rowk)
        
    dataMat_F = np.array(dataMat_F_list) # dim (916, 379) or (102, 379) <-- check shape!!!
    
    return dataMat_F


def step25_func(tflag, mat_list, dstDir_CV):
    # tflag = 'train' or 'valid'
    
    for j, filename in enumerate(mat_list): # for each subject
        print("^^^^^^^^^^^^^^^^^^deal with^^^^^^^^^^^^^^^^^^^^^")
        print(filename)
        
        all_vertices_list = []
        for k in range(new_featDim): # for each new "vertex" feat
            f_pkl = open(dstDir_CV+'tmp/setp2/dataMat_G_'+tflag+'_vertex'+str(k)+'.pkl','rb')
            dataMat_Gk = pickle.load(f_pkl)
            f_pkl.close()
            
            all_vertices_list.append(dataMat_Gk[j,:])
            
        dataMat_subject = np.array(all_vertices_list) # dim (32,32) <-- check shape!!!
        
        featureDict = {
                      'filename': filename,
                      'train_or_valid': tflag,
                      'dataMat_feat': dataMat_subject, # dim (32, 32) --> input
                      #'DSM_Anxi_T': ??? # the original target value --> output
                      }

        if tflag == 'train':
            connData_train_dimReduc_PCA.append(featureDict)
        else:
            connData_valid_dimReduc_PCA.append(featureDict)
    
    return




if __name__ == '__main__':
    
    # load all mat file names into list:
    with open(file_allMatNames) as f:
        content = f.readlines()
    allMatNames_list = [x.strip() for x in content]
    
    for i in range(10): # for each CV fold
        print('$$$$$$$$$$$$$$$$$$$$$$$ CV ' + str(i) + ' $$$$$$$$$$$$$$$$$$$$$$$')
        
        dstDir_CV = dstRootDir + 'CV' + str(i) + '/'
        CV_trainValDir = train_valid_root + 'CV' + str(i) + '/'
        CV_trainDir = CV_trainValDir + 'train_allSubject.txt'
        CV_validDir = CV_trainValDir + 'test_allSubject.txt'
        
        # load valid mat file names into list:
        with open(CV_validDir) as f:
            content_ = f.readlines()
        CV_validNames = [x.strip() for x in content_]
        
        # load training mat file names into list:
        with open(CV_trainDir) as f:
            content__ = f.readlines()
        CV_trainNames = [x.strip() for x in content__]
        
        idx_valid = [z in CV_validNames for z in allMatNames_list]
        idx_train = [z in CV_trainNames for z in allMatNames_list]
        
        """
        # step 1) to save memory (but needs lots of time!) , here we deal with each region one by one and saved to tmp files:
        print('@@@@@@@@@@@@@@@@@@@@@@@@ step 1 @@@@@@@@@@@@@@@@@@@@@@@@')
        for j in range(379): # for each region
            print(j)
            dataMat_D_pklName = dataMat_D_rootDir+'dataMat_D_all_region'+str(j)+'.pkl'
            f_pkl = open(dataMat_D_pklName,'rb')
            dataMat_Dj = pickle.load(f_pkl)
            f_pkl.close()
            assert(dataMat_Dj.shape == (1018, 64984))
            
            # (a) for training set:
            dataMat_Dj_train = dataMat_Dj[idx_train,:]
            # (a1) do PCA on dataMat_D:
            eigenVecs_V3, dataMat_E = myPCA_trainVer(dataMat_Dj_train, new_featDim)
            # eigenVecs_V3.shape: (64984,32); dataMat_E.shape: (916, 32)
            # save dataMat_E to tmp file:
            f_pkl = open(dstDir_CV+'tmp/setp1/dataMat_E_train_region'+str(j)+'.pkl', 'wb')
            pickle.dump(dataMat_E,f_pkl)
            f_pkl.close()
            
            # (b) for valid set:
            dataMat_Dj_valid = dataMat_Dj[idx_valid,:]
            # (b1) do PCA on dataMat_D using eigenVecs_V3 computed from training:
            dataMat_E = np.matmul(dataMat_Dj_valid, eigenVecs_V3) # dim (102, new_featDim)
            # save dataMat_E to tmp file:
            f_pkl = open(dstDir_CV+'tmp/setp1/dataMat_E_valid_region'+str(j)+'.pkl', 'wb')
            pickle.dump(dataMat_E,f_pkl)
            f_pkl.close()
            
            #print()
            
        # step 1.5) concate each region column of above, to get a (32,379) data mat for each subject:
        print('@@@@@@@@@@@@@@@@@@@@@@@@ step 1.5 @@@@@@@@@@@@@@@@@@@@@@@@')
        step15_func('train', CV_trainNames, dstDir_CV)
        step15_func('valid', CV_validNames, dstDir_CV)
        
        # step 2) need to load&concate the matrices above:
        print('@@@@@@@@@@@@@@@@@@@@@@@@ step 2 @@@@@@@@@@@@@@@@@@@@@@@@')
        for k in range(new_featDim): # ie 32: for each new "vertex" feat
            # (a) for train:
            dataMat_F = get_dataMat_F('train',CV_trainNames, k, dstDir_CV) # for train
            assert(dataMat_F.shape == (len(CV_trainNames), 379))
            # (a2) do PCA on dataMat_F:
            eigenVecs_V4, dataMat_G = myPCA_trainVer(dataMat_F, new_featDim) # <-- check shape!!!
            # save dataMat_G to tmp file:
            f_pkl = open(dstDir_CV+'tmp/setp2/dataMat_G_train_vertex'+str(k)+'.pkl', 'wb')
            pickle.dump(dataMat_G,f_pkl)
            f_pkl.close()
            
            # (b) for valid:
            dataMat_F = get_dataMat_F('valid',CV_validNames, k, dstDir_CV) # for valid
            assert(dataMat_F.shape == (len(CV_validNames), 379))
            # (b2) do PCA on dataMat_F using eigenVecs_V4 computed from training:
            dataMat_G = np.matmul(dataMat_F, eigenVecs_V4) # dim (306, new_featDim) <-- check shape!!!
            # save dataMat_G to tmp file:
            f_pkl = open(dstDir_CV+'tmp/setp2/dataMat_G_valid_vertex'+str(k)+'.pkl', 'wb')
            pickle.dump(dataMat_G,f_pkl)
            f_pkl.close()
        """    
            
        # step 2.5) get final result:
        # concate each new "region" row of above, to get a (32,32) data mat for each subject and save:
        print('@@@@@@@@@@@@@@@@@@@@@@@@ step 2.5 @@@@@@@@@@@@@@@@@@@@@@@@')
        connData_train_dimReduc_PCA = []
        connData_valid_dimReduc_PCA = []
        step25_func('train', CV_trainNames, dstDir_CV)
        step25_func('valid', CV_validNames, dstDir_CV)
        
        # save final results (BUT WITHOUT target values!!! will add them later) to dst dir pkl:
        f_pkl = open(dstDir_CV+dstPklName_trainData, 'wb')
        pickle.dump(connData_train_dimReduc_PCA,f_pkl)
        f_pkl.close()
        
        f_pkl = open(dstDir_CV+dstPklName_validData, 'wb')
        pickle.dump(connData_valid_dimReduc_PCA,f_pkl)
        f_pkl.close()
        
        
        


