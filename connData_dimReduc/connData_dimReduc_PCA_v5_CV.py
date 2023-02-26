#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 19:34:27 2021

@author: guo.1648
"""

# for the symmetrical MMPconnSparse dataset.

# referenced from connData_dimReduc_PCA_v4_CV.py


import numpy as np
import os
import pickle
import scipy.io as sio
from scipy.sparse.linalg import eigsh


new_featDim = 32 # decided from code connData_dimReduc_PCA_v3_decideNewFeatDim.py

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/'
train_test_root = srcRootDir + 'mat_names_CV/'
dataMat_D_rootDir = srcRootDir + 'dataMat_D_symm/'
file_allMatNames = srcRootDir + 'dataMat_D_symmMatNames.txt'

dstRootDir = 'results/PCA_v5_CV/'
dstPklName_trainData = 'connData_train_dimReduc_PCA_v5_CV.pkl' # to store connData_train_dimReduc_PCA
dstPklName_testData = 'connData_test_dimReduc_PCA_v5_CV.pkl' # to store connData_test_dimReduc_PCA



def myPCA_trainVer(dataMat, new_featDim):
    # for example: dataMat = dataMat_D for train: (709, 379) dim
    
    # calculate the mean of each column:
    M = np.mean(dataMat.T,axis=1)
    # center columns by subtracting column means:
    C = dataMat - M
    # calculate covariance matrix of centered matrix:
    V = np.cov(C.T) # (379, 379) dim
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
    # tflag = 'train' or 'test'
    
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
    # tflag = 'train' or 'test'
    
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
    # tflag = 'train' or 'test'
    
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
    
    # load all mat file names into list:
    with open(file_allMatNames) as f:
        content = f.readlines()
    allMatNames_list = [x.strip() for x in content]
    
    for i in range(10): # for each CV fold
        print('$$$$$$$$$$$$$$$$$$$$$$$ CV ' + str(i) + ' $$$$$$$$$$$$$$$$$$$$$$$')
        
        dstDir_CV = dstRootDir + 'CV' + str(i) + '/'
        if not os.path.exists(dstDir_CV):
            os.makedirs(dstDir_CV)
        if not os.path.exists(dstDir_CV+'tmp/'):
            os.makedirs(dstDir_CV+'tmp/')
        if not os.path.exists(dstDir_CV+'tmp/setp1/'):
            os.makedirs(dstDir_CV+'tmp/setp1/')
        if not os.path.exists(dstDir_CV+'tmp/setp15/'):
            os.makedirs(dstDir_CV+'tmp/setp15/')
        if not os.path.exists(dstDir_CV+'tmp/setp2/'):
            os.makedirs(dstDir_CV+'tmp/setp2/')
        
        CV_trainTestDir = train_test_root + 'CV' + str(i) + '/'
        CV_trainDir = CV_trainTestDir + 'train_allSubject.txt'
        CV_testDir = CV_trainTestDir + 'test_allSubject.txt'
        
        # load test mat file names into list:
        with open(CV_testDir) as f:
            content_ = f.readlines()
        CV_testNames = [x.strip() for x in content_]
        
        # load training mat file names into list:
        with open(CV_trainDir) as f:
            content__ = f.readlines()
        CV_trainNames = [x.strip() for x in content__]
        
        idx_test = [z in CV_testNames for z in allMatNames_list]
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
            assert(dataMat_Dj.shape == (1018, 379))
            
            # (a) for training set:
            dataMat_Dj_train = dataMat_Dj[idx_train,:]
            # (a1) do PCA on dataMat_D:
            eigenVecs_V3, dataMat_E = myPCA_trainVer(dataMat_Dj_train, new_featDim)
            # eigenVecs_V3.shape: (379,32); dataMat_E.shape: (916, 32)
            # save dataMat_E to tmp file:
            f_pkl = open(dstDir_CV+'tmp/setp1/dataMat_E_train_region'+str(j)+'.pkl', 'wb')
            pickle.dump(dataMat_E,f_pkl)
            f_pkl.close()
            
            # (b) for test set:
            dataMat_Dj_test = dataMat_Dj[idx_test,:]
            # (b1) do PCA on dataMat_D using eigenVecs_V3 computed from training:
            dataMat_E = np.matmul(dataMat_Dj_test, eigenVecs_V3) # dim (102, new_featDim)
            # save dataMat_E to tmp file:
            f_pkl = open(dstDir_CV+'tmp/setp1/dataMat_E_test_region'+str(j)+'.pkl', 'wb')
            pickle.dump(dataMat_E,f_pkl)
            f_pkl.close()
            
            #print()
        """
        
        """
        # step 1.5) concate each region column of above, to get a (32,379) data mat for each subject:
        print('@@@@@@@@@@@@@@@@@@@@@@@@ step 1.5 @@@@@@@@@@@@@@@@@@@@@@@@')
        step15_func('train', CV_trainNames, dstDir_CV)
        step15_func('test', CV_testNames, dstDir_CV)
        """
        
        """
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
            
            # (b) for test:
            dataMat_F = get_dataMat_F('test',CV_testNames, k, dstDir_CV) # for test
            assert(dataMat_F.shape == (len(CV_testNames), 379))
            # (b2) do PCA on dataMat_F using eigenVecs_V4 computed from training:
            dataMat_G = np.matmul(dataMat_F, eigenVecs_V4) # dim (306, new_featDim) <-- check shape!!!
            # save dataMat_G to tmp file:
            f_pkl = open(dstDir_CV+'tmp/setp2/dataMat_G_test_vertex'+str(k)+'.pkl', 'wb')
            pickle.dump(dataMat_G,f_pkl)
            f_pkl.close()
        """
        
        #"""
        # step 2.5) get final result:
        # concate each new "region" row of above, to get a (32,32) data mat for each subject and save:
        print('@@@@@@@@@@@@@@@@@@@@@@@@ step 2.5 @@@@@@@@@@@@@@@@@@@@@@@@')
        connData_train_dimReduc_PCA = []
        connData_test_dimReduc_PCA = []
        step25_func('train', CV_trainNames, dstDir_CV)
        step25_func('test', CV_testNames, dstDir_CV)
        
        # save final results (BUT WITHOUT target values!!! will add them later) to dst dir pkl:
        f_pkl = open(dstDir_CV+dstPklName_trainData, 'wb')
        pickle.dump(connData_train_dimReduc_PCA,f_pkl)
        f_pkl.close()
        
        f_pkl = open(dstDir_CV+dstPklName_testData, 'wb')
        pickle.dump(connData_test_dimReduc_PCA,f_pkl)
        f_pkl.close()
        #"""
        




