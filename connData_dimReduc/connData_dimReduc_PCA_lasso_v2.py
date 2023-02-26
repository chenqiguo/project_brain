#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:29:38 2021

@author: guo.1648
"""

# my 2nd try: dimensionality reduction on connData: PCA:
# do PCA independently on each connData(i.e., connL vertically concatenate connR),
# to reduce feature dim from 379 -> 100? 50? 25?
# finally using the eigenVal and flatten eigenVec as new features (into ML algorithm or multiple FC).

# Note: eigenVals are in descending orders!
# To save space, NOT saved the pkl file! Instead, compute the PC while running.

# referenced from feature124_lasso_v2.py and connData_dimReduc_PCA_v1.py.


import numpy as np
import os
#import pickle
import scipy.io as sio

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import statistics
from scipy import stats


new_featDim = 50 #100? 50? 25? 10? 5?

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/'
target_behavior_name = 'DSM_Anxi_T'

train_test_root = '/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/'


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


def generateDataTarget_PCA_v2(srcRootDir, new_featDim, target_behavior_name, mat_list):
    X_all_list = []
    y_all_list = []
    
    for filename in mat_list:
        
        print("------------------deal with---------------------")
        print(filename)
        
        fullFileName = srcRootDir + filename
        mat_contents = sio.loadmat(fullFileName)
        
        # (1) for connData:
        connL = mat_contents['connL'] # (32492, 379)
        connR = mat_contents['connR'] # (32492, 379)
        connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
        # do the PCA on connArray:
        dataMat_feat, eigenValues, eigenVectors = myPCA(connArray, new_featDim)
                
        eigenVecs_flat_list = eigenVectors.flatten('F').tolist()
        eigenVals_list = eigenValues.tolist()
        
        X_list = eigenVals_list + eigenVecs_flat_list
        X_all_list.append(X_list)
        
        # (2) for target values:
        behavior_val = None
        behavior_names = mat_contents['behavior_names']
        for i,element in enumerate(behavior_names[0]):
            this_behavior_name = element[0]
            if this_behavior_name == target_behavior_name:
                behavior_val = mat_contents['behavior'][0][i]
                break
        assert(behavior_val is not None)
        y_all_list.append(behavior_val)
        
    X = np.array(X_all_list)
    y = np.array(y_all_list)
    
    return (X, y)


def zScoreNormalization(X):
    X_normed = np.empty(X.shape)
    # z-score standardization on data X:
    # just subtract the mean and divided by s.t.d for each feature column
    feature_num = X.shape[1]
    for feature_idx in range(feature_num):
        this_feature = X[:,feature_idx]
        this_mean = np.mean(this_feature)
        this_std = np.std(this_feature)
        if this_std < 1e-8:
            this_std = 1e-8
        normed_feature = (this_feature-this_mean)/this_std
        X_normed[:,feature_idx] = normed_feature
    
    return X_normed


def generateTrainTestDataTarget_PCA_v2(srcRootDir, new_featDim, target_behavior_name, train_test_root):
    
    mat_ids = {}
    mat_ids['train'] = [line.rstrip() for line in open(os.path.join(train_test_root, 'train.txt'))]
    mat_ids['test'] = [line.rstrip() for line in open(os.path.join(train_test_root, 'test.txt'))]
    
    X_train, y_train = generateDataTarget_PCA_v2(srcRootDir, new_featDim, target_behavior_name, mat_ids['train'])
    X_test, y_test = generateDataTarget_PCA_v2(srcRootDir, new_featDim, target_behavior_name, mat_ids['test'])
    
    return (X_train, y_train, X_test, y_test)


def lassoMultiRegress_searchPenalty(X, Y, penalty):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=31)
    
    lasso = Lasso(alpha=penalty,max_iter=10e5) #alpha=0.0001 # NOT use: normalize=True
    lasso.fit(X_train,Y_train)
    test_score=lasso.score(X_test, Y_test)
    
    return test_score


def lasso_searchPenalty_wrapper(X, y, penalty_list):
    
    test_score_differentPenalty = []
    for penalty in penalty_list:
        #print("try penalty = " + str(penalty)) # just for debug!
        test_scores = []
        for t in range(1):
            test_score = lassoMultiRegress_searchPenalty(X, y, penalty)
            test_scores.append(test_score)
        test_score_differentPenalty.append(np.mean(test_scores))
    
    max_test_score_idx = np.argmax(test_score_differentPenalty)
    max_test_score = test_score_differentPenalty[max_test_score_idx]
    max_test_score_penalty = penalty_list[max_test_score_idx]
    print("max testing score: " + str(max_test_score))
    print("corresponding penalty: " + str(max_test_score_penalty)) # --> 0.001
    
    return max_test_score_penalty


def lassoMultiRegress_trainTest(X_train, y_train, X_test, y_test, penalty):
    lasso = Lasso(alpha=penalty,max_iter=10e5) #alpha=0.0001 # NOT use: normalize=True
    lasso.fit(X_train,y_train)
    
    coeff_used = np.sum(lasso.coef_!=0)
    print("number of features used: " + str(coeff_used))
    #print("lasso coefficients are:")
    #print(lasso.coef_)
    
    # for train
    print("---train---")
    # baseline: predict the majority
    counts = np.bincount(y_train.astype(int))
    majority = np.argmax(counts)
    baseline_acc = sum(y_train==majority)/len(y_train)
    print("baseline training accuracy = " + str(baseline_acc))
    
    y_train_pred = lasso.predict(X_train)
    y_train_pred = np.rint(y_train_pred)
    acc_train = sum(y_train_pred==y_train)/len(y_train_pred)
    
    train_score=lasso.score(X_train, y_train)
    
    print("training accuracy = " + str(acc_train))
    print("training score: " + str(train_score))
    
    # for test
    print("---test---")
    # baseline: predict the majority
    counts = np.bincount(y_test.astype(int))
    majority = np.argmax(counts)
    baseline_acc = sum(y_test==majority)/len(y_test)
    print("baseline accuracy = " + str(baseline_acc))
    
    y_test_pred = lasso.predict(X_test)
    y_test_pred = np.rint(y_test_pred)
    acc_test = sum(y_test_pred==y_test)/len(y_test_pred)
    
    test_score=lasso.score(X_test, y_test)
    
    print("testing accuracy = " + str(acc_test))
    print("testing score: " + str(test_score))
    
    return test_score



if __name__ == '__main__':
    penalty_list = [1, 0.1, 0.01, 0.001, 0.0001] #, 0.00001
    
    X_train, y_train, X_test, y_test = generateTrainTestDataTarget_PCA_v2(srcRootDir, new_featDim, target_behavior_name, train_test_root)
    
    # NEED to do this!: use z-score standardization to normalize X
    X_train_nonNorm = X_train # store the non-normalized data X
    X_train = zScoreNormalization(X_train)
    X_test_nonNorm = X_test # store the non-normalized data X
    X_test = zScoreNormalization(X_test)
    
    # do binary classification instead: 0: target==50; 1: target>50
    y_train_bc = (y_train>50).astype(int)
    y_test_bc = (y_test>50).astype(int)
    
    max_test_score_penalty = lasso_searchPenalty_wrapper(X_train, y_train_bc, penalty_list)
    penalty = max_test_score_penalty
    
    test_score = lassoMultiRegress_trainTest(X_train, y_train_bc, X_test, y_test_bc, penalty)
    
    

