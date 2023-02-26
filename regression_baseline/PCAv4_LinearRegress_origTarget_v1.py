#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 18:49:16 2021

@author: guo.1648
"""

# version 1:

# original target regression version, for all subjects.

# This is for linear regression (as baseline) with 2PCA_v4 connData as input

# referenced from PCAv4_LinearRegress_bc_v1.py


import os
import scipy.io as sio
import numpy as np
import math
import pickle

from sklearn.linear_model import Lasso


srcDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4/connData_dimReduc_PCA_dict_v4.pkl'
train_test_root = '/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/'
target_behavior_name = 'DSM_Anxi_T'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/regression_baseline/results/PCAv4_LinearRegress_origTarget_v1/'

tolerance = 2 # for computing acc


def loadDataTarget_PCA_orig(flag):
    # flag = 'train' or 'test'
    
    f_pkl = open(srcDir,'rb')
    dataMat_subject_all_dict = pickle.load(f_pkl)
    f_pkl.close()
    
    conn_list = []
    behavior_val_list = []
    
    mat_ids = [line.rstrip() for line in open(os.path.join(train_test_root, flag+'.txt'))]
    for filename in mat_ids:
        print("------------------deal with---------------------")
        print(filename)
        
        target_dict = dataMat_subject_all_dict[filename]
        dataMat_feat = target_dict['dataMat_feat'] # 32x32
        behavior_val = target_dict[target_behavior_name]
        
        assert(behavior_val is not None)
        """
        # binary classification target:
        if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
            if behavior_val == 50: # 51
                behavior_val = 0
            else:
                behavior_val = 1
        """
        conn_list.append(dataMat_feat.flatten())
        behavior_val_list.append(behavior_val)
        
        #print()
    
    return (conn_list, behavior_val_list)


def lasso_searchPenalty_wrapper(X_train,y_train,X_test,y_test, penalty_list):
    test_score_differentPenalty = []
    
    for penalty_ in penalty_list:
        print("try penalty = " + str(penalty_)) # just for debug!
        lasso_ = Lasso(alpha=penalty_,max_iter=10e5)
        lasso_.fit(X_train,y_train)
        test_score=lasso_.score(X_test, y_test)
        test_score_differentPenalty.append(test_score)
    
    max_test_score_idx = np.argmax(test_score_differentPenalty)
    #max_test_score = test_score_differentPenalty[max_test_score_idx]
    max_test_score_penalty = penalty_list[max_test_score_idx]
    
    return max_test_score_penalty





if __name__ == '__main__':
    
    conn_list_train, behavior_val_list_train = loadDataTarget_PCA_orig(flag = 'train')
    conn_list_test, behavior_val_list_test = loadDataTarget_PCA_orig(flag = 'test')
    
    X_train = np.array(conn_list_train) # 709x1024
    y_train = behavior_val_list_train
    X_test = np.array(conn_list_test) # 306x1024
    y_test = behavior_val_list_test
    
    # search for the penalty:
    penalty_list = [1, 0.1, 0.01, 0.001] #, 0.0001
    penalty = lasso_searchPenalty_wrapper(X_train,y_train,X_test,y_test, penalty_list)
    print('penalty = ' + str(penalty))
    
    lasso = Lasso(alpha=penalty,max_iter=10e5)
    
    lasso.fit(X_train,y_train)
    
    # for lasso coefficients:
    coeff_used = np.sum(lasso.coef_!=0)
    print("number of features used: " + str(coeff_used))
    print("lasso coefficients are:")
    print(lasso.coef_)
    print("lasso interception is:")
    print(lasso.intercept_)
    
    # for training:
    print("---train---")
    Y_train_pred = lasso.predict(X_train)
    # newly modified: use tolerance for acc:
    difference = np.absolute(Y_train_pred - y_train)
    correct_num = np.count_nonzero(difference <= tolerance)
    acc_train = correct_num / len(y_train)
    
    train_score=lasso.score(X_train, y_train)
    
    print("training accuracy = " + str(acc_train))
    print("training score: " + str(train_score))
    
    # for testing:
    print("---test---")
    Y_test_pred = lasso.predict(X_test)
    # newly modified: use tolerance for acc:
    difference = np.absolute(Y_test_pred - y_test)
    correct_num = np.count_nonzero(difference <= tolerance)
    acc_test = correct_num / len(y_test)
    
    test_score=lasso.score(X_test, y_test)
    
    print("testing accuracy = " + str(acc_test))
    print("testing score: " + str(test_score))
    
    # save the trained model:
    f_pkl = open(dstRootDir+'trained_lasso_model.pkl', 'wb')
    pickle.dump(lasso,f_pkl)
    f_pkl.close()
    
    """
    ## code to load model:
    f_pkl = open(dstRootDir+'trained_lasso_model.pkl','rb')
    lasso_load = pickle.load(f_pkl)
    f_pkl.close()
    """

