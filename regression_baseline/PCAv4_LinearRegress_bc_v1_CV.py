#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:09:58 2021

@author: guo.1648
"""

# version 1 for 10-fold Cross Validation:

# binary classification (regression) version:
# thresh50: 0 (target==50) v.s. 1 (target>50)
# thresh51: 0 (target==51) v.s. 1 (target>51) <-- NOT did! Target is biased: lots of 1

# This is for linear regression (as baseline) with 2PCA_v4 connData as input

# referenced from PCAv4_LinearRegress_bc_v1.py

import os
import scipy.io as sio
import numpy as np
import math
import pickle

from sklearn.linear_model import Lasso


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/'
train_test_dir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/'
target_behavior_name = 'DSM_Anxi_T'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/regression_baseline/results/DSM_Anxi_T/PCAv4_LinearRegress_bc_v1_CV/'


def loadDataTarget_PCA(srcDir,train_test_root, flag):
    # flag = 'train' or 'test'
    
    f_pkl = open(srcDir,'rb')
    dataMat_subject_all_dict = pickle.load(f_pkl)
    f_pkl.close()
    
    conn_list = []
    behavior_val_list = []
    
    mat_ids = [line.rstrip() for line in open(os.path.join(train_test_root, flag+'.txt'))]
    for filename in mat_ids:
        #print("------------------deal with---------------------")
        #print(filename)
        
        target_dict = dataMat_subject_all_dict[filename]
        
        if flag == 'train':
            assert(target_dict['train_or_valid'] == flag)
        else:
            assert(target_dict['train_or_valid'] == 'valid')
        
        dataMat_feat = target_dict['dataMat_feat'] # 32x32
        behavior_val = target_dict[target_behavior_name]
        
        assert(behavior_val is not None)
        
        # binary classification target:
        if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
            if behavior_val == 50: # 51
                behavior_val = 0
            else:
                behavior_val = 1
        
        conn_list.append(dataMat_feat.flatten())
        behavior_val_list.append(behavior_val)
        
        #print()
    
    return (conn_list, behavior_val_list)


def lasso_searchPenalty_wrapper(X_train,y_train,X_test,y_test, penalty_list):
    test_score_differentPenalty = []
    
    for penalty_ in penalty_list:
        #print("try penalty = " + str(penalty)) # just for debug!
        lasso_ = Lasso(alpha=penalty_,max_iter=10e5)
        lasso_.fit(X_train,y_train)
        test_score=lasso_.score(X_test, y_test)
        test_score_differentPenalty.append(test_score)
    
    max_test_score_idx = np.argmax(test_score_differentPenalty)
    #max_test_score = test_score_differentPenalty[max_test_score_idx]
    max_test_score_penalty = penalty_list[max_test_score_idx]
    
    return max_test_score_penalty




if __name__ == '__main__':
    penalty_list = [1, 0.1, 0.01, 0.001, 0.0001]
    
    acc_train_list = []
    train_score_list = []
    acc_test_list = []
    test_score_list = []
    
    for i in range(10): # for each CV fold
        CVfold = 'CV' + str(i) + '/'
        print('********************* ' + CVfold)
        
        srcDir = srcRootDir + CVfold + 'connData_dimReduc_PCA_dict_v4_CV.pkl'
        train_test_root = train_test_dir + CVfold
        dstDir = dstRootDir + CVfold
        
        conn_list_train, behavior_val_list_train = loadDataTarget_PCA(srcDir,train_test_root, flag = 'train')
        conn_list_test, behavior_val_list_test = loadDataTarget_PCA(srcDir,train_test_root, flag = 'test')
        
        X_train = np.array(conn_list_train) # 913x1024
        y_train = behavior_val_list_train
        X_test = np.array(conn_list_test) # 102x1024
        y_test = behavior_val_list_test
        
        # search for the penalty:
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
        Y_train_pred = np.rint(Y_train_pred)
        Y_train_pred = np.clip(Y_train_pred, a_min = min(y_train), a_max = max(y_train)) 
        
        acc_train = sum(Y_train_pred==y_train)/len(Y_train_pred)
        
        train_score=lasso.score(X_train, y_train)
        
        print("training accuracy = " + str(acc_train))
        print("training score: " + str(train_score))
        acc_train_list.append(acc_train)
        train_score_list.append(train_score)
        
        # for testing:
        print("---test---")
        Y_test_pred = lasso.predict(X_test)
        Y_test_pred = np.rint(Y_test_pred)
        Y_test_pred = np.clip(Y_test_pred, a_min = min(y_test), a_max = max(y_test)) 
        
        acc_test = sum(Y_test_pred==y_test)/len(Y_test_pred)
        
        test_score=lasso.score(X_test, y_test)
        
        print("testing accuracy = " + str(acc_test))
        print("testing score: " + str(test_score))
        acc_test_list.append(acc_test)
        test_score_list.append(test_score)
        
        # save the trained model:
        f_pkl = open(dstDir+'trained_lasso_model.pkl', 'wb')
        pickle.dump(lasso,f_pkl)
        f_pkl.close()
        
        """
        ## code to load model:
        f_pkl = open(dstDir+'trained_lasso_model.pkl','rb')
        lasso_load = pickle.load(f_pkl)
        f_pkl.close()
        """
        
        #print()
    
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Metrics:')
    
    acc_train_mean = np.mean(acc_train_list)
    acc_train_std = np.std(acc_train_list)
    print('acc_train_mean = {:.2f}%'.format(acc_train_mean*100))
    print('acc_train_std = {:.4f}'.format(acc_train_std))
    train_score_mean = np.mean(train_score_list)
    train_score_std = np.std(train_score_list)
    print('train_score_mean = {:.4f}'.format(train_score_mean))
    print('train_score_std = {:.4f}'.format(train_score_std))
    
    acc_test_mean = np.mean(acc_test_list)
    acc_test_std = np.std(acc_test_list)
    print('acc_test_mean = {:.2f}%'.format(acc_test_mean*100))
    print('acc_test_std = {:.4f}'.format(acc_test_std))
    test_score_mean = np.mean(test_score_list)
    test_score_std = np.std(test_score_list)
    print('test_score_mean = {:.4f}'.format(test_score_mean))
    print('test_score_std = {:.4f}'.format(test_score_std))








