#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:20:25 2021

@author: guo.1648
"""

# referenced from fnn_origTarget_CV_v3_tuneParam.py, BUT WITH modifications!!!!!

# for all the original targets at once: do z-score normalization on y, and use
# fully-connected neural network (FNN) (v3) code for 10-fold Cross Validation.
# This code tune the hyper-parameters and save the corresponding hyper-params & metrics.

# Flatten 2PCA connData (v5) as input; predict all original targets at once;
# on 10-fold Cross Validation.


import os
import random
import pickle
import scipy.io as sio
import numpy as np
import math
from scipy import stats
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from keras import backend as K
from keras.optimizers import SGD

from fnn_model_v3 import hcp_fnn


srcRootDir_sparse = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnSparse/'

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v5_CV/'
pklFileName = 'connData_dimReduc_PCA_dict_v5_CV.pkl'

train_test_dir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/fullyConnectNet/FNN/result/fnn_origTarget_CV_v5/'

# in total of 35 behavior targets:
all_target_list = ['DSM_Anxi_T', 'ASR_Anxd_Pct', 'DSM_Depr_T', 'DSM_Adh_T', 'ASR_Attn_T',
                   'DSM_Avoid_T', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'CogTotalComp_Unadj',
                   'PMAT24_A_CR', 'PMAT24_A_RTCR', 'VSPLOT_TC', 'VSPLOT_CRTE', 'VSPLOT_OFF',
                   'ER40_CR', 'ER40_CRT', 'SCPT_SEN', 'SCPT_SPEC', 'SCPT_TPRT', 'CardSort_Unadj',
                   'Flanker_Unadj', 'ProcSpeed_Unadj', 'DDisc_AUC_200', 'DDisc_AUC_40K',
                   'PicSeq_Unadj', 'ReadEng_Unadj', 'PicVocab_Unadj', 'IWRD_TOT', 'IWRD_RTC',
                   'ListSort_Unadj', 'NEOFAC_A', 'NEOFAC_O', 'NEOFAC_C', 'NEOFAC_N', 'NEOFAC_E']



def check_targetNan(mat_contents):
    # check if any behavior_val is NaN
    
    for target_name in all_target_list:
        behavior_val = mat_contents[target_name][0][0]
        if math.isnan(behavior_val):
            return False
    
    return True


def removeNan(mat_ids_tmp):
    # remove those mat files whose target value is Nan:
    
    mat_ids = []
    
    for filename in mat_ids_tmp:
        fullFileName = srcRootDir_sparse + filename # full name of the mat file
        mat_contents = sio.loadmat(fullFileName)
        
        # check if any behavior_val is NaN:
        validFlag = check_targetNan(mat_contents)
        if validFlag:
            mat_ids.append(filename)
    
    return mat_ids


def check_targetNan_newVer5(target_dict):
    # check if any behavior_val is NaN
    
    for target_name in all_target_list:
        behavior_val = target_dict[target_name][0][0]
        if math.isnan(behavior_val):
            return False
    
    return True


def get_all_targets_newVer5(target_dict):
    behavior_val_list = []
    for target_name in all_target_list:
        behavior_val = target_dict[target_name][0][0]
        assert(not math.isnan(behavior_val))
        behavior_val_list.append(behavior_val)
    return behavior_val_list


def loadDataAlltarget_PCAver5(srcDir, mat_ids):
    
    # load the src pkls:
    f_pkl = open(srcDir,'rb')
    dataMat_subject_all_dict = pickle.load(f_pkl)
    f_pkl.close()
    
    X = []
    y = []
    
    for filename in mat_ids:
        target_dict = dataMat_subject_all_dict[filename]
        
        # check if any behavior_val is NaN:
        validFlag = check_targetNan_newVer5(target_dict)
        if validFlag:
            
            dataMat_feat = target_dict['dataMat_feat']
            dataMat_feat_flat = dataMat_feat.flatten() # dim: (1024,)
            X.append(dataMat_feat_flat)
            
            behavior_val_list = get_all_targets_newVer5(target_dict)
            y.append(behavior_val_list)
    
    return (np.array(X), np.array(y))


def zscore_norm_fnn(y_train_nonNorm, y_valid_nonNorm):
    t_mu = y_train_nonNorm.mean(axis=0)
    t_sigma = y_train_nonNorm.std(axis=0)
    
    # perform z-norm:
    y_train = y_train_nonNorm - t_mu[np.newaxis, :]
    y_train = y_train / t_sigma[np.newaxis, :]
    
    y_valid = y_valid_nonNorm - t_mu[np.newaxis, :]
    y_valid = y_valid / t_sigma[np.newaxis, :]
    
    return (y_train, y_valid, t_sigma)


def fit_one_cycle_tune_1epoch(model, X, y, y_sigma):
    
    y_pred = model.predict(X, batch_size=batch_size, verbose=0)
    
    cor = np.zeros((y.shape[-1]))
    mae = np.zeros((y.shape[-1]))
    
    for i in range(y.shape[-1]):
        
        # newly added: remove Nan from y_pred[:, i] and y[:, i]:
        nanIdx_list1 = np.argwhere(np.isnan(y_pred[:, i])).flatten().tolist()
        nanIdx_list2 = np.argwhere(np.isnan(y[:, i])).flatten().tolist()
        nanIdx_list_all = list(set(nanIdx_list1 + nanIdx_list2))
        
        y_pred_i_NoNan = np.delete(y_pred[:, i], nanIdx_list_all)
        y_i_NoNan = np.delete(y[:, i], nanIdx_list_all)
        
        if len(y_pred_i_NoNan) == 0:
            cor[i] = np.nan
            mae[i] = np.nan
        else:
            cor[i] = pearsonr(y_pred_i_NoNan, y_i_NoNan)[0]
            assert(y_sigma is not None)
            mae[i] = np.mean(np.abs(y_pred_i_NoNan - y_i_NoNan)) * y_sigma[i]
    
    return (cor, mae)


def tune_params_func_10foldVer(train_test_dir, srcRootDir):
    # referenced from func tune_params_func_newVer()
    
    print('@@@tuning hyper-params...')
    
    best_params = {'best_lr': -100, # just dummy values!
                   'best_lr_decay': -100,
                   'best_dropout': -100,
                   'best_n_l1': -100,
                   'best_n_l2': -100,
                   'best_n_l3': -100}
    best_Pearson_r_val = -100 # store the highest one!
    
    # txt file name to save the metrics & hyper-params:
    metrics_txtFile = dstRootDir + 'param_metrics_from10folds.txt'
    
    for lr in lr_list:
        for lr_decay in lr_decay_list:
            for dropout in dropout_list:
                for n_l1 in n_l1_list:
                    for n_l2 in n_l2_list:
                        for n_l3 in n_l3_list:
                            
                            if (n_l2 >= n_l1) or (n_l3 >= n_l2):
                                continue
                            
                            param_str = 'lr = ' + str(lr) + '; lr_decay = ' + str(lr_decay) \
                                        + '; dropout = ' + str(dropout) + '; n_l1 = ' + str(n_l1) \
                                        + '; n_l2 = ' + str(n_l2) + '; n_l3 = ' + str(n_l3)
                            Pearson_r_val_list = []
                            
                            for i in range(10): # for each CV fold
                                CVfold = 'CV' + str(i) + '/'
                                print('********************* ' + CVfold)
                                train_test_root = train_test_dir + CVfold
                                srcDir = srcRootDir + CVfold + pklFileName
                                
                                K.clear_session()
                                
                                # load the mat names of train & test set for this testFold_idx i:
                                trainallMat_ids_tmp = [line.rstrip() for line in open(os.path.join(train_test_root, 'train_allSubject.txt'))]
                                testMat_ids_tmp = [line.rstrip() for line in open(os.path.join(train_test_root, 'test_allSubject.txt'))]
                                # newly added: remove those mat files whose target value is Nan:
                                trainallMat_ids = removeNan(trainallMat_ids_tmp)
                                testMat_ids = removeNan(testMat_ids_tmp)
                                
                                # newly modified: load the vectorized data & targets:
                                X_train, y_train_nonNorm = loadDataAlltarget_PCAver5(srcDir, trainallMat_ids) # X_train.shape: (895, 1024)
                                X_test, y_test_nonNorm = loadDataAlltarget_PCAver5(srcDir, testMat_ids)
                                
                                # z normalize y data based on training set:
                                y_train, y_test, y_sigma = zscore_norm_fnn(y_train_nonNorm, y_test_nonNorm)
                                
                                # initialize model:
                                model = hcp_fnn(X_train.shape[1], n_measure, n_l1, n_l2,
                                                n_l3, dropout, l2_regularizer)
                                optimizer = SGD(lr=lr, momentum=momentum, decay=lr_decay, nesterov=False)
                                model.compile(loss='mean_squared_error', optimizer=optimizer)
                                
                                best_validPearson_r = -100 # the highest; used for tuning hyper-params
                                
                                for epoch in range(epochs):
                                    print('** epoch ' + str(epoch))
                                    
                                    # fit model:
                                    model.fit(X_train, y_train, epochs=1, batch_size=batch_size,
                                              verbose=0, validation_data=(X_test, y_test))
                                    
                                    cor, _ = fit_one_cycle_tune_1epoch(model, X_test, y_test, y_sigma)
                                    
                                    # use the mean of cor as best_validPearson_r_j:
                                    cor = cor[~np.isnan(cor)] # ??? can I do this?
                                    best_validPearson_r_j = np.mean(cor)
                                    
                                    print('this epoch\'s best_validPearson_r_j = ' + str(best_validPearson_r_j))
                                    
                                    if best_validPearson_r_j > best_validPearson_r:
                                        best_validPearson_r = best_validPearson_r_j
                                    
                                print('this best_validPearson_r = ' + str(best_validPearson_r)) # the highest mean across 35 targets!
                                Pearson_r_val_list.append(best_validPearson_r)
                                
                            Pearson_r_val_mean = np.mean(Pearson_r_val_list)
                            print_str = param_str + '; Pearson_r_val_mean = {:.4f}'.format(Pearson_r_val_mean)
                            print(print_str)
                            
                            # also save the metrics and hyper-params to txt file:
                            if not os.path.exists(metrics_txtFile):
                                file1 = open(metrics_txtFile,"w")#write mode
                                file1.write(print_str + "\n")
                                file1.close()
                            else:
                                file1 = open(metrics_txtFile,"a")#append mode
                                file1.write(print_str + "\n")
                                file1.close()
                            
                            if Pearson_r_val_mean > best_Pearson_r_val:
                                best_Pearson_r_val = Pearson_r_val_mean
                                best_params['best_lr'] = lr
                                best_params['best_lr_decay'] = lr_decay
                                best_params['best_dropout'] = dropout
                                best_params['best_n_l1'] = n_l1
                                best_params['best_n_l2'] = n_l2
                                best_params['best_n_l3'] = n_l3
                                
    return (best_params, best_Pearson_r_val)
    
    
    
    
    
    




if __name__ == '__main__':
    
    random.seed(10)
    np.random.seed(10)
    
    epochs = 50
    batch_size = 50
    momentum = 0.9
    l2_regularizer = 0.02
    n_measure = len(all_target_list)
    
    # hyper-params to be tuned:
    lr_list = [0.01, 0.1, 0.3] # 
    lr_decay_list = [1e-7, 1e-4, 1e-2] # 
    dropout_list = [0.2, 0.4, 0.5] #
    n_l1_list = [64, 32, 16] #
    n_l2_list = [32, 16, 8] # 
    n_l3_list = [16, 8, 4] # 
    
    # here we predict all the targets at once!!!
    
    # newly modified: for tuning the hyper-params:
    # since our connData PCA_v5 were generated based on the 10-fold CV train-test split,
    # here we tune hyper-param for each test fold instead of only for the first split !!!
    
    #for i in range(10): # for each CV fold
        #CVfold = 'CV' + str(i) + '/'
        #print('********************* ' + CVfold)
        #train_test_root = train_test_dir + CVfold
        #srcDir = srcRootDir + CVfold + pklFileName
        
    best_params, best_Pearson_r_val = tune_params_func_10foldVer(train_test_dir, srcRootDir)
    print_str = 'Summary --> this best_params = '+str(best_params) + '; this best_Pearson_r_val = {:.4f}'.format(best_Pearson_r_val)
    print(print_str)
    
    # txt file name to save the metrics & hyper-params:
    finalmetrics_txtFile = dstRootDir + 'best_param_metrics_from10folds.txt'
    file1 = open(finalmetrics_txtFile,"w")#write mode
    file1.write(print_str + "\n")
    file1.close()
    
    # save this best_params dict to pkl:
    f_pkl = open(dstRootDir + 'best_params_from10folds.pkl', 'wb')
    pickle.dump(best_params,f_pkl)
    f_pkl.close()
    
    
    




