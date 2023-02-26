#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:28:21 2021

@author: guo.1648
"""

# NOT USE !!!

# referenced from /eecf/cbcsl/data100b/Chenqi/project_brain/paper_demo/FNN/fnn_origTarget_CV_v3_tuneParam.py:
# for all the original targets at once: do z-score normalization on y, and use
# fully-connected neural network (FNN) (v3) code for 10-fold Cross Validation.
# This code tune the hyper-parameters and save the corresponding hyper-params & metrics.

# Flatten 2PCA connData (v4) as input; predict all original targets at once;
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

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/results/PCA_v4_CV/'
pklFileName = 'connData_dimReduc_PCA_dict_v4_CV.pkl'

train_test_dir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/fullyConnectNet/FNN/result/fnn_origTarget_CV_v3/'

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


def check_targetNan_newVer(target_dict):
    # check if any behavior_val is NaN
    
    for target_name in all_target_list:
        behavior_val = target_dict[target_name]
        if math.isnan(behavior_val):
            return False
    
    return True


def get_all_targets_newVer(target_dict):
    behavior_val_list = []
    for target_name in all_target_list:
        behavior_val = target_dict[target_name]
        assert(not math.isnan(behavior_val))
        behavior_val_list.append(behavior_val)
    return behavior_val_list


def loadDataAlltarget_PCAver(mat_ids, srcDir):
    
    # load the src pkl file:
    f_pkl = open(srcDir,'rb')
    dataMat_subject_all_dict = pickle.load(f_pkl)
    f_pkl.close()
    
    X = []
    y = []
    
    for filename in mat_ids:
        target_dict = dataMat_subject_all_dict[filename]
        
        # check if any behavior_val is NaN:
        validFlag = check_targetNan_newVer(target_dict)
        if validFlag:
            
            dataMat_feat = target_dict['dataMat_feat']
            dataMat_feat_flat = dataMat_feat.flatten() # dim: (1024,)
            X.append(dataMat_feat_flat)
            
            behavior_val_list = get_all_targets_newVer(target_dict)
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


def tune_params_func_newVer(train_test_root, srcDir):
    # the inner loop to find the best hyper-params by performing on the remaining 9 folds.
    # the test fold is CV0.
    # referenced from function tune_params_func(train_test_root).
    print('@@@tuning hyper-params...')
    
    best_params = {'best_lr': -100, # just dummy values!
                   'best_lr_decay': -100,
                   'best_dropout': -100,
                   'best_n_l1': -100,
                   'best_n_l2': -100,
                   'best_n_l3': -100}
    best_Pearson_r_val = -100 # store the highest one!
    
    # load the mat names of train & val set for this CV0:
    trainMat_ids_tmp = [line.rstrip() for line in open(os.path.join(train_test_root, 'train_allSubject.txt'))]
    # newly added: remove those mat files whose target value is Nan:
    trainMat_ids = removeNan(trainMat_ids_tmp)
    # newly modified: --> NOTE: should also modified in krr !!!! --> will modify & re-run later.
    num_CV = round(len(trainMat_ids) /9) # should be 101 instead of 91 !!! otherwise the last (9-th) fold cannot be validated!!!
    
    # txt file name to save the metrics & hyper-params:
    metrics_txtFile = dstRootDir + 'param_metrics_fromCV0.txt'
    
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
                            
                            for j in range(9): # only for each of the remaining 9 folds (except this CV0)
                                #torch.cuda.empty_cache()
                                #print(Pearson_r_val_list) # just for debug!
                                print('----- j = ' + str(j))
                                
                                K.clear_session()
                                
                                mat_ids_valid = trainMat_ids[j*num_CV:(j+1)*num_CV]
                                mat_ids_train = trainMat_ids[:j*num_CV] + trainMat_ids[(j+1)*num_CV:]
                                
                                # newly modified: also load the vectorized data & targets:
                                X_train, y_train_nonNorm = loadDataAlltarget_PCAver(mat_ids_train, srcDir) # X_train.shape: (796, 1024)
                                X_valid, y_valid_nonNorm = loadDataAlltarget_PCAver(mat_ids_valid, srcDir)
                                
                                # z normalize y data based on training set:
                                y_train, y_valid, y_sigma = zscore_norm_fnn(y_train_nonNorm, y_valid_nonNorm)
                                
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
                                              verbose=0, validation_data=(X_valid, y_valid))
                                    
                                    cor, _ = fit_one_cycle_tune_1epoch(model, X_valid, y_valid, y_sigma)
                                    
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
    n_l1_list = [32, 16] # 64, 32
    n_l2_list = [16, 8] # 32, 16
    n_l3_list = [8, 4] # 16, 8
    
    # here we predict all the targets at once!!!
    
    # From the paper: What we do here:
    # a proper inner-loop 10-fold cross-validation (like the one in krr) would
    # involve tuning the hyperparameters for each DNN 10 times (once for
    # each split of the data into training-test folds), which was computationally
    # prohibitive. Thus, for each DNN (FNN, BrainNetCNN and GCNN), we
    # tuned the hyperparameters once, using the first split of the data into
    # training-test folds, and simply re-used the optimal hyperparameters for
    # the remaining training-test splits of the data. Such a procedure biases the
    # prediction performance in favor of the DNNs (relative to kernel regression),
    # so the results should be interpreted accordingly.
    i = 0 # tune hyper-param only for the first split
    CVfold = 'CV' + str(i) + '/'
    print('********************* ' + CVfold)
    train_test_root = train_test_dir + CVfold
    srcDir = srcRootDir + CVfold + pklFileName
    
    best_params, best_Pearson_r_val = tune_params_func_newVer(train_test_root, srcDir)
    print_str = 'Summary --> this best_params = '+str(best_params) + '; this best_Pearson_r_val = {:.4f}'.format(best_Pearson_r_val)
    print(print_str)
    
    # txt file name to save the metrics & hyper-params:
    finalmetrics_txtFile = dstRootDir + 'best_param_metrics_fromCV0.txt'
    file1 = open(finalmetrics_txtFile,"w")#write mode
    file1.write(print_str + "\n")
    file1.close()
    
    # save this best_params dict to pkl:
    f_pkl = open(dstRootDir + 'best_params_fromCV0.pkl', 'wb')
    pickle.dump(best_params,f_pkl)
    f_pkl.close()
    



