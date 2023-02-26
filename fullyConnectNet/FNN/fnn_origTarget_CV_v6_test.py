#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:00:53 2021

@author: guo.1648
"""

# for each original target:
# FNN (v3) code for 10-fold Cross Validation.

# This code use the hyper-parameters tuned in fnn_origTarget_CV_v6_tuneParam.py
# to test on 10-fold CV.

# NOTE: since we are doing z-score normalization here, which makes the y_std=1 and y_mean=0,
# thus the tolerance value for computing all accuracy is set to 1/2 = 0.5


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


# the specific graph_feat name to be used as X:
feat_name = 'betweenness_centrality' # 'degree_centrality'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/fullyConnectNet/FNN/result/fnn_origTarget_CV_v6/' + feat_name + '/'

srcRootDir_sparse = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnSparse/'
srcParamPkl = dstRootDir + 'best_params_fromCV0.pkl'

srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_graph/graphFeat/v2_MMPconnSparse/'
pklFileName = 'graphFeat_dict_all.pkl'

train_test_dir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/'

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


def check_targetNan_newVer6(filename):
    # check if any behavior_val is NaN
    
    # load the mat file:
    fullFileName = srcRootDir_sparse + filename # full name of the mat file
    mat_contents = sio.loadmat(fullFileName)
    
    for target_name in all_target_list:
        behavior_val = mat_contents[target_name][0][0]
        if math.isnan(behavior_val):
            return False
    
    return True


def get_all_targets_newVer6(filename):
    # load the mat file:
    fullFileName = srcRootDir_sparse + filename # full name of the mat file
    mat_contents = sio.loadmat(fullFileName)
    
    behavior_val_list = []
    for target_name in all_target_list:
        behavior_val = mat_contents[target_name][0][0]
        assert(not math.isnan(behavior_val))
        behavior_val_list.append(behavior_val)
    return behavior_val_list


def loadDataAlltarget_graphVer(mat_ids, srcDir):
    
    # load the src pkl file:
    f_pkl = open(srcDir,'rb')
    feat_dict_all = pickle.load(f_pkl)
    f_pkl.close()
    
    X = []
    y = []
    
    for filename in mat_ids:
        target_dict = feat_dict_all[filename]
        
        # check if any behavior_val is NaN:
        validFlag = check_targetNan_newVer6(filename)
        if validFlag:
            
            this_graph_feat = target_dict[feat_name]
            if type(this_graph_feat) is dict:
                this_graph_feat = list(this_graph_feat.values())
            X.append(this_graph_feat)
            
            behavior_val_list = get_all_targets_newVer6(filename)
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


def accuracy(y_pred_i_NoNan, y_i_NoNan):
    
    tolerance_value = 0.5
    
    difference = (y_pred_i_NoNan - y_i_NoNan) # difference.shape: (num_samples, 1)
    difference = np.absolute(difference)
        
    acc_value = np.count_nonzero(difference <= tolerance_value) / len(difference)
    
    return acc_value


def evaluate_step(model, X, y, sigma):
    y_pred = model.predict(X, batch_size=batch_size, verbose=0)
    cor = np.zeros((y.shape[-1]))
    mae = np.zeros((y.shape[-1]))
    acc = np.zeros((y.shape[-1]))
    r2 = np.zeros((y.shape[-1]))
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
            acc[i] = np.nan
            r2[i] = np.nan
        else:
            cor[i] = pearsonr(y_pred_i_NoNan, y_i_NoNan)[0]
            mae[i] = np.mean(np.abs(y_pred_i_NoNan - y_i_NoNan)) * sigma[i]
            acc[i] = accuracy(y_pred_i_NoNan, y_i_NoNan)
            r2[i] = r2_score(y_pred_i_NoNan, y_i_NoNan)
    
    return (cor, mae, acc, r2)


def fit_one_cycle(epochs, X_train, y_train, X_test, y_test, sigma, result_dir='', saveFlag=False):
    
    history = []
    
    # initialize model:
    model = hcp_fnn(X_train.shape[1], n_measure, n_l1, n_l2,
                    n_l3, dropout, l2_regularizer)
    optimizer = SGD(lr=lr, momentum=momentum, decay=lr_decay, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    # initialize metrics:
    best_testLoss = np.float('inf') # the lowest MAE loss
    best_testMAEArr = [] # corresponding to best_testLoss
    
    best_testAcc = -1 # the highest accuracy (using tolerance=0.5)
    best_testAccArr = [] # corresponding to best_testAcc
    
    best_testPearson_r = -np.float('inf') # the highest Pearson correlation coefficient
    best_testCorArr = [] # corresponding to best_testPearson_r
    
    best_testR2 = -np.float('inf') # the highest R2
    best_testR2Arr = [] # corresponding to best_testR2
    
    for epoch in range(epochs):
        print('** epoch ' + str(epoch))
        result_thisEpoch = {}
        
        # fit model:
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
        
        # for training metrics:
        cor_train, mae_train, acc_train, r2_train = evaluate_step(model, X_train, y_train, sigma)
        result_thisEpoch['train_loss'] = np.mean(mae_train[~np.isnan(mae_train)])
        result_thisEpoch['train_acc'] = np.mean(acc_train[~np.isnan(acc_train)])
        result_thisEpoch['train_Pearson_r'] = np.mean(cor_train[~np.isnan(cor_train)])
        result_thisEpoch['train_r2'] = np.mean(r2_train[~np.isnan(r2_train)])
        # for testing metrics:
        cor_test, mae_test, acc_test, r2_test = evaluate_step(model, X_test, y_test, sigma)
        result_thisEpoch['test_loss'] = np.mean(mae_test[~np.isnan(mae_test)])
        result_thisEpoch['test_acc'] = np.mean(acc_test[~np.isnan(acc_test)])
        result_thisEpoch['test_Pearson_r'] = np.mean(cor_test[~np.isnan(cor_test)])
        result_thisEpoch['test_r2'] = np.mean(r2_test[~np.isnan(r2_test)])
        
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, train_Pearson_r: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}, test_Pearson_r: {:.4f}".format(
              epoch, result_thisEpoch['train_loss'], result_thisEpoch['train_acc'], result_thisEpoch['train_Pearson_r'],
              result_thisEpoch['test_loss'], result_thisEpoch['test_acc'],result_thisEpoch['test_Pearson_r']))
        history.append(result_thisEpoch)
        
        # save "best" models:
        if result_thisEpoch['test_loss'] < best_testLoss:
            best_testLoss = result_thisEpoch['test_loss']
            best_testMAEArr = mae_test
            if saveFlag:
                model.save(result_dir+'model_bestTestLoss')
        if result_thisEpoch['test_Pearson_r'] > best_testPearson_r:
            best_testPearson_r = result_thisEpoch['test_Pearson_r']
            best_testCorArr = cor_test
            if saveFlag:
                model.save(result_dir+'model_bestTestPearson_r')
        if result_thisEpoch['test_acc'] > best_testAcc:
            best_testAcc = result_thisEpoch['test_acc']
            best_testAccArr = acc_test
            if saveFlag:
                model.save(result_dir+'model_bestTestAcc')
        if result_thisEpoch['test_r2'] > best_testR2:
            best_testR2 = result_thisEpoch['test_r2']
            best_testR2Arr = r2_test
            if saveFlag:
                model.save(result_dir+'model_bestTestR2')
        
        
    best_test_metrics_dict = {'best_testLoss': best_testLoss,
                              'best_testMAEArr': best_testMAEArr,
                              'best_testPearson_r': best_testPearson_r,
                              'best_testCorArr': best_testCorArr,
                              'best_testAcc': best_testAcc,
                              'best_testAccArr': best_testAccArr,
                              'best_testR2': best_testR2,
                              'best_testR2Arr': best_testR2Arr}
    
    return (history, best_test_metrics_dict)





if __name__ == '__main__':
    
    random.seed(10)
    np.random.seed(10)
    
    epochs = 50
    batch_size = 50
    momentum = 0.9
    l2_regularizer = 0.02
    n_measure = len(all_target_list)
    
    # load the tuned hyper-params from pkl:
    f_pkl = open(srcParamPkl,'rb')
    best_params = pickle.load(f_pkl)
    f_pkl.close()
    
    lr = best_params['best_lr']
    lr_decay = best_params['best_lr_decay']
    dropout = best_params['best_dropout']
    n_l1 = best_params['best_n_l1']
    n_l2 = best_params['best_n_l2']
    n_l3 = best_params['best_n_l3']
    threshold = best_params['threshold']
    
    # here we predict all the targets at once!!!
    
    loss_test_list = []
    maeArr_test_dict = {key:[] for key in all_target_list}
    
    Pearson_r_test_list = []
    corArr_test_dict = {key:[] for key in all_target_list}
    
    acc_test_list = []
    accArr_test_dict = {key:[] for key in all_target_list}
    
    r2_test_list = []
    r2Arr_test_dict = {key:[] for key in all_target_list}
    
    srcDir = srcRootDir + 'thresh'+str(threshold)+'/' + pklFileName
    
    for i in range(10): # for each CV fold
        # then this CVi is test fold
        CVfold = 'CV' + str(i) + '/'
        print('********************* ' + CVfold)
        
        K.clear_session()
        
        result_dir_CV = dstRootDir + CVfold
        if not os.path.exists(result_dir_CV):
            os.makedirs(result_dir_CV)
        
        train_test_root = train_test_dir + CVfold
        
        print('@@@testing...')
        # load the mat names of train & test set for this testFold_idx:
        trainallMat_ids_tmp = [line.rstrip() for line in open(os.path.join(train_test_root, 'train_allSubject.txt'))]
        testMat_ids_tmp = [line.rstrip() for line in open(os.path.join(train_test_root, 'test_allSubject.txt'))]
        # newly added: remove those mat files whose target value is Nan:
        trainallMat_ids = removeNan(trainallMat_ids_tmp)
        testMat_ids = removeNan(testMat_ids_tmp)
        
        # newly modified: also load the graph_feat data & targets:
        X_train, y_train_nonNorm = loadDataAlltarget_graphVer(trainallMat_ids, srcDir) # X_train.shape: (895, #graph_feat)
        X_test, y_test_nonNorm = loadDataAlltarget_graphVer(testMat_ids, srcDir)
        
        # z normalize y data based on training set:
        y_train, y_test, y_sigma = zscore_norm_fnn(y_train_nonNorm, y_test_nonNorm)
        
        # just set all tolerance values to be 0.5 !!! --> since we do z-score normalization
        
        history, best_test_metrics_dict = fit_one_cycle(epochs, X_train, y_train, X_test, y_test, y_sigma,
                                                        result_dir=result_dir_CV, saveFlag=True)
        
        # save train-test history to pkl:
        f_pkl = open(result_dir_CV+'history.pkl', 'wb')
        pickle.dump(history,f_pkl)
        f_pkl.close()
        
        # save best_test_metrics_dict to pkl:
        f_pkl = open(result_dir_CV+'best_test_metrics_dict.pkl', 'wb')
        pickle.dump(best_test_metrics_dict,f_pkl)
        f_pkl.close()
        
        # append to lists:
        loss_test_list.append(best_test_metrics_dict['best_testLoss'])
        Pearson_r_test_list.append(best_test_metrics_dict['best_testPearson_r'])
        acc_test_list.append(best_test_metrics_dict['best_testAcc'])
        r2_test_list.append(best_test_metrics_dict['best_testR2'])
        
        for k,key in enumerate(all_target_list):
            maeArr_test_dict[key].append(best_test_metrics_dict['best_testMAEArr'][k])
            corArr_test_dict[key].append(best_test_metrics_dict['best_testCorArr'][k])
            accArr_test_dict[key].append(best_test_metrics_dict['best_testAccArr'][k])
            r2Arr_test_dict[key].append(best_test_metrics_dict['best_testR2Arr'][k])
        
        
    
    loss_test_arr = np.array(loss_test_list)
    Pearson_r_test_arr = np.array(Pearson_r_test_list)
    acc_test_arr = np.array(acc_test_list)
    r2_test_arr = np.array(r2_test_list)
    
    # compute mean across CV folds as the final metrics:
    loss_test = np.mean(loss_test_arr[~np.isnan(loss_test_arr)]) 
    Pearson_r_test = np.mean(Pearson_r_test_arr[~np.isnan(Pearson_r_test_arr)])
    acc_test = np.mean(acc_test_arr[~np.isnan(acc_test_arr)])
    r2_test = np.mean(r2_test_arr[~np.isnan(r2_test_arr)])
    
    for k,key in enumerate(all_target_list):
        maeArr_test_dict[key] = np.array(maeArr_test_dict[key])
        corArr_test_dict[key] = np.array(corArr_test_dict[key])
        accArr_test_dict[key] = np.array(accArr_test_dict[key])
        r2Arr_test_dict[key] = np.array(r2Arr_test_dict[key])
        
        maeArr_test_dict[key] = np.mean(maeArr_test_dict[key][~np.isnan(maeArr_test_dict[key])])
        corArr_test_dict[key] = np.mean(corArr_test_dict[key][~np.isnan(corArr_test_dict[key])])
        accArr_test_dict[key] = np.mean(accArr_test_dict[key][~np.isnan(accArr_test_dict[key])])
        r2Arr_test_dict[key] = np.mean(r2Arr_test_dict[key][~np.isnan(r2Arr_test_dict[key])])
    
    
    metrics_str = ''
    
    metrics_str += ' ^^^^^^^^^^^^^^^^ SUMMARY ^^^^^^^^^^^^^^^^ \n'
    metrics_str += 'best avg loss_test = ' + str(loss_test) + '\n'
    metrics_str += 'best avg Pearson_r_test = ' + str(Pearson_r_test) + '\n'
    metrics_str += 'best avg acc_test = ' + str(acc_test) + '\n'
    metrics_str += 'best avg r2_test = ' + str(r2_test) + '\n'
    
    for target_name in all_target_list: # for each behavior target
        metrics_str += '---------------------- ' + target_name + ' ----------------------\n'
        metrics_str += 'Pearson_r_test_meanAcrossCV = ' + str(corArr_test_dict[target_name]) + '\n'
        metrics_str += 'MAE_test_meanAcrossCV = ' + str(maeArr_test_dict[target_name]) + '\n'
        metrics_str += 'acc_test_meanAcrossCV = ' + str(accArr_test_dict[target_name]) + '\n'
        metrics_str += 'r2_test_meanAcrossCV = ' + str(r2Arr_test_dict[target_name]) + '\n'
    
    
    print('\n\n\n\n')
    print(metrics_str)
    
    # write to txt file:
    metrics_txtFile = dstRootDir + 'test_metrics.txt'
    file1 = open(metrics_txtFile,"w")#write mode
    file1.write(metrics_str + "\n")
    file1.close()
        
    
    
    
    
    
    
    




