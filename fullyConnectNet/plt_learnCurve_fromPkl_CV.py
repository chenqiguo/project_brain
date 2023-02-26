#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:51:48 2021

@author: guo.1648
"""

# plot learning curves using history.pkl for 10-fold CV

# referenced from plt_learnCurve_fromPkl.py


import pickle
import numpy as np
import matplotlib.pyplot as plt

srcRootDir = 'results/DSM_Anxi_T/PCA_v4_CV/fc_v2_bc_allSubjects_thresh50/' #'results/DSM_Anxi_T/PCA_v4_CV/fc_v1_bc_allSubjects_thresh50/'

pklFile = 'history.pkl'


if __name__ == '__main__':
    model_type = 'deep_fc_v2_connData2PCAv4_flatten_CV_bc_thresh50' #'deep_fc_v1_connData2PCAv4_flatten_CV_bc_thresh50'
    
    fig_trainAcc = plt.figure()
    ax_trainAcc = fig_trainAcc.add_subplot(111)
    
    fig_trainLoss = plt.figure()
    ax_trainLoss = fig_trainLoss.add_subplot(111)
    
    fig_validAcc = plt.figure()
    ax_validAcc = fig_validAcc.add_subplot(111)
    
    fig_validLoss = plt.figure()
    ax_validLoss = fig_validLoss.add_subplot(111)
    
    for i in range(10): # for each of 10-fold CV
        cvFold = 'CV' + str(i) + '/'
        srcDir = srcRootDir + cvFold
        
        f_pkl = open(srcDir+pklFile,'rb')
        history = pickle.load(f_pkl)
        f_pkl.close()
        
        epochs = []
        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []
        
        for j, dict_ in enumerate(history):
            epochs.append(j)
            train_loss_list.append(dict_['train_loss'])
            train_acc_list.append(dict_['train_acc'])
            valid_loss_list.append(dict_['val_loss'])
            valid_acc_list.append(dict_['val_acc'])
        
        ax_trainAcc.plot(epochs, train_acc_list, label = "CV"+str(i))
        ax_trainLoss.plot(epochs, train_loss_list, label = "CV"+str(i))
        ax_validAcc.plot(epochs, valid_acc_list, label = "CV"+str(i))
        ax_validLoss.plot(epochs, valid_loss_list, label = "CV"+str(i))
        
    ax_trainAcc.legend()
    ax_trainAcc.set_ylim([0,1])
    title_str = model_type + '_trainAcc'
    ax_trainAcc.set_title(title_str)
    fig_trainAcc.savefig(srcRootDir + title_str + '.png')
    
    ax_trainLoss.legend()
    ax_trainLoss.set_ylim([0,1])
    title_str = model_type + '_trainLoss'
    ax_trainLoss.set_title(title_str)
    fig_trainLoss.savefig(srcRootDir + title_str + '.png')
    
    ax_validAcc.legend()
    ax_validAcc.set_ylim([0,1])
    title_str = model_type + '_validAcc'
    ax_validAcc.set_title(title_str)
    fig_validAcc.savefig(srcRootDir + title_str + '.png')
    
    ax_validLoss.legend()
    ax_validLoss.set_ylim([0,2]) # [0,6]
    title_str = model_type + '_validLoss'
    ax_validLoss.set_title(title_str)
    fig_validLoss.savefig(srcRootDir + title_str + '.png')







