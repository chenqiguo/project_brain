#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:41:32 2021

@author: guo.1648
"""

# plot learning curves using history.pkl

import pickle
import numpy as np
import matplotlib.pyplot as plt

rootDir = 'results/DSM_Anxi_T/PCA_v4/maxlr_0_05/fc_v1_bc_allSubjects_thresh50/' #'results/DSM_Anxi_T/PCA_v4/fc_v1_bc_allSubjects_thresh50/'

pklFile = 'history.pkl'


if __name__ == '__main__':
    f_pkl = open(rootDir+pklFile,'rb')
    history = pickle.load(f_pkl)
    f_pkl.close()
    
    epochs = []
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    
    for i, dict_ in enumerate(history):
        epochs.append(i)
        train_loss_list.append(dict_['train_loss'])
        train_acc_list.append(dict_['train_acc'])
        valid_loss_list.append(dict_['val_loss'])
        valid_acc_list.append(dict_['val_acc'])
       
    
    model_type = 'deep_fc_v1_connData2PCAv4_flatten_bc_thresh50'
        
    # plot curves of train_loss & valid_loss:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_loss_list)
    ax.plot(epochs, valid_loss_list)
    ax.legend(['train_loss', 'valid_loss'])
    ax.set_ylim([0,max(max(train_loss_list), max(valid_loss_list))+1])
    title_str = model_type + '_loss'
    plt.title(title_str)
    fig.savefig(rootDir + title_str + '.png')
    # plot curves of train_acc & valid_acc:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_acc_list)
    ax.plot(epochs, valid_acc_list)
    ax.legend(['train_acc', 'valid_acc'])
    ax.set_ylim([0,max(max(train_acc_list), max(valid_acc_list))+0.1])
    title_str = model_type + '_acc'
    plt.title(title_str)
    fig.savefig(rootDir + title_str + '.png')

    


