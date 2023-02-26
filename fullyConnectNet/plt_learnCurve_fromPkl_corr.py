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

rootDir = 'results/fc_v1_origTarget_greater50Subjects_lossPearsonCorr_v1/'

pklFile = 'history.pkl'


if __name__ == '__main__':
    f_pkl = open(rootDir+pklFile,'rb')
    history = pickle.load(f_pkl)
    f_pkl.close()
    
    epochs = []
    train_loss_list = []
    train_corr_list = []
    valid_loss_list = []
    valid_corr_list = []
    
    for i, dict_ in enumerate(history):
        epochs.append(i)
        train_loss_list.append(dict_['train_loss'])
        train_corr_list.append(dict_['train_corr'])
        valid_loss_list.append(dict_['val_loss'])
        valid_corr_list.append(dict_['val_corr'])
       
    
    model_type = 'deep_fc_v1_connData2PCA_flatten_regression_corrMetric'
        
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
    # plot curves of train_corr & valid_corr:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_corr_list)
    ax.plot(epochs, valid_corr_list)
    ax.legend(['train_corr', 'valid_corr'])
    ax.set_ylim([-1,max(max(train_corr_list), max(valid_corr_list))+0.1])
    title_str = model_type + '_corr'
    plt.title(title_str)
    fig.savefig(rootDir + title_str + '.png')

    


