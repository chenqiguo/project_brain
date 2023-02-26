#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:12:01 2021

@author: guo.1648
"""

# plot learning curves for custom model training procedures.

from csv import reader
import numpy as np
import matplotlib.pyplot as plt


rootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/col_row_size_kernel_resnet/results_customModel_origTarget/' # results_customModel 

csvFile = 'rowcolKernel/maxpool/v4/rowcolKernel_10_50_statistics.csv' #'rowKernel/v4/rowKernel_10_50_statistics.csv' #'rowKernel/v2/rowKernel_1_50_statistics.csv' #'colKernel/v2/colKernel_1_50_statistics.csv'

if __name__ == '__main__':
    
    epochs = []
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    
    with open(rootDir+csvFile, 'r') as read_obj:
        csv_reader = reader(read_obj)
        isHeader = True
        for row in csv_reader:
            #print(row)
            if isHeader:
                isHeader = False
                continue
            epoch,train_loss,train_acc,valid_loss,valid_acc = row
            
            epochs.append(int(epoch))
            train_loss_list.append(float(train_loss))
            train_acc_list.append(float(train_acc))
            valid_loss_list.append(float(valid_loss))
            valid_acc_list.append(float(valid_acc))
            
    #print()
    
    model_type = 'rowcolKernel_maxpool_origTarget' #'rowcolKernel_logTarget' #'rowKernel_logTarget' #'colKernel_origTarget'
    
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



