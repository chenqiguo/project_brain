#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:34:14 2021

@author: guo.1648
"""

# generate mat file names of training and testing(validation) for 10-fold CV,
# used for all kinds of brain project CV predictions.

# referenced from generate_imageNames_CV.py


import numpy as np
import cv2
import os
import random


file_allMatNames = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/dataMat_D_matNames.txt'

dstRootDir_CV = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_CV/'

percentage_CV = 0.1 # each CV set is 10% of the whole dataset




if __name__ == '__main__':
    
    with open(file_allMatNames) as f:
        content = f.readlines()
    all_mat_names = [x.strip() for x in content]
    
    # shuffle
    random.shuffle(all_mat_names)
    
    num_CV = round(len(all_mat_names) * percentage_CV) #102
    num_CV_set = int(1/percentage_CV) #10
    
    # get each CV fold mat file names:
    CV_set_dict = {}
    for i in range(num_CV_set):
        #print("-"*20)
        if i != num_CV_set-1:
            CV_set = all_mat_names[i*num_CV:(i+1)*num_CV]
            #print(len(CV_set))
            #print(i*num_CV)
            #print((i+1)*num_CV)
        else:
            CV_set = all_mat_names[i*num_CV:]
            #print(len(CV_set))
            #print(i*num_CV)
        CV_set_dict[i] = CV_set
    
    # write each CV train & test mat file names to txt:
    for i in range(num_CV_set):
        #print("-"*20)
        CV_trainValDir = dstRootDir_CV + 'CV' + str(i) + '/'
        CV_trainDir = CV_trainValDir + 'train.txt'
        CV_validDir = CV_trainValDir + 'valid.txt'
        f_CV_train = open(CV_trainDir, 'w')
        f_CV_valid = open(CV_validDir, 'w')
        
        # for validation set:
        for item in CV_set_dict[i]:
            f_CV_valid.write("%s\n" % item)
        f_CV_valid.close()
        
        # for training set:
        CV_train_list = []
        for j in range(num_CV_set):
            if j != i:
                CV_train_list += CV_set_dict[j]
        for item in CV_train_list:
            f_CV_train.write("%s\n" % item)
        f_CV_train.close()
        
