#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:39:54 2021

@author: guo.1648
"""

# From all the 1018 samples,
# (1) randomly select 118 samples as test set;
# (2) randomly split the rest 900 samples into 10-fold Cross Validation (i.e., each fold with 90 samples).
# used for brain project CV predictions, to slect and test the early-stopping criteria.

# referenced from generate_matNames_CV.py


import numpy as np
import cv2
import os
import random


file_allMatNames = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/dataMat_D_matNames.txt'

dstRootDir_CV = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/mat_names_testCV/'

testSet_num = 118
percentage_CV = 0.1 # each CV set is 10% of the rest of 900 dataset


if __name__ == '__main__':
    
    with open(file_allMatNames) as f:
        content = f.readlines()
    all_mat_names = [x.strip() for x in content]
    
    # shuffle
    random.shuffle(all_mat_names)
    
    # (1) select 118 samples as test set:
    test_set_mat_names = all_mat_names[:testSet_num]
    trainVal_set_mat_names = all_mat_names[testSet_num:]
    
    f_CV_test = open(dstRootDir_CV+'test_set/test_allSubject.txt', 'w') # test set
    for item in test_set_mat_names:
        f_CV_test.write("%s\n" % item)
    f_CV_test.close()
    
    f_CV_wholeTrain = open(dstRootDir_CV+'test_set/train_allSubject.txt', 'w') # whole train set (900 samples)
    for item in trainVal_set_mat_names:
        f_CV_wholeTrain.write("%s\n" % item)
    f_CV_wholeTrain.close()
    
    # (2) randomly split the rest 900 samples into 10-fold Cross Validation:
    num_CV = round(len(trainVal_set_mat_names) * percentage_CV) #90
    num_CV_set = int(1/percentage_CV) #10
    
    # get each CV fold mat file names:
    CV_set_dict = {}
    for i in range(num_CV_set):
        #print("-"*20)
        if i != num_CV_set-1:
            CV_set = trainVal_set_mat_names[i*num_CV:(i+1)*num_CV]
            #print(len(CV_set))
            #print(i*num_CV)
            #print((i+1)*num_CV)
        else:
            CV_set = trainVal_set_mat_names[i*num_CV:]
            #print(len(CV_set))
            #print(i*num_CV)
        CV_set_dict[i] = CV_set
    
    # write each CV train & test mat file names to txt:
    for i in range(num_CV_set):
        #print("-"*20)
        CV_trainValDir = dstRootDir_CV + 'CV' + str(i) + '/'
        CV_trainDir = CV_trainValDir + 'train_allSubject.txt' # train set
        CV_validDir = CV_trainValDir + 'test_allSubject.txt' # valid set
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


