#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:07:22 2021

@author: guo.1648
"""

# for the symmetrical MMPconnSparse dataset.

# referenced from generate_usefulData_dataMatD.py


import os
import numpy as np
import pickle
import scipy.io as sio


srcRootDir_mat = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnSparse/'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/'


def get_all_matNames():
    all_mat_names = []
    
    for (dirpath, dirnames, filenames) in os.walk(srcRootDir_mat):
        #print(filenames)
        for filename in filenames:
            #print(filename)
            if ".mat" in filename:
                #print("------------------deal with---------------------")
                #print(filename)
                all_mat_names.append(filename)
    
    return all_mat_names


def get_dataMat_D(mat_list, i):
    dataMat_D_list = []
    
    for filename in mat_list:
        
        print("------------------deal with---------------------")
        print(filename)
        
        fullFileName = srcRootDir_mat + filename
        assert(os.path.exists(fullFileName))
        
        mat_contents = sio.loadmat(fullFileName)
        
        fc = mat_contents['fc'] # (379, 379)
        
        # only get the i-th column:
        fc_coli = fc[:,i]
        dataMat_D_list.append(fc_coli)
        
    dataMat_D = np.array(dataMat_D_list) # shape (1018, 379)
    
    return dataMat_D



if __name__ == '__main__':
    
    all_mat_names = get_all_matNames()
    # saved in txt:
    with open(dstRootDir+'dataMat_D_symmMatNames.txt', 'w') as f:
        for item in all_mat_names:
            f.write("%s\n" % item)
    
    for i in range(379): # for each region
        dataMat_D = get_dataMat_D(all_mat_names, i)
        #print() # check shape
        
        # save dataMat_D to pkl file:
        f_pkl = open(dstRootDir+'dataMat_D_symm/dataMat_D_all_region'+str(i)+'.pkl', 'wb')
        pickle.dump(dataMat_D,f_pkl)
        f_pkl.close()



