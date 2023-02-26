#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:07:12 2021

@author: guo.1648
"""

# Stack each column of all mat files' connData to generate
# dataMat_D_regionxxx and saved to pkls for each region.
# This is for further usage (like in Cross Validation).

# Note: the three Nan DSM_Anxi_T target value mats are ALSO included
# in this newly generated pkl files!!!
# Also, for reference, the corresponding mat file names are also recorded in txt file,
# in the SAME order of the dataMat_D_regionxxx (i.e., every row of dataMat_D_regionxxx
# corresponds to every row of txt).

# referenced from connData_dimReduc_PCA_v4.py


import os
import numpy as np
import pickle
import scipy.io as sio


#srcRootDir_pkl = 'results/PCA_v4/tmp/setp1/'
srcRootDir_mat = '/eecf/cbcsl/data100b/Chenqi/project_brain/'
srcFolders = ['MMPconnMesh/', 'MMPconnMesh_targetNan/']
#train_test_root = '/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_dimReduc/useful_data/'



def get_all_matNames():
    all_mat_names = []
    
    for srcFolder in srcFolders:
        srcDir_mat = srcRootDir_mat + srcFolder
        
        for (dirpath, dirnames, filenames) in os.walk(srcDir_mat):
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
        
        if filename in ['109830.mat', '614439.mat', '677968.mat']:
            fullFileName = srcRootDir_mat + 'MMPconnMesh_targetNan/' + filename
        else:
            fullFileName = srcRootDir_mat + 'MMPconnMesh/' + filename
        assert(os.path.exists(fullFileName))
        mat_contents = sio.loadmat(fullFileName)
        
        # (1) for connData:
        connL = mat_contents['connL'] # (32492, 379)
        connR = mat_contents['connR'] # (32492, 379)
        connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
        
        # only get the i-th column:
        connArray_coli = connArray[:,i]
        dataMat_D_list.append(connArray_coli)
            
    dataMat_D = np.array(dataMat_D_list) # shape (1018, 64984)
    
    return dataMat_D


if __name__ == '__main__':
    
    all_mat_names = get_all_matNames()
    # saved in txt:
    with open(dstRootDir+'dataMat_D_matNames.txt', 'w') as f:
        for item in all_mat_names:
            f.write("%s\n" % item)
    
    for i in range(379): # for each region
        dataMat_D = get_dataMat_D(all_mat_names, i)
        #print() # check shape
        
        # save dataMat_D to pkl file:
        f_pkl = open(dstRootDir+'dataMat_D/dataMat_D_all_region'+str(i)+'.pkl', 'wb')
        pickle.dump(dataMat_D,f_pkl)
        f_pkl.close()
        
        #print()
                    
        
    
    


