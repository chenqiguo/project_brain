#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:49:10 2021

@author: guo.1648
"""

# Do data mat production to generate symmetric matrix for each subject:
# datamat A : connArray dim (64984, 379)
# datamat B = A.T x A dim (379, 379) --> symmetric !!!
# This contains all the mat files (the three Nan DSM_Anxi_T target value mats are ALSO included).
# And the symmetric matrices B will be later used to generate graph theory features.


import os
import numpy as np
import pickle
import scipy.io as sio


srcRootDir_mat = '/eecf/cbcsl/data100b/Chenqi/project_brain/'
srcFolders = ['MMPconnMesh/', 'MMPconnMesh_targetNan/']

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_graph/symmetric_connData/v1_product/'




if __name__ == '__main__':
    
    matDict_list = []
    
    for srcFolder in srcFolders:
        srcRootDir = srcRootDir_mat + srcFolder
        for (dirpath, dirnames, filenames) in os.walk(srcRootDir):
            #print(filenames)
            for filename in filenames:
                #print(filename)
                if ".mat" in filename:
                    print("------------------deal with---------------------")
                    print(filename)
                    fullFileName = srcRootDir + filename
                    mat_contents = sio.loadmat(fullFileName)
                    
                    # get mat fields:
                    connL = mat_contents['connL'] # (32492, 379)
                    connR = mat_contents['connR'] # (32492, 379)
                    
                    connArray = np.concatenate((np.array(connL), np.array(connR))) # A: (64984, 379)
                    
                    # (1) get symmetric data mat by matrix production:
                    connArray_sym = np.matmul(connArray.T, connArray)
                    assert(connArray_sym.shape == (379,379))
                    
                    matDict = {
                              'filename': filename,
                              'connArray_sym': connArray_sym, # dim (379,379)
                              #'DSM_Anxi_T': ??? # the original target value --> output
                              }
                    
                    # (2) load the behavior values:
                    behavior_names = mat_contents['behavior_names'][0]
                    assert(behavior_names[0][0] == 'DSM_Anxi_T')
                    
                    for i,element in enumerate(behavior_names):
                        this_behavior_name = str(element[0])
                        this_behavior_val = mat_contents['behavior'][0][i]
                        
                        matDict[this_behavior_name] = this_behavior_val
                                        
                    matDict_list.append(matDict)
    
    # save results to pkl:
    f_pkl = open(dstRootDir+'connArrays_productSymm.pkl', 'wb')
    pickle.dump(matDict_list,f_pkl)
    f_pkl.close()
    
    





