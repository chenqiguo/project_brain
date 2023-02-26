#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:02:48 2020

@author: guo.1648
"""


# Compute the eigenvalues corresponding to the Principal Components in ab2d feature space,
# for both left breast and right breast, in each case.



import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pickle



imgSrcRootDir = '/eecf/cbcsl/data100//Chenqi/BreastCosmesis/cosmesis_scoring/br_mask_out_marker/'
pltDstRootDir = '/eecf/cbcsl/data100/Chenqi/BreastCosmesis/cosmesis_scoring/features/hist2d_ab2d/'

treatedBrDir = '/eecf/cbcsl/data100/Chenqi/BreastCosmesis/cosmesis_scoring/features/groundTruthScores.pkl'
featureDstDir = '/eecf/cbcsl/data100/Chenqi/BreastCosmesis/cosmesis_scoring/features/feature4_PCA_ab2d.pkl'

trials = ['1014/', '0413/']

feature4_PCA_ab2d = []


# featureDict = {
#        'eigenValues': eigenValues,
#        'eigenVectors': eigenVectors,
#        'treated_breast': 'L' # (or 'R' or '')
#        }


def getNonZeroLoc(img2d):
    # get all the nonzero pixel (y,x) coordinates
    non_zero_idx = np.nonzero(img2d)
    arr1 = non_zero_idx[0]
    arr1 = arr1.reshape(-1,1)
    arr2 = non_zero_idx[1]
    arr2 = arr2.reshape(-1,1)
    nonZeroLoc = np.hstack((arr1,arr2)) # (y,x)
    return nonZeroLoc



def pltHistFunc(imgIdx, pltDstDir, nonZero_L_lb,nonZero_A_lb,nonZero_B_lb,nonZero_L_rb,nonZero_A_rb,nonZero_B_rb):
    # 1) for lb:
    plt.hist2d(nonZero_A_lb,nonZero_B_lb,density=True)
    plt.colorbar()
    plt.title('img ' + imgIdx + ' lb: hist2d for ab2d')
    plt.xlabel('A channel pixVal')
    plt.ylabel('B channel pixVal')
    plt.savefig(pltDstDir + imgIdx + '_lb_ab2d.jpg')
    plt.close()
    # 2) for rb:
    plt.hist2d(nonZero_A_rb,nonZero_B_rb,density=True)
    plt.colorbar()
    plt.title('img ' + imgIdx + ' rb: hist2d for ab2d')
    plt.xlabel('A channel pixVal')
    plt.ylabel('B channel pixVal')
    plt.savefig(pltDstDir + imgIdx + '_rb_ab2d.jpg')
    plt.close()
    
    # plot hist for l
    # 1) for lb:
    plt.hist(nonZero_L_lb,density=True)
    plt.title('img ' + imgIdx + ' lb: hist for L channel')
    plt.xlabel('L channel pixVal')
    plt.savefig(pltDstDir + imgIdx + '_lb_L.jpg')
    plt.close()
    # 2) for rb:
    plt.hist(nonZero_L_rb,density=True)
    plt.title('img ' + imgIdx + ' rb: hist for L channel')
    plt.xlabel('L channel pixVal')
    plt.savefig(pltDstDir + imgIdx + '_rb_L.jpg')
    plt.close()



def generateDataMat(nonZero_A,nonZero_B):
    # generate data matrix dataMat: N*2 dim, N is the number of non-zero pixals,
    # and the 1st col is A channel pixVal, the 2nd col is B channel pixVal
    min_samp_len = min(len(nonZero_A),len(nonZero_B))
    dataMat = np.hstack((nonZero_A[:min_samp_len],nonZero_B[:min_samp_len]))
    
    return dataMat


def myPCA(dataMat):
    # dataMat: N*2 dim
    
    # calculate the mean of each column
    M = np.mean(dataMat.T,axis=1)
    # center columns by subtracting column means
    C = dataMat - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    eigenValues, eigenVectors = np.linalg.eig(V) # each column of eigenVectors is an eigenvector
    
    # order the eigenVectors based on eigenValues
    sort_order = np.argsort(-eigenValues)
    vec1 = eigenVectors[:,sort_order[0]].reshape(-1,1)
    vec2 = eigenVectors[:,sort_order[1]].reshape(-1,1)
    eigenVectors = np.hstack((vec1, vec2))
    # order the eigenValues in descending orders
    eigenValues = -np.sort(-eigenValues)
    
    
    # note: the normalized eigenVectors s.t. the column v[:,i] is the eigenvec
    # corresponding to the eigenval w[i]
    return (eigenValues, eigenVectors)


def getTreatedBr(trial, imgName):
    for _dict in scores:
        if trial == _dict['trial'] and imgName == _dict['imgName']:
            return _dict['treated_breast']
    
    return ''



def dealWithData(imgSrcDir,pltDstDir,trial):
    # the main function generating the data
    for (dirpath, dirnames, filenames) in os.walk(imgSrcDir):
        #print(filenames)
        for filename in filenames:
            #print(filename)
            if "_B_lb_3.npy" in filename: # deal with a set of six npy file at a time
                print("------------------deal with---------------------")
                print(filename)
                B_lb_name = imgSrcDir + filename
                B_rb_name = imgSrcDir + filename.split('_lb_3.npy')[0] + '_rb_3.npy'
                A_lb_name = imgSrcDir + filename.split('B_lb_3.npy')[0] + 'A_lb_3.npy'
                A_rb_name = imgSrcDir + filename.split('B_lb_3.npy')[0] + 'A_rb_3.npy'
                L_lb_name = imgSrcDir + filename.split('B_lb_3.npy')[0] + 'L_lb_3.npy'
                L_rb_name = imgSrcDir + filename.split('B_lb_3.npy')[0] + 'L_rb_3.npy'
                
                # load the set of masked image (without ar & np)
                img_B_lb = np.load(B_lb_name)
                img_B_rb = np.load(B_rb_name)
                img_A_lb = np.load(A_lb_name)
                img_A_rb = np.load(A_rb_name)
                img_L_lb = np.load(L_lb_name)
                img_L_rb = np.load(L_rb_name)
                
                # get non-zero elements
                # get all the nonzero pixel (y,x) coordinates
                # 1) for lb
                nonZeroLoc_A_lb = getNonZeroLoc(img_A_lb) # (y,x)
                nonZeroLoc_B_lb = getNonZeroLoc(img_B_lb) # (y,x)
                nonZeroLoc_L_lb = getNonZeroLoc(img_L_lb) # (y,x)
                # 2) for rb
                nonZeroLoc_A_rb = getNonZeroLoc(img_A_rb) # (y,x)
                nonZeroLoc_B_rb = getNonZeroLoc(img_B_rb) # (y,x)
                nonZeroLoc_L_rb = getNonZeroLoc(img_L_rb) # (y,x)
                # get all the nonzero pixel elements
                # 1) for lb
                nonZero_L_lb = img_L_lb[nonZeroLoc_L_lb[:,0],nonZeroLoc_L_lb[:,1]]
                nonZero_A_lb = img_A_lb[nonZeroLoc_A_lb[:,0],nonZeroLoc_A_lb[:,1]]
                nonZero_B_lb = img_B_lb[nonZeroLoc_B_lb[:,0],nonZeroLoc_B_lb[:,1]]
                # 2) for rb
                nonZero_L_rb = img_L_rb[nonZeroLoc_L_rb[:,0],nonZeroLoc_L_rb[:,1]]
                nonZero_A_rb = img_A_rb[nonZeroLoc_A_rb[:,0],nonZeroLoc_A_rb[:,1]]
                nonZero_B_rb = img_B_rb[nonZeroLoc_B_rb[:,0],nonZeroLoc_B_rb[:,1]]
                
                # plot hist2d for ab2d
                imgIdx = filename.split('_B_lb_3.npy')[0]
                pltHistFunc(imgIdx, pltDstDir, nonZero_L_lb,nonZero_A_lb,nonZero_B_lb,nonZero_L_rb,nonZero_A_rb,nonZero_B_rb)
                
                # compute eigenvalues corresponding to the Principal Components in ab2d feature space,
                # for both left breast and right breast:
                nonZero_A_lb = nonZero_A_lb.reshape(-1,1)
                nonZero_B_lb = nonZero_B_lb.reshape(-1,1)
                nonZero_A_rb = nonZero_A_rb.reshape(-1,1)
                nonZero_B_rb = nonZero_B_rb.reshape(-1,1)
                # 1) First, generate data matrix dataMat: N*2 dim, N is the number of non-zero pixals,
                # and the 1st col is A channel pixVal, the 2nd col is B channel pixVal
                dataMat_lb = generateDataMat(nonZero_A_lb,nonZero_B_lb)
                dataMat_rb = generateDataMat(nonZero_A_rb,nonZero_B_rb)
                # 2) do the PCA on dataMat
                eigenValues_lb, eigenVectors_lb = myPCA(dataMat_lb)
                eigenValues_rb, eigenVectors_rb = myPCA(dataMat_rb)
                
                
                treated_breast = getTreatedBr(trial,imgIdx+'.jpg')
                
                featureDict = {
                        'eigenValues_lb': eigenValues_lb,
                        'eigenVectors_lb': eigenVectors_lb,
                        'eigenValues_rb': eigenValues_rb,
                        'eigenVectors_rb': eigenVectors_rb
                        }
                
                _dict = {
                        'trial': trial,
                        'imgName': filename.split('_B_lb_3.npy')[0]+'.jpg',
                        'lbExists': True,
                        'rbExists': True,
                        'bothBrExist': True,
                        'featureVal': featureDict,
                        'treated_breast': treated_breast
                        }
                
                feature4_PCA_ab2d.append(_dict)
                
                #print()
                
                
                
                
    
    
  
if __name__ == '__main__':
    global scores
    
    # load info of treated_breast
    f = open(treatedBrDir,'rb')
    scores = pickle.load(f)
    f.close()
    
    
    for trial in trials:
        if trial == '1014/':
            imgSrcDir = imgSrcRootDir + trial
            pltDstDir = pltDstRootDir + trial
            dealWithData(imgSrcDir,pltDstDir,trial)
        elif trial == '0413/':
            for i in range(1,21):#21
                groupIdx = 'group' + str(i) + '/'
                imgSrcDir = imgSrcRootDir + trial + groupIdx
                pltDstDir = pltDstRootDir + trial + groupIdx
                dealWithData(imgSrcDir,pltDstDir,trial)
    
    # save the dicts into pickle files
    f_pkl = open(featureDstDir, 'wb')
    pickle.dump(feature4_PCA_ab2d,f_pkl)
    f_pkl.close()





