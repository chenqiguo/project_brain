#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:50:27 2020

@author: guo.1648
"""

# referenced from feature124_lasso.py


# for all the images with known treated breast label,
# construct the skinColor feature of the form:
# [eigenVals_treat,eigenVecs_treat.flatten, eigenVals_untreat,eigenVecs_untreat.flatten]
# and then regress to each score (especially skinColor score)


import numpy as np
import pickle

import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

import random
import statistics
from scipy import stats


srcDir = '/eecf/cbcsl/data100/Chenqi/BreastCosmesis/cosmesis_scoring/features/scoresFeatures_v2.pkl'

from generateDataTarget import generateDataTarget
from generateDataTarget_v2 import generateDataTarget_v2
from generateDataTarget_v3 import generateDataTarget_v3
from generateDataTarget_v4 import generateDataTarget_v4
from generateDataTarget_v5 import generateDataTarget_v5
from generateDataTarget_v6 import generateDataTarget_v6
from generateDataTarget_v7 import generateDataTarget_v7
from generateDataTarget_v8 import generateDataTarget_v8
from generateDataTarget_v9 import generateDataTarget_v9
from generateDataTarget_v10 import generateDataTarget_v10
from generateDataTarget_v11 import generateDataTarget_v11
from generateDataTarget_v12 import generateDataTarget_v12
from generateDataTarget_v13 import generateDataTarget_v13
from generateDataTarget_v13_cntCenterVertex import generateDataTarget_v13_cntCenterVertex
from generateDataTarget_v13_cntCenterVertex_v2 import generateDataTarget_v13_cntCenterVertex_v2
from generateDataTarget_v13_cntCenterVertex_v3 import generateDataTarget_v13_cntCenterVertex_v3
from generateDataTarget_v13_arNpVertex import generateDataTarget_v13_arNpVertex
from generateDataTarget_v14 import generateDataTarget_v14
from generateDataTarget_v15 import generateDataTarget_v15
from generateDataTarget_v16 import generateDataTarget_v16
from generateDataTarget_v17 import generateDataTarget_v17
from generateDataTarget_v18 import generateDataTarget_v18
from generateDataTarget_v19 import generateDataTarget_v19
from generateDataTarget_v20 import generateDataTarget_v20
from generateDataTarget_v21 import generateDataTarget_v21
from generateDataTarget_v22 import generateDataTarget_v22
from generateDataTarget_v23 import generateDataTarget_v23
from generateDataTarget_v24 import generateDataTarget_v24
from generateDataTarget_v25 import generateDataTarget_v25
from generateDataTarget_v26 import generateDataTarget_v26
from generateDataTarget_v27 import generateDataTarget_v27
from generateDataTarget_v28 import generateDataTarget_v28
from generateDataTarget_v29 import generateDataTarget_v29
from generateDataTarget_v30 import generateDataTarget_v30
from generateDataTarget_v31 import generateDataTarget_v31
from generateDataTarget_v32 import generateDataTarget_v32


from generateDataTarget_v35 import generateDataTarget_v35
from generateDataTarget_v36 import generateDataTarget_v36
from generateDataTarget_v37 import generateDataTarget_v37

from generateDataTarget_v41 import generateDataTarget_v41


def fill_in_valid(EMD_final_list):
    none_ele_idx_list = []
    valid_ele_list = []
    for k in range(len(EMD_final_list)):
        if EMD_final_list[k] == None:
            none_ele_idx_list.append(k)
        else:
            valid_ele_list.append(EMD_final_list[k])
    """
    # 1st try: use median
    fill_in = statistics.median(valid_ele_list)
    for none_idx in none_ele_idx_list:
        EMD_final_list[none_idx] = fill_in
    """
    # 2nd try: use mean
    fill_in = statistics.mean(valid_ele_list)
    for none_idx in none_ele_idx_list:
        EMD_final_list[none_idx] = fill_in
    
    return EMD_final_list




def zScoreNormalization(X):
    X_normed = np.empty(X.shape)
    # z-score standardization on data X:
    # just subtract the mean and divided by s.t.d for each feature column
    feature_num = X.shape[1]
    for feature_idx in range(feature_num):
        this_feature = X[:,feature_idx]
        this_mean = np.mean(this_feature)
        this_std = np.std(this_feature)
        if this_std < 1e-8:
            this_std = 1e-8
        normed_feature = (this_feature-this_mean)/this_std
        X_normed[:,feature_idx] = normed_feature
    
    return X_normed
    

def zScoreNormalization_exceptBOVW(X, k_numWords):
    # only used when features include bovw features
    X_normed = np.empty(X.shape)
    feature_num = X.shape[1]-k_numWords*2
    for feature_idx in range(feature_num):
        this_feature = X[:,feature_idx]
        this_mean = np.mean(this_feature)
        this_std = np.std(this_feature)
        normed_feature = (this_feature-this_mean)/this_std
        X_normed[:,feature_idx] = normed_feature
    
    #print()
    
    X_normed[:,feature_num:] = X[:,feature_num:]
    
    #print()
    
    return X_normed
        


def lassoMultiRegress_trainTest(X, Y, penalty):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=31) #random_state=31; 21; 1
    
    lasso = Lasso(alpha=penalty,max_iter=10e5) #alpha=0.0001 # NOT use: normalize=True
    lasso.fit(X_train,Y_train)
    coeff_used = np.sum(lasso.coef_!=0)
    print("number of features used: " + str(coeff_used))
    print("lasso coefficients are:")
    print(lasso.coef_)
    
    # for train
    print("---train---")
    # baseline: predict the majority
    counts = np.bincount(Y_train)
    majority = np.argmax(counts)
    baseline_acc = sum(Y_train==majority)/len(Y_train)
    print("baseline accuracy = " + str(baseline_acc))
    
    Y_train_pred = lasso.predict(X_train)
    Y_train_pred = np.rint(Y_train_pred)
    acc_train = sum(Y_train_pred==Y_train)/len(Y_train_pred)
    
    train_score=lasso.score(X_train, Y_train)
    
    print("training accuracy = " + str(acc_train))
    print("training score: " + str(train_score))
    
    # for test
    print("---test---")
    # baseline: predict the majority
    counts = np.bincount(Y_test)
    majority = np.argmax(counts)
    baseline_acc = sum(Y_test==majority)/len(Y_test)
    print("baseline accuracy = " + str(baseline_acc))
    
    Y_test_pred = lasso.predict(X_test)
    Y_test_pred = np.rint(Y_test_pred)
    acc_test = sum(Y_test_pred==Y_test)/len(Y_test_pred)
    
    test_score=lasso.score(X_test, Y_test)
    
    print("testing accuracy = " + str(acc_test))
    print("testing score: " + str(test_score))
    
    return test_score



def lassoMultiRegress_trainTest_CV(X, Y, penalty):
    lasso = Lasso(alpha=penalty,max_iter=10e5)
    # evaluate the test score (same as lasso.score) using 10-fold CV
    cross_val_score_list = cross_val_score(lasso, X, Y, cv=10)
    
    
    
    return cross_val_score_list




def lassoMultiRegress_searchPenalty(X, Y, penalty):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=31)
    
    lasso = Lasso(alpha=penalty,max_iter=10e5) #alpha=0.0001 # NOT use: normalize=True
    lasso.fit(X_train,Y_train)
    #coeff_used = np.sum(lasso.coef_!=0)
    #print("number of features used: " + str(coeff_used))
    #print("lasso coefficients are:")
    #print(lasso.coef_)
    
    # for train
    #print("---train---")
    # baseline: predict the majority
    #counts = np.bincount(Y_train)
    #majority = np.argmax(counts)
    #baseline_acc = sum(Y_train==majority)/len(Y_train)
    #print("baseline accuracy = " + str(baseline_acc))
    
    #Y_train_pred = lasso.predict(X_train)
    #Y_train_pred = np.rint(Y_train_pred)
    #acc_train = sum(Y_train_pred==Y_train)/len(Y_train_pred)
    
    #train_score=lasso.score(X_train, Y_train)
    
    #print("training accuracy = " + str(acc_train))
    #print("training score: " + str(train_score))
    
    # for test
    #print("---test---")
    # baseline: predict the majority
    #counts = np.bincount(Y_test)
    #majority = np.argmax(counts)
    #baseline_acc = sum(Y_test==majority)/len(Y_test)
    #print("baseline accuracy = " + str(baseline_acc))
    
    #Y_test_pred = lasso.predict(X_test)
    #Y_test_pred = np.rint(Y_test_pred)
    #acc_test = sum(Y_test_pred==Y_test)/len(Y_test_pred)
    
    test_score=lasso.score(X_test, Y_test)
    
    #print("testing accuracy = " + str(acc_test))
    #print("testing score: " + str(test_score))
    return test_score


def lassoMultiRegress_trainTest_noPrint(X, Y, penalty):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=31)
    
    lasso = Lasso(alpha=penalty,max_iter=10e5) #alpha=0.0001 # NOT use: normalize=True
    lasso.fit(X_train,Y_train)
    
    test_score=lasso.score(X_test, Y_test)
    
    return test_score
    
    





def permutationTest(X, Y, true_test_score, penalty):
    # the permutation test to get p value.
    # referenced from Qianli, see my note:
    # Randomly permute score N = 500 times;
    # for each permutation, fit and get the result;
    # count for how many times in the N = 500 times,
    # the test score is >= the true test score. This number of times is P_val.
    N = 500 #1000
    n = 0
    for i in range(N):
        random.shuffle(Y)
        permu_test_score = lassoMultiRegress_trainTest_noPrint(X, Y, penalty)
        if permu_test_score >= true_test_score:
            n += 1
        #if (i%100 == 0):
        #    print(i)
    
    p_val = (n+1)/N
    
    return p_val




def lasso_searchPenalty_wrapper(X, Y, penalty_list):
    
    test_score_differentPenalty = []
    for penalty in penalty_list:
        #print("try penalty = " + str(penalty)) # just for debug!
        test_scores = []
        for t in range(1):
            test_score = lassoMultiRegress_searchPenalty(X, Y, penalty)
            test_scores.append(test_score)
        test_score_differentPenalty.append(np.mean(test_scores))
    
    max_test_score_idx = np.argmax(test_score_differentPenalty)
    max_test_score = test_score_differentPenalty[max_test_score_idx]
    max_test_score_penalty = penalty_list[max_test_score_idx]
    print("max testing score: " + str(max_test_score))
    print("corresponding penalty: " + str(max_test_score_penalty)) # --> 0.001
    
    
    return max_test_score_penalty



def lasso_featureSelect(X,Y,penalty):
    # referenced from lasso_featureSelect(X,Y,penalty)
    lasso = Lasso(alpha=penalty,max_iter=10e5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=31) #random_state=31; 21; 1
    lasso.fit(X_train,Y_train)
    
    # 0) Get the list of feature names
    # feat_labels = ...?
    
    # 2) Create a selector object that will use the random forest classifier to identify
    # features that have an importance of more than 0.15:
    sfm = SelectFromModel(lasso, threshold=1e-10)
    # 3) Train the selector:
    sfm.fit(X_train, Y_train)
    # 4) Print the names of the most important features:
    for feature_list_index in sfm.get_support(indices=True):
        #print(feat_labels[feature_list_index])
        print(feature_list_index)
    # 5) Create a data subset with only the most important features:
    X_important_train = sfm.transform(X_train)
    X_important_test = sfm.transform(X_test)
    # 6) Train a new lasso regressor using only most important features:
    lasso_important = Lasso(alpha=penalty,max_iter=10e5)
    lasso_important.fit(X_important_train, Y_train)
    
    # 7) get the training and test score for this limit feature lasso:
    # 7.1) for train:
    print("---train---")
    train_important_score = lasso_important.score(X_important_train, Y_train) # 0.11475
    print("training score for limit featLasso = " + str(train_important_score))
    # 7.2) for test:
    print("---test---")
    test_important_score = lasso_important.score(X_important_test, Y_test)
    print("testing accuracy for limit featLasso = " + str(test_important_score)) # -0.0471





if __name__ == '__main__':
    penalty_list = [1, 0.1, 0.01, 0.001, 0.0001] #, 0.00001
    
    """
    # X: data, of dim: (sample_num, feature_dim)--> (1576, 12)
    # Y_dict: Y_dict['xxx_score'] = Y
    # Y: target, of dim: (sample_num,) --> (1576,)
    X, Y_dict = generateDataTarget(srcDir)
    """
    
    # X: data, of dim: (sample_num, feature_dim)--> (1576, 79)
    # Y_dict: Y_dict['xxx_score'] = Y
    # Y: target, of dim: (sample_num,) --> (1576,)
    #X, Y_dict = generateDataTarget_v2(srcDir)
    #X, Y_dict = generateDataTarget_v3(srcDir)
    #X, Y_dict = generateDataTarget_v4(srcDir)
    #X, Y_dict = generateDataTarget_v5(srcDir)
    #X, Y_dict = generateDataTarget_v6(srcDir)
    #X, Y_dict = generateDataTarget_v7(srcDir)
    #X, Y_dict = generateDataTarget_v8(srcDir)
    #X, Y_dict = generateDataTarget_v9(srcDir)
    #X, Y_dict = generateDataTarget_v10(srcDir)
    #X, Y_dict = generateDataTarget_v11(srcDir)
    #X, Y_dict = generateDataTarget_v12(srcDir)
    #X, Y_dict = generateDataTarget_v13(srcDir)
    #X, Y_dict = generateDataTarget_v14(srcDir)
    #X, Y_dict = generateDataTarget_v13_cntCenterVertex(srcDir) # <-- similar to v13, bur for angSec, use cntCenterVertex
    #X, Y_dict = generateDataTarget_v13_arNpVertex(srcDir) # <-- similar to v13, bur for angSec, use arNpVertex
    #X, Y_dict = generateDataTarget_v13_cntCenterVertex_v2(srcDir) # <-- use v2 of EMD_cntCenterVertex
    #X, Y_dict = generateDataTarget_v13_cntCenterVertex_v3(srcDir) # <-- use v3 of EMD_cntCenterVertex
    #X, Y_dict = generateDataTarget_v15(srcDir) # add breastNorm_pixLocColor_EMD based on generateDataTarget_v13_cntCenterVertex
    #X, Y_dict = generateDataTarget_v16(srcDir, 'x')
    #X, Y_dict = generateDataTarget_v16(srcDir, 'y')
    #X, Y_dict = generateDataTarget_v16(srcDir, 'xynorm2')
    #X, Y_dict = generateDataTarget_v16(srcDir, 'xyangle')
    #X, Y_dict = generateDataTarget_v17(srcDir, 'x')
    #X, Y_dict = generateDataTarget_v17(srcDir, 'y')
    #X, Y_dict = generateDataTarget_v17(srcDir, 'r')
    #X, Y_dict = generateDataTarget_v17(srcDir, 't')
    #X, Y_dict = generateDataTarget_v18(srcDir)
    #X, Y_dict = generateDataTarget_v19(srcDir,'xylab')
    #X, Y_dict = generateDataTarget_v19(srcDir,'lab')
    #X, Y_dict = generateDataTarget_v20(srcDir,'lab')
    #X, Y_dict = generateDataTarget_v20(srcDir,'xylab')
    #X, Y_dict = generateDataTarget_v21(srcDir, 'y')
    #X, Y_dict = generateDataTarget_v21(srcDir, 'r')
    #X, Y_dict = generateDataTarget_v22(srcDir,'xylab')
    #X, Y_dict = generateDataTarget_v22(srcDir,'lab')
    #X, Y_dict = generateDataTarget_v23(srcDir,'xylab')
    #X, Y_dict = generateDataTarget_v24(srcDir)
    #X, Y_dict = generateDataTarget_v25(srcDir,'xylab')
    #X, Y_dict = generateDataTarget_v26(srcDir)
    
    ########## bovw_v1 ##########
    #X, Y_dict = generateDataTarget_v27(srcDir,'xylab',50) # <-- better than 80
    #X, Y_dict = generateDataTarget_v27(srcDir,'xylab',80)
    #X, Y_dict = generateDataTarget_v27(srcDir,'xylab',30) # <-- same results as 50
    #X, Y_dict = generateDataTarget_v27(srcDir,'xylab',20) # <-- same results as 50
    #X, Y_dict = generateDataTarget_v27(srcDir,'xylab',10) # similar to original v_23
    #X, Y_dict = generateDataTarget_v27(srcDir,'xylab',100) # similar to original v_23
    
    #X, Y_dict = generateDataTarget_v28(srcDir,'xylab',50) # <-- better than 80
    #X, Y_dict = generateDataTarget_v28(srcDir,'xylab',80)
    #X, Y_dict = generateDataTarget_v28(srcDir,'xylab',30) # <-- better than 50!!! <-- highest so far
    #X, Y_dict = generateDataTarget_v28(srcDir,'xylab',20) # similar to original v_25
    #X, Y_dict = generateDataTarget_v28(srcDir,'xylab',10) # similar to original v_25
    #X, Y_dict = generateDataTarget_v28(srcDir,'xylab',100) # similar to original v_25
    
    ########## bovw_v2 ##########
    #X, Y_dict = generateDataTarget_v29(srcDir,'xylab',50)
    #X, Y_dict = generateDataTarget_v29(srcDir,'xylab',500) # <-- overfitting!
    #X, Y_dict = generateDataTarget_v29(srcDir,'xylab',100) # worse than 50
    #X, Y_dict = generateDataTarget_v29(srcDir,'xylab',30) # better than 50
    #X, Y_dict = generateDataTarget_v29(srcDir,'xylab',10) # worse than 30
    #X, Y_dict = generateDataTarget_v29(srcDir,'xylab',80) # <-- similar to 100
    #X, Y_dict = generateDataTarget_v29(srcDir,'xylab',20) # <-- similar to 50
    
    #X, Y_dict = generateDataTarget_v30(srcDir,'xylab',50)
    #X, Y_dict = generateDataTarget_v30(srcDir,'xylab',500)  # <-- overfitting!
    #X, Y_dict = generateDataTarget_v30(srcDir,'xylab',100) # worse than 50
    #X, Y_dict = generateDataTarget_v30(srcDir,'xylab',30) # better than 50
    #X, Y_dict = generateDataTarget_v30(srcDir,'xylab',10) # similar to (slightly lower than) 30
    #X, Y_dict = generateDataTarget_v30(srcDir,'xylab',80) # <-- similar to 50
    #X, Y_dict = generateDataTarget_v30(srcDir,'xylab',20) # better than 30!!! <-- highest so far!
    
    #X, Y_dict = generateDataTarget_v31(srcDir,'v1')
    
    #X, Y_dict = generateDataTarget_v32(srcDir,'xylab',20,'v1') # 0.2703
    #X, Y_dict = generateDataTarget_v32(srcDir,'xylab',20,'v2') # 0.2587
    #X, Y_dict = generateDataTarget_v32(srcDir,'xylab',20,'v3') # 0.257 # 0.2777 
    #X, Y_dict = generateDataTarget_v32(srcDir,'xylab',20,'v4') # 0.2691
    
    #X, Y_dict = generateDataTarget_v35(srcDir,'xylab',20,'v3') #0.249
    
    #X, Y_dict = generateDataTarget_v36(srcDir,'xylab',20,'v3') #0.222
    
    #X, Y_dict = generateDataTarget_v37(srcDir,'xylab',20,'v3') # 0.2749 <-- highest so far!
    
    X, Y_dict = generateDataTarget_v41(srcDir) # just for test unsup3d features!
    
    
    # newly added: use z-score standardization to normalize X
    X_nonNorm = X # store the non-normalized data X
    X = zScoreNormalization(X)
    #X = zScoreNormalization_exceptBOVW(X,30) # <-- used only for v_27 & v_28 because of sparseness
    
    
    # CAUTION!: I need to deep copy y for permutationTest
    """
    print("**************skinColor**************")
    Y = Y_dict['Y_skinColor_score']
    max_test_score_penalty = lasso_searchPenalty_wrapper(X, Y, penalty_list)
    penalty = max_test_score_penalty
    test_score = lassoMultiRegress_trainTest(X, Y, penalty)
    cross_val_score_list = lassoMultiRegress_trainTest_CV(X, Y, penalty)
    cross_val_score_ = statistics.mean(cross_val_score_list)
    print("cross_val_score = " + str(cross_val_score_))
    cross_val_score_std = statistics.pstdev(cross_val_score_list)
    print("cross_val_score_std = " + str(cross_val_score_std))
    cross_val_score_stderror = stats.sem(np.array(cross_val_score_list))
    print("cross_val_score_stderror = " + str(cross_val_score_stderror))
    p_val = permutationTest(X, Y, test_score, penalty)
    print("p value = " + str(p_val))
    
    # do the feature selection using this lasso clf
    #lasso_featureSelect(X,Y,penalty) # <-- NOT use this! use randomForest to do selection instead!
    
    """
    #"""
    print("**************size**************")
    Y = Y_dict['Y_size_score']
    max_test_score_penalty = lasso_searchPenalty_wrapper(X, Y, penalty_list)
    penalty = max_test_score_penalty
    test_score = lassoMultiRegress_trainTest(X, Y, penalty)
    cross_val_score_list = lassoMultiRegress_trainTest_CV(X, Y, penalty)
    cross_val_score_ = statistics.mean(cross_val_score_list)
    print("cross_val_score = " + str(cross_val_score_))
    cross_val_score_std = statistics.pstdev(cross_val_score_list)
    print("cross_val_score_std = " + str(cross_val_score_std))
    cross_val_score_stderror = stats.sem(np.array(cross_val_score_list))
    print("cross_val_score_stderror = " + str(cross_val_score_stderror))
    p_val = permutationTest(X, Y, test_score, penalty)
    print("p value = " + str(p_val))
    #"""
    #"""
    print("**************shape**************")
    Y = Y_dict['Y_shape_score']
    max_test_score_penalty = lasso_searchPenalty_wrapper(X, Y, penalty_list)
    penalty = max_test_score_penalty
    test_score = lassoMultiRegress_trainTest(X, Y, penalty)
    cross_val_score_list = lassoMultiRegress_trainTest_CV(X, Y, penalty)
    cross_val_score_ = statistics.mean(cross_val_score_list)
    print("cross_val_score = " + str(cross_val_score_))
    cross_val_score_std = statistics.pstdev(cross_val_score_list)
    print("cross_val_score_std = " + str(cross_val_score_std))
    cross_val_score_stderror = stats.sem(np.array(cross_val_score_list))
    print("cross_val_score_stderror = " + str(cross_val_score_stderror))
    p_val = permutationTest(X, Y, test_score, penalty)
    print("p value = " + str(p_val))
    #"""
    """
    print("**************scar**************")
    Y = Y_dict['Y_scar_score']
    max_test_score_penalty = lasso_searchPenalty_wrapper(X, Y, penalty_list)
    penalty = max_test_score_penalty
    test_score = lassoMultiRegress_trainTest(X, Y, penalty)
    cross_val_score_list = lassoMultiRegress_trainTest_CV(X, Y, penalty)
    cross_val_score_ = statistics.mean(cross_val_score_list)
    print("cross_val_score = " + str(cross_val_score_))
    cross_val_score_std = statistics.pstdev(cross_val_score_list)
    print("cross_val_score_std = " + str(cross_val_score_std))
    cross_val_score_stderror = stats.sem(np.array(cross_val_score_list))
    print("cross_val_score_stderror = " + str(cross_val_score_stderror))
    p_val = permutationTest(X, Y, test_score, penalty)
    print("p value = " + str(p_val))
    """
    """
    print("**************nipple**************")
    Y = Y_dict['Y_nipple_score']
    max_test_score_penalty = lasso_searchPenalty_wrapper(X, Y, penalty_list)
    penalty = max_test_score_penalty
    test_score = lassoMultiRegress_trainTest(X, Y, penalty)
    cross_val_score_list = lassoMultiRegress_trainTest_CV(X, Y, penalty)
    cross_val_score_ = statistics.mean(cross_val_score_list)
    print("cross_val_score = " + str(cross_val_score_))
    cross_val_score_std = statistics.pstdev(cross_val_score_list)
    print("cross_val_score_std = " + str(cross_val_score_std))
    cross_val_score_stderror = stats.sem(np.array(cross_val_score_list))
    print("cross_val_score_stderror = " + str(cross_val_score_stderror))
    p_val = permutationTest(X, Y, test_score, penalty)
    print("p value = " + str(p_val))
    """
    """
    print("**************global**************")
    Y = Y_dict['Y_global_score']
    max_test_score_penalty = lasso_searchPenalty_wrapper(X, Y, penalty_list)
    penalty = max_test_score_penalty
    test_score = lassoMultiRegress_trainTest(X, Y, penalty)
    cross_val_score_list = lassoMultiRegress_trainTest_CV(X, Y, penalty)
    cross_val_score_ = statistics.mean(cross_val_score_list)
    print("cross_val_score = " + str(cross_val_score_))
    cross_val_score_std = statistics.pstdev(cross_val_score_list)
    print("cross_val_score_std = " + str(cross_val_score_std))
    cross_val_score_stderror = stats.sem(np.array(cross_val_score_list))
    print("cross_val_score_stderror = " + str(cross_val_score_stderror))
    p_val = permutationTest(X, Y, test_score, penalty)
    print("p value = " + str(p_val))
    """



