#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 17:15:39 2021

@author: guo.1648
"""

# NOT run this!!! --> out of memory even when loading the input lists!!!

# version 2:
# instead of using sklearn lasso package,
# here we try to train the same model iteratively on data:
# first on X[:100], then on X[100:200], ...

# binary classification (regression) version:
# thresh50: 0 (target==50) v.s. 1 (target>50)
# thresh51: 0 (target==51) v.s. 1 (target>51) <-- NOT did! Target is biased: lots of 1

# This is for original connData input linear regression (as another baseline);
# use the original flatten connData as input.

# referenced from sparseLinearRegress_bc_v2.py


import os
import scipy.io as sio
import numpy as np
import math
import pickle

#from sklearn.linear_model import Lasso
#from scipy import sparse


srcDir = '/scratch/Chenqi/project_brain/MMPconnMesh/'
train_test_root = '/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/'
target_behavior_name = 'DSM_Anxi_T'

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/regression_baseline/results/sparseLinearRegress_bc_v2/'



def loadDataTarget_nonThresh(flag):
    # flag = 'train' or 'test'
    
    conn_list = []
    behavior_val_list = []
    
    mat_ids = [line.rstrip() for line in open(os.path.join(train_test_root, flag+'.txt'))]
    for filename in mat_ids:
        print("------------------deal with---------------------")
        print(filename)
        fullFileName = srcDir + filename # full name of the mat file
        mat_contents = sio.loadmat(fullFileName)
        
        #print()
        # get mat fields:
        connL = mat_contents['connL']
        connR = mat_contents['connR']
        connArray = np.concatenate((np.array(connL), np.array(connR))) # (64984, 379)
        """
        # threshold the connData to produce a binary adjacency matrix as input:
        dataMat_feat = connArray >= threshold # bool array
        """
        
        behavior_val = None
        behavior_names = mat_contents['behavior_names']
        for i,element in enumerate(behavior_names[0]):
            this_behavior_name = element[0]
            if this_behavior_name == target_behavior_name:
                behavior_val = mat_contents['behavior'][0][i]
                break
        assert(behavior_val is not None)
        
        # binary classification target:
        if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
            if behavior_val == 50: # 51
                behavior_val = 0
            else:
                behavior_val = 1
        
        conn_list.append(connArray.flatten())
        behavior_val_list.append(behavior_val)
        
        #print()
    
    return (conn_list, behavior_val_list)


class l1_regularization():
    """ Regularization for Lasso Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, ord=1) # modified to be l1 norm
    def grad(self, w):
        return self.alpha * np.sign(w)


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


class LassoRegression() :
    
    def __init__( self,learning_rate,iterations,l1_penality, pretrained=False,pretrain_weight=None) :
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penality = l1_penality
        self.pretrained = pretrained
        self.pretrain_weight = pretrain_weight # pretrained W np array --> modified: also contains bias b !!!
        #self.pretrained_bias = pretrained_bias # pretrained bias value
        
        self.regularization = l1_regularization(alpha=l1_penality)
        
    def fit( self, X, Y) : # func of model training
        # ??? normalize input data X
        X = normalize(X) # ??? polynomial_features(X, degree=self.degree) ???
        
        self.X = np.insert(X, 0, 1, axis=1) # Insert constant ones for bias weights
        self.Y = Y
        self.training_errors = []
        
        self.m, self.n = self.X.shape  # num_of_training_examples, num_of_features(+1)
        
        # weight initialization:
        if self.pretrained:
            self.w = self.pretrain_weight
            #self.b = self.pretrained_bias
        else:
            # Initialize weights randomly [-1/N, 1/N]:
            limit = 1 / math.sqrt(self.n)
            self.w = np.random.uniform(-limit, limit, (self.n, ))
            #self.W = np.zeros( self.n )
            #self.b = 0
         
        # gradient descent learning for num of iterations:
        for i in range( self.iterations ) :
            #print('******** iter ' + str(i) + ':') # for debug
            self.update_weights()
        
        return self
    
    def update_weights( self ): # Helper function to update weights in gradient descent
        #print('self.X = ' + str(self.X))
        #Y_pred = self.predict( self.X )
        Y_pred = self.X.dot( self.w )
        """
        # for debug:
        print('self.X.shape = ' + str(self.X.shape))
        print('Y_pred = ' + str(Y_pred))
        """
        """
        # calculate l2 loss: overflow encountered in square !!!
        mse = np.mean(0.5 * (self.Y - Y_pred)**2 + self.regularization(self.w))
        self.training_errors.append(mse)
        """
        # calculate MAE = E[|Y-Y_pred|] + l1_norm(w)/m
        mae = np.linalg.norm((self.Y - Y_pred), ord=1) / self.m + self.regularization(self.w) / self.n
        self.training_errors.append(mae)
        
        # Gradient of l2 loss w.r.t w
        #print('computing grad_w ...') # for debug
        grad_w = -(self.Y - Y_pred).dot(self.X) + self.regularization.grad(self.w)
        
        """
        # for debug:
        print('grad_w = ' + str(grad_w))
        #print(-(self.Y - Y_pred).dot(self.X))
        #print(self.regularization.grad(self.w))
        #assert(False)
        """
        
        # Update the weights
        #print('updating w ...') # for debug
        self.w -= self.learning_rate * grad_w
        #print('self.w = ' + str(self.w))
        
        return self
    
    def predict( self, X ): # Hypothetical function h( x ): NOT called in fit()
        X = normalize(X)
        X = np.insert(X, 0, 1, axis=1) # Insert constant ones for bias weights
        #print(X.shape)
        return X.dot( self.w ) #+ self.b



if __name__ == '__main__':
    #"""
    conn_list_train, behavior_val_list_train = loadDataTarget_nonThresh(flag = 'train')
    conn_list_test, behavior_val_list_test = loadDataTarget_nonThresh(flag = 'test')
    #"""
    
    

