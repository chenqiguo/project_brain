#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:17:29 2021

@author: guo.1648
"""

# version 2:
# instead of using sklearn lasso package,
# here we try to train the same model iteratively on data:
# first on X[:100], then on X[100:200], ...

# original target regression version, for all subjects.

# This is for sparse linear regression (as baseline);
# threshold the connData to produce a binary adjacency matrix as input.

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

dstRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/regression_baseline/results/sparseLinearRegress_origTarget_v2/'

threshold = 0.05 # threshold for connData
tolerance = 2 # for computing acc


def loadDataTarget_orig(flag):
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
        # threshold the connData to produce a binary adjacency matrix as input:
        dataMat_feat = connArray >= threshold # bool array
        
        behavior_val = None
        behavior_names = mat_contents['behavior_names']
        for i,element in enumerate(behavior_names[0]):
            this_behavior_name = element[0]
            if this_behavior_name == target_behavior_name:
                behavior_val = mat_contents['behavior'][0][i]
                break
        assert(behavior_val is not None)
        
        """
        # binary classification target:
        if target_behavior_name == 'DSM_Anxi_T' and not math.isnan(behavior_val):
            if behavior_val == 50: # 51
                behavior_val = 0
            else:
                behavior_val = 1
        """
        
        conn_list.append(dataMat_feat.flatten())
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
    """
    conn_list_train, behavior_val_list_train = loadDataTarget_orig(flag = 'train')
    conn_list_test, behavior_val_list_test = loadDataTarget_orig(flag = 'test')
    """
    
    """
    batch_size = 100
    w_thisBatchItr = None
    
    print('******************* training *******************')
    train_itr_num = math.ceil(len(behavior_val_list_train) / batch_size)
    for i in range(train_itr_num):
        print('-------- iter ' + str(i) + ':')
        # only deal with batch_size samples each time:
        if i==train_itr_num-1: # for last iter
            X_train = np.array(conn_list_train[i*batch_size:]).astype('uint8')
            y_train = behavior_val_list_train[i*batch_size:]
        else:
            X_train = np.array(conn_list_train[i*batch_size:(i+1)*batch_size]).astype('uint8')
            #X_train_sp = sparse.coo_matrix(X_train)
            y_train = behavior_val_list_train[i*batch_size:(i+1)*batch_size]
        
        if i==0: # for first iter
            model = LassoRegression(learning_rate=0.01, iterations=100, l1_penality=0.001) # iterations=200; ???penalty = 1; 0.1???
        else: # load the pretrained weights w
            model = LassoRegression(learning_rate=0.01, iterations=100, l1_penality=0.001, pretrained=True, pretrain_weight=w_thisBatchItr)
        
        model.fit(X_train, y_train)
        
        # model weights for this iter of data_batch:
        w_thisBatchItr = model.w # of shape (24628937,)
        # model mae (list of iterations) for this iter of data_batch
        trainErr_thisBatchItr = model.training_errors # len=iterations
        
        # save above check points into pkls:
        checkpts_thisBatchItr = {'iter': i,
                                 'train_test': 'train',
                                 'model_w': w_thisBatchItr,
                                 'mae': trainErr_thisBatchItr[-1]}
        
        print('checkpts_thisBatchItr[\'mae\'] = ' + str(checkpts_thisBatchItr['mae']))
        
        f_pkl = open(dstRootDir+'checkpts/train_iter'+str(i)+'.pkl', 'wb')
        pickle.dump(checkpts_thisBatchItr,f_pkl)
        f_pkl.close()
    """
    #"""
    print('******************* testing *******************')
    test_totalCorrectNum = 0
    test_mae_list = []
    test_itr_num = math.ceil(len(behavior_val_list_test) / batch_size)
    for j in range(test_itr_num):
        print('-------- iter ' + str(j) + ':')
        # only deal with batch_size samples each time:
        if j==test_itr_num-1: # for last iter
            X_test = np.array(conn_list_test[j*batch_size:]).astype('uint8')
            y_test = behavior_val_list_test[j*batch_size:]
        else:
            X_test = np.array(conn_list_test[j*batch_size:(j+1)*batch_size]).astype('uint8')
            #X_test_sp = sparse.coo_matrix(X_test)
            y_test = behavior_val_list_test[j*batch_size:(j+1)*batch_size]
        
        test_pred = model.predict(X_test)
        #print(test_pred)
        #test_pred_bc = ((np.clip(test_pred,0,1)) >= 0.5)
        
        # newly modified: use tolerance for acc:
        #test_totalCorrectNum += np.sum(test_pred_bc == y_test) #/ len(y_test)
        difference = np.absolute(test_pred - y_test)
        test_totalCorrectNum += np.count_nonzero(difference <= tolerance)
        
        test_mae = np.linalg.norm((y_test - test_pred), ord=1)/len(y_test) + model.regularization(model.w)/X_test.shape[1]
        test_mae_list.append(test_mae)
        
    test_acc = test_totalCorrectNum / len(behavior_val_list_test)
    print('test_acc = ' + str(test_acc))
    print('test_mae_list = ' + str(test_mae_list))
    #"""
    

