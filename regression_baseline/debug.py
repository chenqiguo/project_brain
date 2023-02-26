#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 02:17:03 2021

@author: guo.1648
"""

import numpy as np
import math

from sklearn.linear_model import Lasso
from scipy import sparse




if __name__ == '__main__':
    dummy_X_iter1 = np.array([[1],[2],[3]])
    dummy_X_iter2 = np.array([[1],[2],[0]])
    
    dummy_y_iter1 = [1,2,3]
    dummy_y_iter2 = [2,1,3]
    
    penalty = 0 # 0.1
    sparse_lasso = Lasso(alpha=penalty,max_iter=10e5)
    
    sparse_lasso.fit(dummy_X_iter1,dummy_y_iter1)
    coeffs_iter1 = sparse_lasso.coef_
    print(coeffs_iter1)
    intercept_iter1 = sparse_lasso.intercept_
    print(intercept_iter1)
    
    print('*********')
    
    sparse_lasso.fit(dummy_X_iter2,dummy_y_iter2)
    coeffs_iter2 = sparse_lasso.coef_
    print(coeffs_iter2)
    intercept_iter2 = sparse_lasso.intercept_
    print(intercept_iter2)
    
    
    

