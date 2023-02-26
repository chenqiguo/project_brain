#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 10:37:45 2021

@author: guo.1648
"""

# this model will be used after BrainNetCNN_res_model_v3 model.py is called.

# use 2-layer deep fully-connected neural network (FNN) v4 with flatten input.
# v4: outputs all the 35 target values each time; referenced from the paper's github:
# hcp_fnn()

# NOTE: here we tune our new hyper-params!!!


from keras.layers import Input, Dropout, Dense, Activation, LeakyReLU, Conv2D
from keras.layers import Lambda, Flatten
from keras.models import Model
from keras.layers.merge import add
from keras.layers.core import Permute
from keras import backend as K
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization


def hcp_fnn(input_shape, n_measure, n_l1, n_l2, dropout, l2_reg):
    """FNN model for HCP dataset
    Args:
        input_shape (int): dimension of input x
        n_measure (int): number of behavioral measures
        n_l1 (int): number of node for GCNN layer 1
        n_l2 (int): number of node for GCNN layer 2
        dropout (float): dropout rate
        l2_reg (float): l2 regularizer rate
    Returns:
        keras.models.Model: FNN model
    """
    init_method = 'glorot_uniform'
    model_in = Input(shape=(input_shape, ))
    H = Dropout(dropout)(model_in)
    H = Dense(n_l1, kernel_initializer=init_method)(H)
    H = Activation('linear')(H)
    H = BatchNormalization()(H)
    H = Dropout(dropout)(H)
    H = Dense(
        n_l2, kernel_initializer=init_method, kernel_regularizer=l2(l2_reg))(H)
    H = Activation('linear')(H)
    H = BatchNormalization()(H)
    H = Dropout(dropout)(H)
    model_out = Dense(n_measure, kernel_initializer=init_method)(H)
    return Model(model_in, model_out)






