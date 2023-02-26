#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:44:32 2021

@author: guo.1648
"""

# compute mean & std val_loss and val_acc for 10-fold CV.


import numpy as np



val_loss_list = [0.8776, 1.5479, 0.6793, 0.9711, 0.5974, 0.6863, 0.5812,
                 1.1452, 1.1856, 0.9589]

val_acc_list =  [0.7650, 0.5700, 0.8150, 0.7950, 0.7800, 0.7800, 0.8000,
                 0.7550, 0.7550, 0.6000]


val_loss_mean = np.mean(val_loss_list)
val_loss_std = np.std(val_loss_list)
print('val_loss_mean = ' + str(val_loss_mean))
print('val_loss_std = ' + str(val_loss_std))

print()

val_acc_mean = np.mean(val_acc_list)
val_acc_std = np.std(val_acc_list)
print('val_acc_mean = ' + str(val_acc_mean))
print('val_acc_std = ' + str(val_acc_std))

