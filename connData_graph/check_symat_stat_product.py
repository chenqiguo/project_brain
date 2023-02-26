#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:30:25 2021

@author: guo.1648
"""

# check connArray_sym matrices element values statistics, to determine the potential
# thresholds later for generating graphFeat.

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


srcPkl = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_graph/symmetric_connData/v1_product/connArrays_productSymm.pkl'


if __name__ == '__main__':
    
    f_pkl = open(srcPkl,'rb')
    matDict_list = pickle.load(f_pkl)
    f_pkl.close()
    
    all_ele_list = []
    
    for dict_ in matDict_list:
        connArray_sym = dict_['connArray_sym']
        
        connArray_sym_flt = connArray_sym.flatten().tolist()
        all_ele_list += connArray_sym_flt
    
    #"""
    # get statistics for all_ele_list:
    min_ = min(all_ele_list) # -1401.4608730106645
    max_ = max(all_ele_list) # 5845.986596396486
    mean_ = np.mean(all_ele_list) # 408.700052310907
    #majority = max(set(all_ele_list), key = all_ele_list.count)
    std_ = np.std(all_ele_list) # 505.7515198664627
    median_ = np.median(all_ele_list) # 242.46390008245805
    
    # plot & save hist for all_ele_list:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(all_ele_list, density=True, bins=30)  # density=False would make counts
    
    bin_max = np.argmax(n) # 6
    x_peak = bins[bin_max] # 48.02862087076551
    
    title_str = 'connArray_sym matrices all element values statistics (v1_product)'
    plt.title(title_str)
    plt.ylabel('Probability')
    plt.xlabel('Mat element value')
    textstr = 'mean = %s' % float('%.2g' % mean_) + '\nstd = %s' % float('%.2g' % std_) \
              + '\nx_peak = %s' % float('%.2g' % x_peak) + '\nmedian = %s' % float('%.2g' % median_)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    plt.text(0.5,0.5, textstr, verticalalignment='top', transform=ax.transAxes, bbox=props) # for target_list_orig
    
    fig.savefig('ele_stat_v1_product.png')
    #"""




