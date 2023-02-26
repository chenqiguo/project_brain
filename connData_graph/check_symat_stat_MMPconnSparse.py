#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:15:03 2021

@author: guo.1648
"""

# check fc matrices element values statistics, to determine the potential
# thresholds later for generating graphFeat.

# referenced from check_symat_stat_product.py


import os
import numpy as np
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnSparse/'


if __name__ == '__main__':
    
    all_ele_list = []
    
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
                fc = mat_contents['fc'] # (379, 379)
                
                fc_flt = fc.flatten().tolist()
                all_ele_list += fc_flt                

    # get statistics for all_ele_list:
    min_ = min(all_ele_list) # -0.6024935993563694
    max_ = max(all_ele_list) # 1.0
    mean_ = np.mean(all_ele_list) # 0.16256235249074563
    #majority = max(set(all_ele_list), key = all_ele_list.count)
    std_ = np.std(all_ele_list) # 0.16289989490860418
    median_ = np.median(all_ele_list) # 0.1327555563949981
    
    # plot & save hist for all_ele_list:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(all_ele_list, density=True, bins=30)  # density=False would make counts
    
    bin_max = np.argmax(n) # 12
    x_peak = bins[bin_max] # 0.03850384038617838
    
    title_str = 'fc matrices all element values statistics (v2_MMPconnSparse)'
    plt.title(title_str)
    plt.ylabel('Probability')
    plt.xlabel('Mat element value')
    textstr = 'mean = %s' % float('%.2g' % mean_) + '\nstd = %s' % float('%.2g' % std_) \
              + '\nx_peak = %s' % float('%.2g' % x_peak) + '\nmedian = %s' % float('%.2g' % median_)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    plt.text(0.7,0.7, textstr, verticalalignment='top', transform=ax.transAxes, bbox=props) # for target_list_orig
    
    fig.savefig('ele_stat_v2_MMPconnSparse.png')


