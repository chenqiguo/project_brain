#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:28:32 2021

@author: guo.1648
"""

import os
import scipy.io as sio

from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

from loadMatFile_myTest2 import loadMatFile, loadTargetMatFile


class MyCustomMatData(Dataset):
    """Custom Dataset.
    """
    
    def __init__(self, root, label_name):
        """
        Args:
            root (string): Directory with all the mat files.
            label_name (string): Behavior name to be regressed.
        """
        self.root = root
        self.label_name = label_name
        self.mat_names = os.listdir(root) # a list of mat file names
        
        # load all the mat files: connL, connR, behavior_val(label):
        #self.connData_list, self.targets = loadMatFile(root, label_name) # return two lists <-- too slow!
        #self.classes = np.unique(self.targets) # may NOT need since we are doing regression?
    
    def __len__(self):
        return len(self.mat_names)
    
    def __getitem__(self, index):
        #target = self.targets[index]
        #connData = self.connData_list[index]
        
        target_mat_name = self.mat_names[index]
        
        connData, target = loadTargetMatFile(self.root, self.label_name, target_mat_name)
        
        
        """
        # NO transform!
        if self.transform is not None:
            pos_1 = self.transform(image)
        """
        
        return connData, target, target_mat_name



