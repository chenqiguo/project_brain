'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''

# This is for sparse linear regression (as baseline)

import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

#import scipy.io as sio
import math

from data_utils_baseline.loadMatFile_bc import loadTargetMatFile_sparse


warnings.filterwarnings('ignore')




class ModelNetDataLoader_sparse(Dataset):
    def __init__(self, args, split='train'): # , process_data=False
        
        self.label_name = args.label_name
        self.train_test_root = args.train_test_root # e.g. /eecf/cbcsl/data100b/Chenqi/project_brain/Pointnet2_myCustom/train_test_split/train7test3/
        self.data_dir = args.data_dir # orig mat data root dir
        self.threshold = args.threshold # threshold of connData
        
        # Note: in our case we do not have classes, since we are doing regression.
        
        mat_ids = {}
        mat_ids['train'] = [line.rstrip() for line in open(os.path.join(self.train_test_root, 'train.txt'))]
        mat_ids['test'] = [line.rstrip() for line in open(os.path.join(self.train_test_root, 'test.txt'))]
        
        # newly modified:
        assert (split == 'train' or split == 'test' or split == 'all')
        if split == 'all':
            self.subjectFileNames = mat_ids['train'] + mat_ids['test']
        else:
            self.subjectFileNames = mat_ids[split]
        
        print('The size of %s data is %d' % (split, len(self.subjectFileNames)))
        
        
    def __len__(self):
        return len(self.subjectFileNames)

    def _get_item(self, index):
        
        target_subjectFileName = self.subjectFileNames[index] # this mat file: e.g., '704238.mat'
        
        dataMat_feat, target = loadTargetMatFile_sparse(self.data_dir, self.label_name, target_subjectFileName, self.threshold) # dataMat_feat of dim (32,32)
        
        return dataMat_feat, target, target_subjectFileName # torch.Size([B, 32, 32]), torch.Size([B])
    
    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__': # just for debug
    import torch
    import argparse
    
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--data_dir', default='/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/', type=str, help='Dir of the mat dataset')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/ResNet/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    parser.add_argument('--threshold', type=float, default=0.05, help='threshold for connData')
    
    args = parser.parse_args()
    
    data = ModelNetDataLoader_sparse(args, split='all') #train
    DataLoader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    for point, label, _ in DataLoader:
        print(point.shape) # torch.Size([100, 32, 32]), dtype=torch.float64
        print(label.shape) # torch.Size([100])
        
        #print()
