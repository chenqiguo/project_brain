'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''

# Modified by Chenqi according to our mat files.
# Referenced from loadMatFile_myTest2.py and utils_myTest2.py

import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

import scipy.io as sio
import math

from data_utils.loadMatFile_origTarget import loadTargetMatFile, loadTargetMatFile_xyz


warnings.filterwarnings('ignore')


"""
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
"""

"""
def farthest_point_sample(point, npoint):
    
    # Input:
    #     xyz: pointcloud data, [N, D]
    #     npoint: number of samples
    # Return:
    #     centroids: sampled pointcloud index, [npoint, D]
    
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
"""


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train'): # , process_data=False
        self.root = root # mat data dir: /eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/
        #self.process_data = process_data
        
        self.label_name = args.label_name
        self.train_test_root = args.train_test_root # e.g. /eecf/cbcsl/data100b/Chenqi/project_brain/Pointnet2_myCustom/train_test_split/train7test3/
        #self.npoints = args.num_point
        #self.uniform = args.use_uniform_sample
        #self.use_normals = args.use_normals
        #self.num_category = args.num_category
        
        # Note: in our case we do not have classes, since we are doing regression.
        
        mat_ids = {}
        mat_ids['train'] = [line.rstrip() for line in open(os.path.join(self.train_test_root, 'train.txt'))]
        mat_ids['test'] = [line.rstrip() for line in open(os.path.join(self.train_test_root, 'test.txt'))]
        
        assert (split == 'train' or split == 'test')
        self.datapath = [os.path.join(self.root, mat_ids[split][i]) for i in range(len(mat_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))
        
        # Note: we do not do uniform either, since the brain points are already sampled&normalized to standard brain.
        
        # Note: do not save data since its too large!
        
    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        
        target_mat_fullname = self.datapath[index] # this mat file fullname: fn[1]
        
        #connData_xyz, target = loadTargetMatFile_xyz(self.label_name, target_mat_fullname) # v1
        connData, target = loadTargetMatFile(self.label_name, target_mat_fullname) # v2
        
        # Note: in original Pointnet2 code, each row of point_set is: (x,y,z,normVecInfo1,normVecInfo2,normVecInfo3).
        # Here since each row of connData_xyz is: (x,y,z, correlation coeff between this point and each region),
        # and x,y,z are all the same across subjects, we do not need normalize or uniform.
        # ??? May cause problem since our data xyz are all the same ???
        #point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # Not Use
        
        # deal with (x,y,z): for v1 only
        #connData_xyz[:, 0:3] = pc_normalize(connData_xyz[:, 0:3]) # ??? should I do this ???
        
        return connData, target, target_mat_fullname # torch.Size([B, 64984, 379]), torch.Size([B])
    
    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__': # just for debug
    import torch
    import argparse
    
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--train_test_root', type=str, default='/eecf/cbcsl/data100b/Chenqi/project_brain/Pointnet2_myCustom/train_test_split/train7test3/', help='Root dir of train-test spilt txt files')
    parser.add_argument('--label_name', default='DSM_Anxi_T', type=str, help='Behavior name to be regressed')
    #parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    #parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    #parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    #parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    
    args = parser.parse_args()
    
    data = ModelNetDataLoader('/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnMesh/', args, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for point, label, _ in DataLoader:
        print(point.shape)
        print(label.shape)


