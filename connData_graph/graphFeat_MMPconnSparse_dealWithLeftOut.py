#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 16:22:37 2021

@author: guo.1648
"""

# NOT USED: re-run the graphFeat_MMPconnSparse instead!!!

# This code deals with the left out mat files when running graphFeat_MMPconnSparse.py,
# and will merge the new results to the previously generated graphFeat pkl files.


import os
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io as sio


srcRootDir = '/eecf/cbcsl/data100b/Chenqi/project_brain/MMPconnSparse/'

dstRootDir_draw = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_graph/graphDraw/v2_MMPconnSparse/'
dstRootDir_feat = '/eecf/cbcsl/data100b/Chenqi/project_brain/connData_graph/graphFeat/v2_MMPconnSparse/'


if __name__ == '__main__':
    
    # the list of potential threshold vals that are left out:
    threshold_list = [0.5, 0.6, 0.7]
    
    for threshold in threshold_list:
        print('--------------- threshold = ' + str(threshold))
        
        # list to store all the corresponding graph features for this threshold:
        feat_dict_list = []
        
        # the folder for this threshold value:
        dstDir_draw = dstRootDir_draw + 'thresh'+str(threshold)+'/'
        dstDir_feat = dstRootDir_feat + 'thresh'+str(threshold)+'/'
        
        for (dirpath, dirnames, filenames) in os.walk(srcRootDir):
            #print(filenames)
            for filename in filenames:
                #print(filename)
                if ".mat" in filename:
                    
                    # only deal with the left out mat files:
                    if (filename == '125222.mat') or (threshold>=0.6 and filename == '456346.mat') or (threshold==0.7 and filename == '548250.mat') or (threshold==0.7 and filename == '547046.mat'):
                        
                        print('*** ' + filename)
                        fullFileName = srcRootDir + filename
                        mat_contents = sio.loadmat(fullFileName)
                        
                        # get mat fields:
                        fc = mat_contents['fc'] # (379, 379)
                        
                        fc_thresh = (fc > threshold).astype(int) # the adjacency mat
                        assert(np.all(fc_thresh.T == fc_thresh)) # symmetric mat!!!
                        
                        G = nx.from_numpy_array(fc_thresh) # un-directed graph!!!
                        
                        # draw & save the graph of G:
                        fig = plt.figure(figsize=(12,12))
                        ax = fig.add_subplot(111)
                        ax.set_title('Graph of fc_thresh', fontsize=10)
                        nx.draw(G, with_labels=True) #, font_weight='bold'
                        
                        plt.savefig(dstDir_draw+filename.split('.')[0]+'_graph.png', format="PNG")
                        plt.close()
                        
                        # get graph features for G:
                        # NOTE: below all use default param --> modify later???
                        # (1) The degree centrality for a node v is the fraction of nodes it is connected to:
                        degree_centrality = nx.degree_centrality(G)
                        # (2) Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors:
                        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=10000)
                        """
                        # (3) Katz centrality computes the relative influence of a node within a network by measuring the number
                        #     of the immediate neighbors (first degree nodes) and also all other nodes in the network that connect
                        #     to the node under consideration through these immediate neighbors:
                        katz_centrality = nx.katz_centrality(G) # power iteration failed to converge within 1000 iterations!
                        """
                        # (4) Closeness centrality of a node u is the reciprocal of the average shortest path distance to u
                        #     over all n-1 reachable nodes. Notice that higher values of closeness indicate higher centrality.
                        #     Here we use Wasserman and Faust's improved formula:
                        closeness_centrality = nx.closeness_centrality(G)
                        """
                        # (5) Current-flow closeness centrality is variant of closeness centrality based on effective resistance
                        #     between nodes in a network. This metric is also known as information centrality
                        current_flow_closeness_centrality = nx.current_flow_closeness_centrality(G) # networkx.exception.NetworkXError: Graph not connected
                        """
                        # (6) Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v:
                        betweenness_centrality = nx.betweenness_centrality(G)
                        """
                        # (7) Current-flow betweenness centrality (also known as random-walk betweenness centrality) uses
                        #     an electrical current model for information spreading in contrast to betweenness centrality which uses shortest paths:
                        current_flow_betweenness_centrality = nx.current_flow_betweenness_centrality(G) # NetworkXError: Graph not connected
                        """
                        """
                        # (8) Communicability betweenness measure makes use of the number of walks connecting every pair of nodes
                        #     as the basis of a betweenness centrality measure
                        communicability_betweenness_centrality = nx.communicability_betweenness_centrality(G) # invalid value encountered in true_divide
                        """
                        # (9) The load centrality of a node is the fraction of all shortest paths that pass through that node:
                        load_centrality = nx.load_centrality(G)
                        # (10) Subgraph centrality of a node n is the sum of weighted closed walks of all lengths starting and ending at node n.
                        #      The weights decrease with path length. Each closed walk is associated with a connected subgraph:
                        subgraph_centrality = nx.subgraph_centrality(G)
                        # (11) The Estrada Index is a topological index of folding or 3D “compactness”:
                        estrada_index = nx.estrada_index(G)
                        # (12) Harmonic centrality of a node u is the sum of the reciprocal of the shortest path distances
                        #      from all other nodes to u:
                        harmonic_centrality = nx.harmonic_centrality(G)
                        # (13) Select a list of influential nodes in a graph:
                        voterank = nx.voterank(G) # Only nodes with positive number of votes are returned
                        
                        feat_dict = {'degree_centrality': degree_centrality,
                                     'eigenvector_centrality': eigenvector_centrality,
                                     'closeness_centrality': closeness_centrality,
                                     #'current_flow_closeness_centrality': current_flow_closeness_centrality,
                                     'betweenness_centrality': betweenness_centrality,
                                     #'current_flow_betweenness_centrality': current_flow_betweenness_centrality,
                                     #'communicability_betweenness_centrality': communicability_betweenness_centrality,
                                     'load_centrality': load_centrality,
                                     'subgraph_centrality': subgraph_centrality,
                                     'estrada_index': estrada_index,
                                     'harmonic_centrality': harmonic_centrality,
                                     'voterank': voterank}
                        
                        feat_dict_list.append(feat_dict)
                        
        # save this feat_dict_list for this threshold val into this dstDir_feat:
        f_pkl = open(dstDir_feat+'graphFeat_dict_list_leftOut.pkl', 'wb')
        pickle.dump(feat_dict_list,f_pkl)
        f_pkl.close()
        
        """
        # merge the graphFeat_dict_list_leftOut.pkl and graphFeat_dict_list.pkl:
        f_pkl = open(srcPklDir_tmp,'rb')
        feat_dict_list_prev = pickle.load(f_pkl)
        f_pkl.close()
        """
    








