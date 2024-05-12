import glob
from tqdm import tqdm
import extract_utils as utils
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import pickle
import os
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans


def metric(original_matrix,typ):
    if typ!='dot':
        output_matrix = 1/(1+pairwise_distances(original_matrix, metric=typ))
        output_matrix = np.nan_to_num(output_matrix , nan=0.0, posinf=0.0, neginf=0.0)
        return output_matrix
    else:
        return original_matrix @ original_matrix.T
    

def our_affinty(feats_sep,K=5):
    eps=0.000000001
    feats_sep = feats_sep.cpu().numpy()
    eigs={}
       
    W_featb = metric(feats_sep,'braycurtis')
    W_featc = metric(feats_sep,'chebyshev')
    W_feat = W_featb  / (W_featc)

    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max()
    W_comb = W_feat
    D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

    eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]#

   
    return [eigenvalues, eigenvectors]


def other_affinty(feats_sep,matric_name,K=5):
    eps=0.000000001
    feats_sep = feats_sep.cpu().numpy()
    eigs={}
       
    #W_featb = metric(feats_sep,'braycurtis')
    #W_featc = metric(feats_sep,'chebyshev')
    #W_feat = W_featb  / (W_featc)
    
    W_feat = metric(feats_sep,matric_name)

    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max()
    W_comb = W_feat
    D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

    eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]#

   
    return [eigenvalues, eigenvectors]
    
    
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code')
    
    
    with open('tmpfeat.obj', 'rb') as fp:
        [feats_sep,outname,aother]=pickle.load(fp)
    
    if aother=='0':
        output = our_affinty(feats_sep)
    else:
        #print(aother)
        output = other_affinty(feats_sep,aother)
    
    with open(outname, 'wb') as fp:
        pickle.dump(output, fp)
    
    '''
    parser.add_argument('--type', help='select extract_eigs or clustring')
    parser.add_argument('--feats_root', help='feature root path')
    parser.add_argument('--fgbg_root', help='prediction mask fgbg root path')
    parser.add_argument('--gt_root', help='grand-truth mask root path')
    parser.add_argument('--eigs_root', help='eigs root path')
    parser.add_argument('--export_root', help='export root path')
    parser.add_argument('--std_threshold', type=int, help='std threshold')
    parser.add_argument('--other', type=str, default = '0', help='type of metric')
    
    args = parser.parse_args()
    if args.type=='extract_eigs':
        extract_instance_eigs(args.feats_root,args.fgbg_root,args.export_root,args.std_threshold,args.other)
    else:
        clustring(args.eigs_root,args.fgbg_root,args.gt_root,args.export_root)
        
    '''