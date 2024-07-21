from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
#import fire
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

import extract_utils as utils

import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import warnings

import torch.nn as nn

class Landaloss(nn.Module):
    def __init__(self):
        super(Landaloss, self).__init__()

    def geteig(self,feats,K):

      #print('feats 1',feats.size())

      B,C,H,W= feats.size()
      new_H = int(H // 4)
      new_W = int(W // 4)
      feats = F.interpolate(feats, size=(new_H, new_W), mode='bilinear', align_corners=False)
      feats = feats.reshape(B,C,new_H*new_W).permute(0,2,1)
        
      #print('feats2 ',feats.size())
      '''
      B,N,C,H,W= feats.size()

      # Calculate the new height and width
      new_H = int(H // 4)
      new_W = int(W // 4)

      # Resize the tensor
      feats = F.interpolate(feats.view(-1, C, H, W), size=(new_H, new_W), mode='bilinear', align_corners=False).view(B, N, C, new_H, new_W)
      B,N,C,H,W= feats.size()

      feats = feats.reshape(B*N,C,H*W).permute(0,2,1)
      '''
    
    
      # Normalize features across the last dimension
      feats = F.normalize(feats, p=2, dim=-1)

      # Initialize a list to store results
      eigenvalues_list = []

      for b in range(B):
          W_feat = feats[b]

          # Feature affinities
          W_feat = (W_feat @ W_feat.T)
          W_feat = (W_feat * (W_feat > 0))

          W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
          W_feat = W_feat.detach().cpu().numpy()

          #D_comb = np.array(utils.get_diagonal(W_feat).todense())  # Check if dense or sparse is faster
          D_comb = np.array(utils.get_diagonal(W_feat).toarray())  # Use toarray() instead of todense()


          try:
              eigenvalues, eigenvectors = eigsh(D_comb - W_feat, k=K, sigma=0, which='LM', M=D_comb)
          except:
              eigenvalues, eigenvectors = eigsh(D_comb - W_feat, k=K, which='SM', M=D_comb)

          eigenvalues = torch.from_numpy(eigenvalues)
          eigenvalues_list.append(eigenvalues.cuda())

      # Convert list of eigenvalues to a tensor
      eigenvalues_tensor = torch.stack(eigenvalues_list)

      return eigenvalues_tensor

    def forward(self, feats , K):

      lan1 = self.geteig(feats[:,:1],K)
      lan2 = self.geteig(feats[:,1:2],K)
      lan3 = self.geteig(feats[:2:3],K)

        # Compute the pairwise MSE losses
      loss = F.mse_loss(lan1, lan2)
      loss = loss +  F.mse_loss(lan1, lan3)
      loss = loss +  F.mse_loss(lan2, lan3)

      return loss*5;
