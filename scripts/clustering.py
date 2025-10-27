import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
import time
import random
import pickle as pkl
from nilearn import datasets, image
from nilearn.image import load_img, index_img
from nilearn.plotting import plot_epi
from nilearn.maskers import NiftiLabelsMasker
from nilearn.masking import apply_mask
import nibabel as nib
# from PyIF import te_compute as te
from joblib import delayed, Parallel
import pandas as pd
from scipy.stats import zscore
from scipy.optimize import curve_fit
from scipy.signal import convolve
from joblib import Parallel, delayed
import glob
import sys 

project_root = os.getcwd()
print(project_root)

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 gives it higher priority.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_loader import DataLoader
from utils.plotting import plot_voxels, plot_w_fit
from utils.relaxation_time import RelaxationTime
from utils.timer import TimerController
# from spatial_correlations import linear_map, binarize_lm

# import seaborn as sns
import scienceplots

plt.style.use(['science','no-latex', 'ieee'])
plt.rcParams.update({
    "font.family": "DejaVu Sans",   # specify font family here
})

global tc
tc = TimerController()

def linear_map(signal, ignore_nans=True):
    if not type(signal) is np.ndarray:
        signal = np.array(signal)
    if ignore_nans:
        a, b = np.nanmax(signal), np.nanmin(signal)
    else:
        a, b = max
        
        signal), min(signal)
    
    return (2*signal-(a+b))/(a - b)

def binarize(signal):
    b_signal = np.array(signal)
    b_signal[b_signal > 0] = 1
    b_signal[b_signal <= 0] = -1
    return b_signal

def get_nan_masked_data(data):
    skr_data = np.empty_like(data)
    skr_data[:] = np.nan
    for i in range(sk_idx.shape[1]):
        x, y, z = sk_idx[0][i], sk_idx[1][i], sk_idx[2][i]
        skr_data[x, y, z, :] = data[x, y, z, :]
    return skr_data


def get_mask(sk_idx, data_shape):
    mask = np.zeros(shape=data_shape)
    for i in range(sk_idx.shape[1]):
        x, y, z = sk_idx[0][i], sk_idx[1][i], sk_idx[2][i]
        mask[x, y, z] = 1
    return mask

class UnionFind:
    def __init__(self, arr_shape):
        n = arr_shape[0]*arr_shape[1]*arr_shape[2]
        self.parent = [i for i in range(n)]
        self.size = [1] * n
        self.arr_shape = arr_shape

    def ravel_index(self, idxs):
        return np.ravel_multi_index(idxs, self.arr_shape)
        
    def unravel_index(self, idx):
        return np.unravel_index(idx, self.arr_shape)
        
    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return
        if self.size[root_i] < self.size[root_j]:
            self.parent[root_i] = root_j
            self.size[root_j] += self.size[root_i]
        else:
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]

    def connected(self, i, j):
        return self.find(i) == self.find(j)
    
    

def get_clusters(bin_data, mask, t=50):
    arr_shape = bin_data.shape[:-1]
    uf = UnionFind(arr_shape)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                if mask[x, y, z] == 0:
                    uf.parent[uf.ravel_index([x, y, z])] = np.nan

    for x in range(arr_shape[0]):
        for y in range(arr_shape[1]):
            for z in range(arr_shape[2]):
                if mask[x, y, z]:

                    pt = bin_data[x, y, z, t]
                    idx = uf.ravel_index([x, y, z])
                    left, above, behind = None, None, None
                    lidx, aidx, bidx = None, None, None

                    if (x-1 > 0) and (mask[x-1, y, z]): 
                        left = bin_data[x-1, y, z, t]
                        lidx = uf.ravel_index([x-1, y, z])
                    if (y-1 > 0) and (mask[x, y-1, z]): 
                        above = bin_data[x, y-1, z, t]
                        aidx = uf.ravel_index([x, y-1, z])
                    if (z-1 > 0) and (mask[x, y, z-1]): 
                        behind = bin_data[x, y, z-1, t]
                        bidx = uf.ravel_index([x, y, z-1])

                    if (left != pt) and (above != pt) and (behind != pt):
                        # Retain label
                        pass
                    elif (left == pt) and (above != pt) and (behind != pt): 
                        # Join left and current
                        uf.union(idx, lidx)
                    elif (left != pt) and (above == pt) and (behind != pt): 
                        # Join above and current
                        uf.union(idx, aidx)
                    elif (left != pt) and (above != pt) and (behind == pt): 
                        # Join behind and current
                        uf.union(idx, bidx)
                    elif (left == pt) and (above == pt) and (behind != pt):
                        # Join current, left, above
                        uf.union(idx, lidx)
                        uf.union(idx, aidx)
                    elif (left == pt) and (above != pt) and (behind == pt): 
                        # Join current, left, behind
                        uf.union(idx, lidx)
                        uf.union(idx, bidx)
                    elif (left != pt) and (above == pt) and (behind == pt): 
                        # Join current, above, behind
                        uf.union(idx, aidx)
                        uf.union(idx, bidx)
                    elif (left == pt) and (above == pt) and (behind == pt): 
                        # Join all
                        uf.union(idx, lidx)
                        uf.union(idx, aidx)
                        uf.union(idx, bidx)
    uf.parent = np.array(uf.parent)
    return uf


def get_saved_fp(sub, t):
    return os.path.join(CLUSTERS_SAVE_DIR, sub, f'clusters-{t}.pkl')

def get_t_clusters(bin_data, mask, save=False):
    t_clusters = []
    print("t = ", flush=True)
    for t in range(bin_data.shape[-1]):
        print(t, end=" ", flush=True)
        clusters_uf = get_clusters(bin_data, mask, t=t)
        t_clusters.append(clusters_uf.parent)
    if save:
        print(f"Saving to {save}")
        with open(save, 'wb') as f:
            pkl.dump(t_clusters, f)
    print()
    return np.array(t_clusters)

def process_subject(sub):
    print(f"On sub {sub}", flush=True)
    tc.start_timer(sub)
    
    sub_save_fp = get_saved_fp(sub, "all")
    if os.path.exists(sub_save_fp): 
        print(f"Skipping subject {sub}")
        return None, None
    
    
    sub_save_dir = os.path.join(CLUSTERS_SAVE_DIR, sub)
    if not os.path.exists(sub_save_dir): os.makedirs(sub_save_dir)
    
    data = dloader.get_func(sub).get_fdata()
    skr_data = get_nan_masked_data(data)
#     lm_data = linear_map(skr_data)
    lm_data = np.apply_along_axis(linear_map, 3, skr_data)
    bin_data = binarize(lm_data)

    mask = get_mask(sk_idx, data.shape[:-1])
    t_clusters = get_t_clusters(bin_data, mask, save=sub_save_fp)
    
    print(f"Finished {sub} in {tc.get_elapsed_time(sub)} s")
#     n_clusters = []
#     for cluster in t_clusters:
#         n_clusters.append(len(set(cluster[~np.isnan(cluster)])))
    
    return None, t_clusters

if __name__ == "__main__":
    root_dir = '/scratch/anirudh.palutla/adni/preproc'
    CLUSTERS_SAVE_DIR = '/scratch/anirudh.palutla/clusters'

    dloader = DataLoader(root_dir)
    subjects_list = dloader.subjects_list
    sk_idx = np.load('skull_indices.npy')

#     for sub_idx in range(0, len(dloader.subjects_list)):
# #     sub_idx = 40
#         sub = dloader.subjects_list[sub_idx]
        
#         n_clusters, t_clusters = process_subject(sub)
    print("Starting jobs")
    results = Parallel(n_jobs=20)(
        delayed(process_subject)(sub) for sub in dloader.subjects_list)
