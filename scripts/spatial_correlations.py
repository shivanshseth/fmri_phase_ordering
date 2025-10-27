import numpy as np
from utils.data_loader import DataLoader
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore
import time
from joblib import delayed, Parallel
import multiprocessing 
from concurrent.futures import ProcessPoolExecutor
import sys
import argparse

NUM_WORKERS = 32
USE_GPU = False

def linear_map(signal):
    if not type(signal) is np.ndarray:
        signal = np.array(signal)
    a, b = max(signal), min(signal)
    return (2*signal-(a+b))/(a - b)

def binarize_lm(signal):
    if not type(signal) is np.ndarray:
        signal = np.array(signal)
    a, b = max(signal), min(signal)
    signal_lm = linear_map(signal)
    signal_bin = np.ones_like(signal_lm)
    signal_bin[signal_lm > 0] = 1
    signal_bin[signal_lm < 0] = -1
    return signal_bin

def compute_correlation_task(args):
    i, grid, max_l, x_len, y_len, z_len = args
    x, y, z = i
    correlations = np.zeros(max_l + 1)
    n_points = np.zeros(max_l + 1)
    for dx in range(-max_l, max_l + 1):
        for dy in range(-max_l, max_l + 1):
            for dz in range(-max_l, max_l + 1):
                #if dx == dy == dz == 0:
                #    continue

                x_neighbor = x + dx
                y_neighbor = y + dy
                z_neighbor = z + dz

                if (0 <= x_neighbor < x_len and
                    0 <= y_neighbor < y_len and
                    0 <= z_neighbor < z_len):

                    j = np.array([x_neighbor, y_neighbor, z_neighbor])
                    distance = int(np.round(np.linalg.norm(i - j)))

                    if 0 <= distance <= max_l:
                        correlations[distance] += grid[x, y, z] * grid[x_neighbor, y_neighbor, z_neighbor]
                        n_points[distance] +=1

    return correlations, n_points

def spatial_correlation(grid, max_l, idx):
    print(f"Max L: {max_l}")
    x_len, y_len, z_len = grid.shape
    correlations = np.zeros(max_l + 1)
    n_points = np.zeros(max_l + 1)
    #pool = multiprocessing.Pool(processes=NUM_WORKERS)
    # Convert the grid to a CuPy array for GPU acceleration

    # Generate all possible i (3D coordinates) tuples
    # i_tuples = [(x, y, z) for x in range(x_len) for y in range(y_len) for z in range(z_len)]
    i_tuples = idx

    # Use a ProcessPoolExecutor to parallelize the computation across multiple CPUs
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = executor.map(compute_correlation_task, [(i, grid, max_l, x_len, y_len, z_len) for i in i_tuples])
    #args_tuples = [(i, grid, max_l, x_len, y_len, z_len) for i in i_tuples]

    #mp_task = lambda i: compute_correlation_task(i, grid, max_l, x_len, y_len, z_len)
    #results = pool.map(mp_task, i_tuples)
    # Combine the results
    for partial_correlations, partial_n_points in results:
        correlations += partial_correlations
        n_points += partial_n_points
    return correlations, n_points


if __name__ == '__main__':
    #multiprocessing.set_start_method('spawn', force=True)
    root_dir = '/scratch/shivansh.seth/adni/preproc'
    SCRATCH_DATA_DIR = '/scratch/shivansh.seth/tmp/'
    PRINT_TIME_INTERVAL = 20
    if not os.path.exists(SCRATCH_DATA_DIR): os.makedirs(SCRATCH_DATA_DIR)
    L = 20
    dloader = DataLoader(root_dir)
    sk = np.load('skull_indices_flat.npy')
    sk_idx = np.load('skull_indices.npy')
    sk_x = slice(min(sk_idx[0]), max(sk_idx[0])+1)
    sk_y = slice(min(sk_idx[1]), max(sk_idx[1])+1)
    sk_z = slice(min(sk_idx[2]), max(sk_idx[2])+1)
    ad = [81, 70, 43, 40, 14]
    cn = [59, 61, 33, 28, 2]
    ad_sc = {}
    for ad_sub_idx, cn_sub_idx in zip(ad, cn): 
        ad_fdata = dloader.get_func(ad_sub_idx).get_fdata()
        cn_fdata = dloader.get_func(cn_sub_idx).get_fdata()
        st = time.time()
        ad_img = np.apply_along_axis(linear_map, 3, ad_fdata)
        cn_img = np.apply_along_axis(linear_map, 3, cn_fdata)
        ad_sub_sc = []
        cn_sub_sc = []
        for T in [5, 10, 30, 60]: 
#         for T in [1]: 
            st = time.time()
            print(f"Subject: AD_{ad_sub_idx}, T: {T} shape: {ad_img.shape}")
            ad_sc, ad_sc_n_points = spatial_correlation(ad_img[:, :, :, T], L, idx=sk_idx.T) 
            ad_norm = ad_sc/(ad_sc_n_points+1e-7)
            ad_sub_sc.append([ad_norm, ad_sc_n_points])
            print(f"Time taken: {time.time()-st}")
            
            print(f"Subject: CN_{cn_sub_idx}, T: {T} shape: {cn_img.shape}")
            cn_sc, cn_sc_n_points = spatial_correlation(cn_img[:, :, :, T], L, idx=sk_idx.T) 
            cn_norm = cn_sc/(cn_sc_n_points+1e-7)
            cn_sub_sc.append([cn_norm, cn_sc_n_points])
            print(f"Time taken: {time.time()-st}")
        ad_sub_sc = np.array(ad_sub_sc)
        cn_sub_sc = np.array(cn_sub_sc)
        np.save(f'results/ad_sc_{ad_sub_idx}', ad_sub_sc)
        np.save(f'results/cn_sc_{cn_sub_idx}', cn_sub_sc)