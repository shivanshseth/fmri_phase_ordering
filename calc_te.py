import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
import time
from nilearn import datasets
from nilearn.image import load_img, index_img
from nilearn.plotting import plot_epi
from nilearn.maskers import NiftiLabelsMasker
from PyIF import te_compute as te
from joblib import delayed, Parallel
import pandas as pd
from scipy.stats import zscore
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import glob
from utils.data_loader import DataLoader
from utils.plotting import plot_voxels, plot_w_fit
# from utils.relaxation_time import RelaxationTime
import seaborn as sns
import nibabel as nib
from nilearn import image, plotting, datasets
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def calc_te(timeseries, sig_rois, sk=None):
    stic = time.time()
    if sk is None:
        sk = range(timeseries.shape)
    te_map = []
    for i in sig_rois:
        istic = time.time()
        
        res = Parallel(n_jobs=36)(
            delayed(te.te_compute)(timeseries[i], timeseries[j], embedding=1) 
            for j in sk
        )
        print(f"Ran {i} in {time.time()-istic} seconds")
        te_map.append(res)
        
    print(f"Finished calc in {time.time()-stic} seconds")    
    return np.array(te_map)

def calc_te_t(timeseries, sig_rois, sk=None):
    stic = time.time()
    if sk is None:
        sk = range(timeseries.shape)
    te_map = []
    for i in sig_rois:
        istic = time.time()
        
        res = Parallel(n_jobs=36)(
            delayed(te.te_compute)(timeseries[j], timeseries[i], embedding=1) 
            for j in sk
        )
        print(f"Ran {i} in {time.time()-istic} seconds")
        te_map.append(res)
        
    print(f"Finished calc in {time.time()-stic} seconds")    
    return np.array(te_map)

if __name__ == "__main__":
    # For func output
    root_dir = '/scratch/shivansh.seth/adni/preproc'
    dloader = DataLoader(root_dir)

    print("Loaded " + str(len(dloader.subjects_list)) + " subjects")
    ### Running TE calc
    sr_fp = "/home/shivansh.seth/phase_diagram_analysis/results/significant_regions_ss.npy"
    sr = np.load(sr_fp)
    print("Significant regions:")
    print(sr)
    sk = np.load('skull_indices_flat.npy')

    ### Loading typical subjects

    typical_subs_fp = "/home/shivansh.seth/phase_diagram_analysis/results/typical_subjects.npz"
    with open(typical_subs_fp, 'rb') as f:
        typical_subs = np.load(f)
        ad_sub_idx = typical_subs['ad_sub_idx'][:10]
        cn_sub_idx = typical_subs['cn_sub_idx'][:10]
    for ad in ad_sub_idx:
        print('AD sub:', ad)
        ad_ts = dloader.get_func(dloader.subjects_list[ad]).get_fdata()
        ad_ts = ad_ts.reshape((-1, ad_ts.shape[-1]))
        trunc_lim = 140
        ad_ts = ad_ts[:, :trunc_lim]
        print(ad_ts.shape)
        ad_te = calc_te(ad_ts, sr, sk)
        np.save(f"/home/shivansh.seth/phase_diagram_analysis/results/ad_te_{ad}.npy", ad_te)
    for ad in cn_sub_idx:
        print('CN sub:', ad)
        ad_ts = dloader.get_func(dloader.subjects_list[ad]).get_fdata()
        ad_ts = ad_ts.reshape((-1, ad_ts.shape[-1]))
        trunc_lim = 140
        ad_ts = ad_ts[:, :trunc_lim]
        print(ad_ts.shape)
        ad_te = calc_te(ad_ts, sr, sk)
        np.save(f"/home/shivansh.seth/phase_diagram_analysis/results/cn_te_{ad}.npy", ad_te)

