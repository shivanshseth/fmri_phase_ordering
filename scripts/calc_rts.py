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
from joblib import delayed, Parallel
import pandas as pd
from scipy.stats import zscore
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

import sys 

project_root = os.getcwd()
print(project_root)

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 gives it higher priority.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_loader import DataLoader

root_dir = '/scratch/shivansh.seth/adni/preproc'
SCRATCH_DATA_DIR = '/scratch/shivansh.seth/tmp/'
PRINT_TIME_INTERVAL = 20
if not os.path.exists(SCRATCH_DATA_DIR): os.makedirs(SCRATCH_DATA_DIR)
    
def autocorr(x):
    xp = x - np.mean(x)
    n = len(x)
    f = np.fft.fft(xp, n*2)
    acf = np.real(np.fft.ifft(f * np.conjugate(f))[:n])
    # acf /= (4*np.var(x))
    acf /= acf[0]
    return acf

def plot_w_fit(data, tau, A, B):
    plt.plot(autocorr(data))
    t = np.arange(len(data))
#     tau, A, B = roi_rts[roi_idx, sub_idx, 0:3]
    plt.plot(exp_decay(t, tau, A, B))
    return

def exp_decay(t, tau, A, B=0):
    return A * np.exp(-t/tau) + B

def st_exp_decay(t, tau, A, beta, B=0):
    return A * np.exp(-np.power(t/tau, beta)) + B


# def relaxation_time_og(x, t, t_max=None):
#     if t_max is None:
#         t_max = len(t) // 5
#     p0 = [t[t_max], ]
#     B = np.mean(x[50:])
#     A = 1 - B
#     func = lambda t, tau: exp_decay(t, tau, A=A, B=B)
#     popt, pcov = curve_fit(func, t[:t_max], x[:t_max], p0=p0)
#     se = np.sqrt(np.mean(np.diag(pcov)))
#     t = np.array(t)
# #     print(t.shape)
#     res = x - func(t, *popt)
#     rmse = np.sqrt(np.mean(res**2))
#     print
#     tau = popt[0]
#     return tau, A, B, rmse, se

def relaxation_time(x, t, t_max=None):
    if t_max is None:
        t_max = len(t) // 2
    t = np.array(t)

    p0 = [t[t_max], 1]
    bounds = [[0, 0], [np.inf, 1.0]]
    B = np.mean(x[t_max//2: t_max])
    A = 1 - B
    func = lambda t, tau: exp_decay(t, tau, A=A, B=B)
    try:
        popt, pcov = curve_fit(func, t[:t_max], x[:t_max], p0=p0[:1], bounds=[[0], [np.inf]])
        tau_exp = popt[0]
        res = x - func(t, *popt)
        rmse_exp = np.sqrt(np.mean(res**2))
    except:
        tau_exp = np.nan
        rmse_exp = np.nan
    func = lambda t, tau, beta: st_exp_decay(t, tau, beta=beta, A=A, B=B)
    try: 
        popt, pcov = curve_fit(func, t[:t_max], x[:t_max], p0=p0, bounds=bounds)
        se = np.sqrt(np.mean(np.diag(pcov)))
        res = x - func(t, *popt)
        rmse = np.sqrt(np.mean(res**2))
        tau = popt[0]
        beta = popt[1]
    except:
        tau, beta, rmse = np.nan, np.nan, np.nan
    return tau, beta, tau_exp, A, B, rmse, rmse_exp

def append_res_to_csv(csv_save_path, data):
    df = pd.read_csv(csv_save_path)
    df = pd.concat([ df, pd.DataFrame([data,], columns=df.columns) ])
    df.to_csv(csv_save_path, index=False)
    return

# def get_ts_data(dloader):
#     ts = dloader.get_func(0).get_fdata()
#     trunc_lim = 140
#     n_rois = 116

#     ts_data = np.memmap(os.path.join(SCRATCH_DATA_DIR, 'ts_data.dat'), dtype='float32', mode='w+', shape=(len(subjects_list), *ts.shape[:3], trunc_lim))

#     stic = time.time()
#     for i, sub in enumerate(dloader.subjects_list):
#         tic = time.time()

#         ts = dloader.get_func(sub).get_fdata()
#         ts_data[i] = ts[:, :, :, :trunc_lim]

#         print(f"{i}:\tLoaded sub {sub}; took {time.time()-tic} s; {time.time()-stic} s elapsed")

#     return ts_data

def calc_func_rt_for_sub(sub_ind, dloader, roi_rts, rts_csv_save_dir, rts_save_path, sk_ind, do_save=False):
    print(f"{sub_ind}: Starting calculation for sub {sub_ind}")
    subtic = time.time()
    lasttic = time.time()
#     print(lasttic)
    
    sub_data = dloader.get_func(sub_ind).get_fdata()
    rts = np.zeros(shape=(*sub_data.shape[:-1], 7))
    rts.fill(np.nan)
    csv_save_path = os.path.join(rts_csv_save_dir, f"rts_results_{sub_ind}_{dloader.subjects_list[sub_ind]}.npy")
    txt_save_path = os.path.join(rts_csv_save_dir, f"rts_results_{sub_ind}_{dloader.subjects_list[sub_ind]}.txt")
    
    if os.path.exists(csv_save_path):
        return
    #rts_df = pd.DataFrame(columns=['sub_ind', 'sub_id', 'x', 'y', 'z', 'tau', 'A', 'B', 'rmse', 'se'])
    #rts_df.to_csv(csv_save_path, index=False)
    
#     for xind in range(sub_data.shape[0]):
#         print(f"{sub_ind}: On ({xind} ); {time.time()-subtic} s elapsed")
#         for yind in range(sub_data.shape[1]):
#             for zind in range(sub_data.shape[2]):
# #                 if not (xind, yind, zind) in sk_ind: 
# #                     rts[xind, yind, zind] = np.nan
# #                     continue
#                 xyztic = time.time()
#                 ts = sub_data[xind, yind, zind, :]
#                 signal = autocorr(ts)
#                 rt = relaxation_time(signal, list(range(len(signal))))
#                 #roi_rts[sub_ind, xind, yind, zind] = rt
#                 rts[xind, yind, zind] = rt
#                 if rt[0] == np.nan and rt[2] == np.nan:
#                     #rt = np.array([ np.nan,]*7)
#                     #roi_rts[sub_ind, xind, yind, zind] = rt
#                     #rts[xind, yind, zind] = rt
#                     print(f"Unable to fit curve for: subject {sub_ind} ({dloader.subjects_list[sub_ind]}) - ( {xind}, {yind}, {zind} )")
                
                    
# #                if time.time() - lasttic > PRINT_TIME_INTERVAL:
# #                    lasttic = time.time()
                    
# #                 print(f"{sub_ind}: ({xind}, {yind}, {zind}); took {time.time()-xyztic} s")
# #                 if do_save:
# #                     append_res_to_csv(csv_save_path, 
# #                                       [ sub_ind, dloader.subjects_list[sub_ind], xind, yind, zind, *rt ])
    for (xind, yind, zind) in sk_ind:
        xyztic = time.time()
        ts = sub_data[xind, yind, zind, :]
        signal = autocorr(ts)
        rt = relaxation_time(signal, list(range(len(signal))))
        #roi_rts[sub_ind, xind, yind, zind] = rt
        rts[xind, yind, zind] = rt
        if rt[0] == np.nan and rt[2] == np.nan:
            print(f"Unable to fit curve for: subject {sub_ind} ({dloader.subjects_list[sub_ind]}) - ( {xind}, {yind}, {zind} )")

    if do_save:
        np.save(csv_save_path, rts)
    del sub_data
    #np.save(rts_save_path, roi_rts)
    print(f"{sub_ind}: Finished {sub_ind}; took {time.time()-subtic} s;\n")
    return

if __name__ == '__main__':
    dloader = DataLoader(root_dir)
    subjects_list = dloader.subjects_list
#     ts_data = get_ts_data(dloader)
    
    sk_ind = np.load('skull_indices.npy').T
    #rts_df = pd.DataFrame(columns=['sub_ind', 'sub_id', 'x', 'y', 'z', 'tau', 'A', 'B', 'rmse', 'se'])
    rts_csv_save_dir = './rts_results_both_new'
    rts_npy_save_path = './roi_rts_both.npy'
    do_save = True
    if not os.path.exists(rts_csv_save_dir):
        os.makedirs(rts_csv_save_dir)
        
#     For func data
    roi_rts = np.memmap(
        os.path.join(SCRATCH_DATA_DIR, 'roi_rts_both.dat'), 
        dtype='float32', 
        mode='w+', 
        shape=(len(dloader.subjects_list), *dloader.get_func(0).shape[:-1], 5)
    )
    sub_start, sub_end = 0, len(dloader.subjects_list)
#     sub_start, sub_end = 0, 2
    results = Parallel(n_jobs=10)(delayed(calc_func_rt_for_sub)(i, dloader, roi_rts, rts_csv_save_dir, rts_npy_save_path, sk_ind, do_save) for i in range(sub_start, sub_end))
#     results = calc_func_rt_for_sub(0, dloader, roi_rts, rts_csv_save_dir, rts_npy_save_path, do_save)
    
    
    pass
