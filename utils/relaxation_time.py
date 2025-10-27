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
# from PyIF import te_compute as te
from joblib import delayed, Parallel
import pandas as pd
from scipy.stats import zscore
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import glob
# import seaborn as sns
import time
from utils.data_loader import DataLoader
import scienceplots
import warnings
warnings.filterwarnings(action='ignore')
plt.style.use(['science','no-latex', 'ieee'])

class RelaxationTime:
    def __init__(self, root_dir, SCRATCH_DATA_DIR):
        self.dloader = DataLoader(root_dir)
     
    @staticmethod
    def autocorr(x):
        xp = x - np.mean(x)
        n = len(x)
        f = np.fft.fft(xp, n*2)
        acf = np.real(np.fft.ifft(f * np.conjugate(f))[:n])
        # acf /= (4*np.var(x))
        acf /= acf[0]
        return acf
    
    @staticmethod
    def st_exp_decay(t, tau, beta, A, B=0):
        return A * np.exp(-np.power(t/tau, beta)) + B

    @staticmethod
    def exp_decay(t, tau, A, B=0):
        return A * np.exp(-t/tau) + B
   
    @staticmethod
    def get_st_exp_fit(x, t_max=None):
        t = list(range(len(x)))
        if t_max is None:
            t_max = len(t) // 2
        p0 = [t[t_max], 1]
        bounds = [[0, 0], [np.inf, np.inf]]
        B = np.mean(x[50:])
        A = 1 - B
        func = lambda t, tau, beta: RelaxationTime.st_exp_decay(t, tau, beta=beta, A=A, B=B)
        popt, pcov = curve_fit(func, t[:t_max], x[:t_max], p0=p0, bounds=bounds)
        se = np.sqrt(np.mean(np.diag(pcov)))
        t = np.array(t)
    #     print(t.shape)
        res = x - func(t, *popt)
        rmse = np.sqrt(np.mean(res**2))
        tau = popt[0]
        beta = popt[1]
        return tau, beta, A, B, rmse, se
    @staticmethod
    def get_exp_fit(x, t_max=None):
        t = list(range(len(x)))                                            
        if t_max is None:
            t_max = len(t) // 2
        p0 = [t[t_max]]
        bounds = [[0], [np.inf]]
        B = np.mean(x[50:])
        A = 1 - B
        func = lambda t, tau: RelaxationTime.exp_decay(t, tau, A=A, B=B)
        popt, pcov = curve_fit(func, t[:t_max], x[:t_max], p0=p0, bounds=bounds)
        se = np.sqrt(np.mean(np.diag(pcov)))
        t = np.array(t)
    #     print(t.shape)
        res = x - func(t, *popt)
        rmse = np.sqrt(np.mean(res**2))
        tau = popt[0]
        return tau, A, B, rmse, se
    
    
    def plot_w_fit(self, sub_idx, fl_idx):
        func_data = self.dloader.get_func(sub_idx, get_image=False)
        vox_idx = np.unravel_index(fl_idx, shape=func_data.shape[:-1])
        data = func_data[vox_idx]
        t = np.arange(len(data))
        signal = self.autocorr(data)
        plt.plot(signal)
        st_fit = self.get_st_exp_fit(signal)
        plt.plot(self.st_exp_decay(t, *st_fit[:-2]), label=f'\\tau_\{st-exp\}={st_fit[0]:.2f}, \beta={st_fit[1]:.2f}')
        try: 
            exp_fit = self.get_exp_fit(signal)
            plt.plot(self.exp_decay(t, *exp_fit[:-2]), label=f'$\\tau_\{exp\}={exp_fit[0]:.2f}$')
        except:
            print('Exp fit failed')
        plt.legend()
        return plt
    
    def plot_w_fit_log(self, sub_idx, fl_idx):
        func_data = self.dloader.get_func(sub_idx, get_image=False)
        vox_idx = np.unravel_index(fl_idx, shape=func_data.shape[:-1])
        data = func_data[vox_idx]
        t = np.arange(len(data))
        signal = self.autocorr(data)
        plt.plot(np.log(signal[signal>0]))
        st_fit = self.get_st_exp_fit(signal)
        exp_fit = self.get_exp_fit(signal)
        plt.plot(np.log(self.st_exp_decay(t, *st_fit[:-2])), label=f'St exp tau={st_fit[0]:.2f}, beta={st_fit[1]:.2f}, rmse={st_fit[4]:.2f}')
        plt.plot(np.log(self.exp_decay(t, *exp_fit[:-2])), label=f'Exp tau={exp_fit[0]:.2f}, rmse={exp_fit[3]:.2f}')
        plt.legend()
        return plt