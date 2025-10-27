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

def fl_to_vox(idx, shape=(51, 67, 67)):
    return np.unravel_index(idx, shape=shape)

class DataLoader:
    def __init__(self, root_dir, z_sk_dir=None, subjects_list=False):
        '''
        Note that `root_dir` should only contain folders with subject names and should
        follow the structure `root_dir/<subject_dir_prefix+subject_name/modality_filename`
        '''
        
        self.func_name = 'fwhm-5_sfunc.nii'
        self.anat_name = 'anat_brain.nii.gz'
        self.z_sk_func_name = '{}_z-sk-ts.npy'
        self.z_sk_dir = z_sk_dir
        self.z_name = 'z_func.nii.gz'
        self.subject_dir_prefix = 'sub-'
        self.SKULL_INDICES_FP = "/home/shivansh.seth/phase_diagram_analysis/skull_indices.npy"
        
        self.z_region_suffix = '_z_timeseries.txt'
        
        self.cn_names_fp = '/home/shivansh.seth/phase_diagram_analysis/cn_subject_names.txt'
        self.ad_names_fp = '/home/shivansh.seth/phase_diagram_analysis/ad_subject_names.txt'
        
        self.cn_subs = [ i.strip() for i in open(self.cn_names_fp).readlines() ]
        self.ad_subs =[ i.strip() for i in open(self.ad_names_fp).readlines() ]
        
        self.root_dir = root_dir
        self.subject_dirs_list = os.listdir(root_dir)
        self.subject_dirs_list.sort()
        if (type(subjects_list) == type(False)) and (subjects_list == False):
            self.subjects_list = self._get_subjects_list()
        else:
            self.subjects_list = subjects_list
        
        self.diag = []
        for sub in self.subjects_list:
            self.diag.append(1 if sub in self.ad_subs else 0)
            
        return
    
    def _get_subjects_list(self):
#         print("RUN")
        subs = []
        for i in self.subject_dirs_list:
            subs.append(i.replace(self.subject_dir_prefix, ''))
        return subs
    
    def get_func(self, subject_name, get_image=True):
        fp = self.get_nii_fp(subject_name, self.func_name)
        
#         print(f"Retrieving from {fp}")
        img = load_img(fp)
        data = img.get_fdata()
        if get_image:
            return img
        return data
    
    def get_skstrip_func_flat(self, subject_name):
        ff = self.get_func(subject_name).get_fdata()
        sk = np.load(self.SKULL_INDICES_FP)
        data = []
        for x, y, z in sk.T:
            data.append(ff[x, y, z])
        data = np.array(data)
        return data
    
    def get_z_sk_func(self, subject_name):
        if type(subject_name) == int:
            subject_name = self.subjects_list[subject_name]
        fname = self.z_sk_func_name.format(subject_name)
        print(fname)
        ts = np.load(os.path.join(self.z_sk_dir, fname))
        return ts
    
    def get_anat(self, subject_name, get_image=True):
        fp = self.get_nii_fp(subject_name, self.anat_name)
        img = load_img(fp)
        data = img.get_fdata()
        if get_image:
            return img
        return data
    
    def get_nii_fp(self, subject_name, modal_name):
        fp = os.path.join(self.root_dir, self.subject_dir_prefix + str(subject_name), modal_name)
        if type(subject_name) == int:
            fp = os.path.join(self.root_dir, self.subject_dirs_list[subject_name], modal_name)
        return fp
    
    def get_z_region_output(self, subject_name):
        fp = os.path.join(self.root_dir, str(subject_name) + self.z_region_suffix)
        if type(subject_name) == int:
            fp = os.path.join(self.root_dir, self.subjects_list[subject_name] + self.z_region_suffix)
        ts = np.loadtxt(fp)
        return ts
    
    pass
