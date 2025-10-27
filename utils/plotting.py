import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting

def autocorr(x):
    xp = x - np.mean(x)
    n = len(x)
    f = np.fft.fft(xp, n*2)
    acf = np.real(np.fft.ifft(f * np.conjugate(f))[:n])
    # acf /= (4*np.var(x))
    acf /= acf[0]
    return acf

def exp_decay(t, tau, A, B=0):
    return A * np.exp(-t/tau) + B

def plot_w_fit(data, tau, A, B):
    plt.plot(autocorr(data))
    t = np.arange(len(data))
#     tau, A, B = roi_rts[roi_idx, sub_idx, 0:3]
    plt.plot(exp_decay(t, tau, A, B))
    return plt

def plot_voxels(sig_rois, shape=(51, 67, 67)):
    fli_to_vox = np.unravel_index(sig_rois, shape=shape)
    bts = np.array(fli_to_vox).T
    plotting.plot_markers([1, ] * len(bts), bts)
#     plotting.plot_markers([1, ], [ (0, 0, 0) ])

def optimal_bin_size(data):
    # Calculate mean and variance of count per bin for different bin sizes
    k = []
    v = []
    for i in range(1, len(data)//10):
        hist, edges = np.histogram(data, bins=i)
        mean_count = np.mean(hist)
        var_count = np.var(hist)
        k.append(mean_count)
        v.append(var_count)

    # Calculate the cost function C for different bin sizes
    c = []
    for i in range(1, len(k)):
        delta = (np.max(data) - np.min(data)) / i
        c_i = (2 * k[i] - v[i]) / delta ** 2
        c.append(c_i)

    # Find the bin size that minimizes the cost function C
    opt_bin_size = np.argmin(c) + 1
    return opt_bin_size
