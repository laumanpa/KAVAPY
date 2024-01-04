#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:01:10 2023

@author: patrick
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import sliding_window_view

def estimate_mode(values, plot_flag=False, allpeaks=False, out_flag=False):
    """
    Estimate the mode (most frequent value) of a dataset using kernel density estimation.

    Parameters:
    - values (array-like): Input data.
    - plot_flag (bool): If True, plot the kernel density estimate.
    - allpeaks (bool): If True, identify all peaks in the density estimate.
    - out_flag (bool): If True, return an array containing x values and density estimates.

    Returns:
    - most_frequent_value (float or array): Mode(s) of the dataset.
    """
    if len(values) == 0:
        return np.nan
    values = np.array(values)
    
    dimflag = values.ndim == 1
    print(dimflag)
    
    if dimflag:
        num = 1
        dim = 0
    else:
        num = np.shape(values)[0]
        dim = 1
    
    most_frequent_value = np.zeros(num)
    
    for i in range(0,num):
        
        if dimflag:
            values_temp = values
        else:
            values_temp = values[i,:]
        
        # Remove nan vallues
        values_temp = values_temp[~np.isnan(values_temp)]
    
        # Compute kernel density estimate
        kde = gaussian_kde(values_temp)
        xi = np.linspace(min(values_temp), max(values_temp), 1000)
        f = kde(xi)
    
        if not allpeaks:
            # Find the peak of the density estimate
            maxIndex = np.argmax(f)
            most_frequent_value[i] = xi[maxIndex]
        else:
            # Find all peaks
            peak_indices, _ = find_peaks(f, height=0.01)
            most_frequent_value = xi[peak_indices]
    
        if out_flag:
            return np.vstack((xi, f))
        else:
            if plot_flag:
                # Plotting
                plt.figure(figsize=(8, 5))
                plt.plot(xi, f, 'k', label='Kernel Density Estimate')
                
                if allpeaks:
                    plt.plot(xi[peak_indices], f[peak_indices], 'ro', label='All Peaks')
                else:
                    plt.plot(xi[maxIndex], f[maxIndex], 'ro', label='Most Frequent Value')
    
                plt.title("Kernel Density Estimate")
                plt.xlabel("Values")
                plt.legend()
                plt.show()

    return most_frequent_value

def rolling_stats(x, func, window=3):
    """
    Function, for calculation rolling statistics of a timeseries

    Parameters
    ----------
    x : array
        Timeseries.
    func : function
        The corresponding statistical function (e.g. np.mean).
    window : int, optional
        Window size of the rolling window. The default is 3.

    Returns
    -------
    array
        rolling statistics timeseries.

    """
    return func(sliding_window_view(x, window),-1)

values = [1,1,1]
values = np.array(values)
print(values.ndim)
estimate_mode(values)
rs = rolling_stats(np.array(values),estimate_mode)
print(rs)