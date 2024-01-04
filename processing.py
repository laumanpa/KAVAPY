import numpy as np
from sklearn.linear_model import HuberRegressor
import config
from multiprocessing import Pool

def processing(data, window, dist_x, dist_y, corr_res, sr):
    """"
    Function for parallel array analysis.
    Input:
        data: traces (array)
        time: time (array)
        window: length of the window in samples (int)
        dist_x: matrix containing all distances between the stations in x direction (array)
        dist_y: matrix containing all distances between the stations in y direction (array)
    Output:
        v_app: apparent velocity (array)
        BAZ: Back azimuth (array) 
        rmse: Root mean square error (array) 
        max_d: Max value of each window (array)
        median_c: the median of the cross correlation matrix (array)
        
    """
    # Preallocating arrays
    step_length = np.arange(0, data.shape[1] - window + 1, corr_res)
    BAZ = np.zeros(len(step_length))
    v_app = np.zeros(len(step_length))
    rmse = np.zeros(len(step_length))
    median_c = np.zeros(len(step_length))
    max_d = np.zeros(len(step_length))
    
    # Creating input list for parallel processing function
    args_list = [(i,data, window, dist_x, dist_y, step_length, sr) for i in range(0,len(step_length))]

    # Parallel processing
    with Pool() as pool:
        results = pool.map(wrapper_function, args_list)

    # Unpack results from parallel processing output
    for i,entry in enumerate(results):
        v_app[i] = entry[0]
        BAZ[i] = entry[1]
        rmse[i] = entry[2]
        max_d[i] = entry[3]
        median_c[i] = entry[4]

    return median_c, max_d, rmse, v_app, BAZ

def processing_parallel(args):
    """"
    Main parallel processing function
    """
    i, data, window, dist_x, dist_y, step_length, sr = args

    # Cut data and time to window size
    d = data[:,step_length[i]:step_length[i]+window]

    # Calculate cross correlation
    [c, median_c] = cross_corr(d,sr)

    # Calculate max value
    max_d = np.mean(np.max(d,1))

    if median_c > config.c_median_threshold and max_d > config.amp_threshold:

        # Prepare input arrays for Regression
        delays = -1.0 * c
        
        st_used_dum = np.ones(data.shape[0]).astype(bool)
        
        remdiag = np.ones(delays.shape).astype(bool)
        tau = delays[
            np.logical_and(remdiag & ~np.isnan(delays) & np.tile(st_used_dum, (d.shape[0], 1)),
                            delays != 0)].T

        A = np.vstack(
            [-dist_x[np.logical_and(remdiag & ~np.isnan(delays) & np.tile(st_used_dum, (d.shape[0], 1)),
                                            delays != 0)],
                -dist_y[np.logical_and(remdiag & ~np.isnan(delays) & np.tile(st_used_dum, (d.shape[0], 1)),
                                            delays != 0)]]).T
        
        # Start Regression
        mdl = HuberRegressor(epsilon=1.2, fit_intercept=False).fit(A, tau)
        rmse = np.sqrt(np.mean((mdl.predict(A) - tau) ** 2))
        beta = [mdl.coef_[0], mdl.coef_[1]]
        if rmse < config.rmse_threshold:

            # Calculate apparent velocity and backazimuth
            v_app = np.sqrt(1.0 / (beta[0] ** 2 + beta[1] ** 2))
            BAZ = np.arctan2(beta[1], beta[0])
            
        else:
            v_app = np.nan
            BAZ = np.nan
    else:
        v_app = np.nan
        BAZ = np.nan
        rmse = np.nan
    return v_app, BAZ, rmse, max_d, median_c

def wrapper_function(args):
    return processing_parallel(args)

def cross_corr(A, sr):
    num = np.size(A,0)
    delay = np.zeros((num,num))
    c_max = np.zeros((num,num))
    for i in range(0,num):
        for k in range(0,num):
            a = (A[i,:] - np.mean(A[i,:])) / (np.std(A[i,:]) * len(A[i,:]))
            b = (A[k,:] - np.mean(A[k,:])) / (np.std(A[k,:]))
            c = np.correlate(a, b, mode='full')
            c_max[i,k] = np.max(c)
            c_ind = np.argmax(c) - (len(a)-1)
            delay[i,k] = c_ind/sr
    c_max[c_max==1] = np.nan
    c_median = np.nanmedian(c_max)

    return delay, c_median

