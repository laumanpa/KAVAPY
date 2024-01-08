import numpy as np
from sklearn.linear_model import HuberRegressor
import config
from multiprocessing import Pool
import matplotlib.pyplot as plt

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
    [c, median_c] = cross_corr(d, config.max_delay, sr)

    # Calculate max value
    max_d = np.mean(np.max(d,1))

    if median_c > config.c_median_threshold and max_d > config.amp_threshold:

        # Prepare input arrays for Regression
        delays = 1.0 * c

        # Create a boolean matrix with the independent entries
        independent = np.tri(c.shape[0], dtype=bool, k=-1)

        # Create a condition for the non-NaN delays
        valid_delays = ~np.isnan(delays)

        # Apply the condition to the delays and distances
        tau = delays[valid_delays & independent].T
        A = np.vstack([-dist_x[valid_delays & independent], -dist_y[valid_delays & independent]]).T

        # Start Regression
        mdl = HuberRegressor(epsilon=1).fit(A, tau)
        rmse = np.sqrt(np.mean((mdl.predict(A) - tau) ** 2))
        beta = [mdl.coef_[0], mdl.coef_[1]]
        if rmse < config.rmse_threshold:

            # Calculate apparent velocity and backazimuth
            v_app = np.sqrt(1.0 / (beta[0] ** 2 + beta[1] ** 2))
            BAZ = np.arctan2(beta[0], beta[1])
            if config.fit_plot_flag:
                plot3d_fit(A, tau, mdl, rmse)
            
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

def cross_corr(A, maxlag, sr):
    num = np.size(A, 0)
    delay = np.zeros((num, num))
    c_max = np.zeros((num, num))

    for i in range(num):
        mean_A_i = np.mean(A[i, :])

        for k in range(num):
            mean_A_k = np.mean(A[k, :])

            # Calculate the cross-correlation using numpy's correlate function
            cross_corr_result = np.correlate(A[i, :] - mean_A_i, A[k, :] - mean_A_k, mode='full')

            # Calculate the normalization factors
            normalization = np.sqrt(np.sum((A[i, :] - mean_A_i)**2) * np.sum((A[k, :] - mean_A_k)**2))

            # Calculate the normalized cross-correlation
            normalized_corr = cross_corr_result / normalization

            # Cut the cross-correlation function at the maximum lag
            maxlag_samples = int(maxlag * sr)
            normalized_corr = normalized_corr[len(A[i, :]) - 1 - maxlag_samples:len(A[i, :]) + maxlag_samples]

            c_max[i, k] = np.max(normalized_corr)
            c_ind = np.argmax(normalized_corr) - (len(normalized_corr) - 1) / 2
            delay[i, k] = c_ind / sr

    c_max[c_max == 1] = np.nan
    c_median = np.nanmedian(c_max)

    return delay, c_median


import numpy as np
import matplotlib.pyplot as plt

def plot3d_fit(A, tau, mdl, rmse):
    """
    Plots a 3D fit of the data.

    Parameters:
    - A: numpy array, input data
    - tau: numpy array, target values
    - mdl: model object, trained model
    - rmse: numpy array, root mean squared error

    Returns:
    None
    """
    # Create a new figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Predict the target values using the trained model
    Z = mdl.predict(A)

    # Calculate the root mean squared error
    rmse = np.sqrt((mdl.predict(A) - tau) ** 2)

    # Scatter plot of the input data and target values
    ax.scatter(A[:, 0], A[:, 1], tau, c=rmse, marker='o')

    # Create a grid of points for the surface plot
    xi = np.linspace(np.min(A[:, 0]), np.max(A[:, 0]), 100)
    yi = np.linspace(np.min(A[:, 1]), np.max(A[:, 1]), 100)
    X, Y = np.meshgrid(xi, yi)

    # Initialize the Z values for the surface plot
    Z = np.zeros((len(xi), len(yi)))

    # Calculate the predicted values for each point in the grid
    for i in range(len(xi)):
        for j in range(len(yi)):
            Z[i, j] = mdl.predict(np.array([[xi[i], yi[j]]]))

    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.2)

    # Set labels for the axes
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    rmse = np.sqrt(np.mean(rmse ** 2))
    ax.set_title('RMSE = ' + str(round(rmse, 2)) + ' s/km')

    #Number of files in images folder
    import os
    i = len([name for name in os.listdir('images') if os.path.isfile(os.path.join('images', name))])

    # Save the figure as an image in folder 'images'
    plt.savefig('images/3d_fit_' + str(i) + '.png', dpi=600)

    # Close the figure
    plt.close()
    
