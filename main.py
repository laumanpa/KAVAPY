import config
import load_data as ld
import array_analysis as aa
import numpy as np
import visualize as vis
from processing import processing
from time import time

if __name__=="__main__":

    start_time = time()

    # load parameters
    param = ld.get_parameters(config.folder_program, config.parameter_file)

    # loading data
    [data, t, param] = ld.get_data(config.folder_data, config.freqmin, config.freqmax, param)

    # Calculate distance matrices
    dist_x, dist_y = aa.calc_dist(param["xm"], param["ym"], param["zm"])

    # processing 
    [median_c, max_d, rmse, v_app, BAZ] = processing(data, config.processing_window, dist_x, dist_y, config.corr_res, param["sr"])

    # Filter BAZ values
    BAZ_new = aa.rolling_stats(BAZ, aa.estimate_mode, window=config.mode_window) # rolling mode estimation
    v_app_new = aa.rolling_stats(v_app, aa.estimate_mode, window=config.mode_window) # rolling mode estimation
    # BAZ_new = BAZ.copy()
    # v_app_new = v_app.copy()
    std = aa.rolling_stats(BAZ_new, np.std, window=config.std_window) # rolling standard deviation
    std[np.isnan(std)] = 100
    BAZ_new[std > config.std_threshold] = np.nan
    v_app_new[std > config.std_threshold] = np.nan
    BAZ_new[v_app_new < config.v_app_min] = np.nan
    BAZ_new[v_app_new > config.v_app_max] = np.nan
    v_app_new[v_app_new < config.v_app_min] = np.nan
    v_app_new[v_app_new > config.v_app_max] = np.nan

    end_time = time()

    print("Program ran for " + str(round(end_time-start_time, 2)) + " seconds")

    # Visualizing results
    vis.overview(data,t, BAZ_new, v_app_new, rmse, median_c, std, max_d, param["sr"])
    


    


