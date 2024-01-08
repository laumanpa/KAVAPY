import numpy as np
import matplotlib.pyplot as plt
from load_data import generate_datetime_array, cut_data
import config
from array_analysis import pick_events
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def overview(data, t1, BAZ, v_app, rmse, corr, std, max_amp, sr):
    """
    Function, to visualize the final results of the processing

    Parameters
    ----------
    data : array
        Array, containing all traces
    t1 : array
        time array
    BAZ : array
        Array of backazimuth values
    v_app : array
        Array of apparent velocities
    rmse : array
        Array of root mean square errors
    corr : array
        Array of cross correlation data
    std : array
        Array of standard deviation
    max_amp : array
        array of maximum amplitude
    sr : float
        Sampling rate

    Returns
    -------
    None.

    """
    # Calculate sampling rate of backazimuth
    duration = config.duration_minutes*60
    sr2 = len(BAZ)/duration
    # Generate time array for backazimuth
    t2 = generate_datetime_array(config.date, config.year, int(24*60*60*sr2))
    [_, t2] = cut_data(data[:,0:len(t2)], t2, config.start_time, config.duration_minutes, sr2)
    # Choose trace to plot
    trace = data[1,:]
    # Start plotting
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8,8))
    # Plot trace
    plt.subplot(6,1,1)
    plt.plot(t1, trace, 'k')
    # Plot backazimuth
    plt.subplot(6,1,2)
    BAZ2 = np.rad2deg(BAZ)
    BAZ2[np.isnan(BAZ2)] = 300
    plt.scatter(t2, BAZ2, 50, rmse)
    plt.ylim(-180,180)
    # plt.colorbar()
    plt.ylabel("$\\beta [°]$")
    # Plot spans
    spans, modes = pick_events(BAZ)
    # Iterate over the span indices and plot the spans as shaded areas
    for i, (start, end) in enumerate(spans):
        if (end - start) > config.min_length:
            # Fill the area between -1 and 1 with color
            axes[1].fill_betweenx([-180,180], t2[start], t2[end], alpha=0.3, label='Span Area', color='gray')
            plt.text(t2[start],210,str(int(np.rad2deg(modes[i]))))
    # Plot apparent velocity
    plt.subplot(6,1,3)
    v_app[np.isnan(v_app)] = 50
    plt.scatter(t2, v_app, 50, rmse)
    plt.ylabel("$v_{app} [\\frac{km}{s}]$")
    plt.ylim(0,20)
    # Plot Cross correlation
    plt.subplot(6,1,4)
    plt.plot(t2, corr, 'k')
    plt.ylabel("Cross Correlation")
    # Plot a horizontal line at y = 0
    axes[3].axhline(y=config.c_median_threshold, color='red', linestyle='--', label='Horizontal Line')
    # Plot standard deviation of the backazimuth values
    plt.subplot(6,1,5)
    plt.plot(t2, std, 'k')
    plt.ylabel("$\sigma$")
    # Plot a horizontal line at y = 0
    axes[4].axhline(y=config.std_threshold, color='red', linestyle='--', label='Horizontal Line')
    # Plot maximum amplitude
    plt.subplot(6,1,6)
    plt.plot(t2, max_amp,'k')
    plt.ylabel("Maximum amplitude")
    plt.xlabel("time")
    # Plot a horizontal line at y = 0
    axes[5].axhline(y=config.amp_threshold, color='red', linestyle='--', label='Horizontal Line')
    # Set layout
    fig.tight_layout()
    fig.align_ylabels()
    # Customize the x-axis date format
    date_format = mdates.DateFormatter('%H:%M:%S')  # Change the format as needed
    axes[5].xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45) 
    # Deactivate x-axis labels (hide them)
    axes[0].set_xticklabels([]) 
    axes[1].set_xticklabels([]) 
    axes[2].set_xticklabels([]) 
    axes[3].set_xticklabels([]) 
    axes[4].set_xticklabels([]) 
    # Show the plot
    plt.show()

def plot_traces(data, t, v_app, sr):
    """"
    Plot all traces normalized in one figure sorted, after their theoretical arrival time with a linear theoretical line representing arrival times for apparent velocity
    """
    pass