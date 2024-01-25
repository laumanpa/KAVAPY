import numpy as np
import matplotlib.pyplot as plt
from load_data import generate_datetime_array, cut_data
import config
from array_analysis import pick_events
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import cmcrameri.cm as cmc
import os
import matplotlib.cm as cm
from matplotlib.pyplot import figure, show, rc

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
    plt.ylabel("$\\beta [°]$")
    
    # Calculate spans
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
    plt.savefig('images/overview.png', dpi=600)


def plot_traces(data, t, delay, v_app, sr):
    """"
    Plot all traces normalized in one figure sorted, after their theoretical arrival time with a linear theoretical line representing arrival times for apparent velocity
    """
    Ind = np.argsort(delay)
    delay = delay[Ind]
    delay = delay / np.max(delay)
    data = data[Ind,:]
    # Plot all traces normalized
    plt.figure()
    for i in range(len(delay)):
        plt.plot(t, data[i,:]/np.max(data[i,:]) + delay[i])

    # Plot theoretical arrival times
    t_app = np.linspace(0, np.max(t), 1000)
    for i in range(len(delay)):
        plt.plot(t_app,  t_app*v_app/np.max(t_app), 'k--')

def plot3d_fit(A, tau, mdl, rmse ,BAZ, num):
    """
    Plots a 3D fit of the data.

    Parameters:
    - A: numpy array, input data
    - tau: numpy array, target values
    - mdl: model object, trained model
    - rmse: numpy array, root mean squared error
    - BAZ: float, backazimuth angle in radians

    Returns:
    None
    """
    # Create a new figure and 3D axes
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(121, projection='3d')

    # Use the model to predict the target values
    Z = mdl.predict(A)

    # Calculate the root mean squared error
    rmse = np.sqrt((Z - tau) ** 2)

    # Create a scatter plot of the input data and target values, color-coded by RMSE
    ax.scatter(A[:, 0], A[:, 1], tau, c=rmse, marker='o', s=75, cmap=cmc.batlow)

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
    ax.set_xlabel('x [km]', fontsize=18, labelpad=15)
    ax.set_ylabel('y [km]', fontsize=18, labelpad=15)
    ax.set_zlabel('z [km]', fontsize=18, labelpad=15)

    # Calculate total RMSE and set it as the title of the plot
    rmse_total = np.sqrt(np.mean((tau - mdl.predict(A))**2))
    ax.set_title('RMSE = ' + str(round(rmse_total, 2)) + ' s/km', fontsize=22)

    # Set the size of the tick labels
    ax.tick_params(axis='both', labelsize=16)

    # Create a subplot for the actual vs predicted values
    ax2 = fig.add_subplot(122)

    # Plot the actual and predicted values
    ax2.plot(tau, linewidth=2, color='lightseagreen')
    ax2.plot(mdl.predict(A), linewidth=3, color='darkslategray')

    # Set labels for the axes
    ax2.set_xlabel('Measurement number', fontsize=18)
    ax2.set_ylabel(r'$\tau [s]$', fontsize=18)

    # Add a legend
    plt.legend(["Measurement", "Prediction"], fontsize=18)

    # Set the title of the plot to the backazimuth angle
    ax2.set_title(r'$\beta = {}°$'.format(round(np.rad2deg(BAZ), 3)), fontsize=22)

    # Set the size of the tick labels
    ax2.tick_params(axis='both', labelsize=16)

    # Adjust the layout to make sure everything fits
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4) 

    # Count the number of files in the 'images' directory
    i = len([name for name in os.listdir('images') if os.path.isfile(os.path.join('images', name))])

    # Save the figure as an image in the 'images' directory
    plt.savefig('images/' + str(num) +'_3d_fit.png', dpi=600)

    # Close the figure to free up memory
    plt.close()
from matplotlib.collections import LineCollection
def plot_shift(d, delay, dist_x, dist_y, v_app, BAZ, sr, rmse, stations, num):
    """
    Plot the differnt traces in one plot, ordered by their delay time
    Input:
    - d: numpy array, input data
    - delay: numpy array, delay times
    """

    delay = delay / np.max(np.abs(delay)) * 10

    BAZ = -BAZ + np.pi/2

    # Calculate distances in BAZ direction
    dist = dist_x * np.cos(BAZ) + dist_y * np.sin(BAZ)
    # Add -2 to the first element of dist and 2 to the last element
    dist2 = np.insert(dist, 0, -2)
    dist2 = np.append(dist2, 2)
    # Calculate delay times
    theo_delay = dist * sr / v_app
    theo_delay_norm = theo_delay / np.max(np.abs(theo_delay)) * 10

    theo_delay2 = dist2 * sr / v_app
    theo_delay_norm2 = theo_delay2 / np.max(np.abs(theo_delay)) * 10

    theo_line =np.zeros((len(delay),2))

    # Plot all traces normalized
    plt.figure()
    for i in range(len(delay)):
        plt.plot(d[i,:]/np.max(np.abs(d[i,:])) + dist[i] / np.max(np.abs(dist)) * 10)
        # plt.plot(theo_delay[i] + int(len(d[i,:])/2), dist[i] / np.max(np.abs(dist)) * 10, 'rs')
        theo_line[i,:] = [dist[i] / np.max(np.abs(dist)), dist[i] / np.max(np.abs(dist)) * 10]
    # plot gray lines in background
    # for j in range(1,10):
    #     plt.plot(theo_line[:,0] + int(j *len(d[i,:])/10), theo_line[:,1], 'k-', alpha=0.2)
    # Create lines for a grid
    lines = []
    for j in range(-5, 12):
        for k in range(len(theo_delay2)):
            x = theo_delay2[k] + int(j * len(d[1,:]) / 10)
            y = theo_delay_norm2[k]
            if k == 0:  # start of a line
                line = [(x, y)]
            elif k == len(theo_delay2) - 1:  # end of a line
                line.append((x, y))
                lines.append(line)
            else:  # middle of a line
                line.append((x, y))

    # Create a LineCollection from the coordinates
    lc = LineCollection(lines, colors='k', alpha=0.1)

    # Add the LineCollection to the axes
    plt.gca().add_collection(lc)
    
    # Change y span to [min(delay)-1, max(delay)+1]
    # plt.ylim([np.min(delay)-1, np.max(delay)+1])

    # plt.title("RMSE = " + str(round(rmse, 2)) + " s/km")
    plt.title(r'$\beta = {}°, v_{{app}} = {}$ km/s'.format(round(np.rad2deg(np.pi/2-BAZ), 2), round(v_app, 2)))
    
    # Change x ticklabels to time, using the sampling rate
    x_ticks = np.linspace(0, len(d[i,:]), 10)
    x_ticklabels = np.round(x_ticks/sr, 2)
    plt.xticks(x_ticks, x_ticklabels)
    plt.xlabel("time [s]")

    stations = np.array(stations)
    # plt.legend(stations[Ind], fontsize=8)
    plt.yticks(dist / np.max(np.abs(dist)) * 10, stations)

    plt.savefig('images/' + str(num) + '_shift.png', dpi=600)

def rose_plot(BAZ, v_app):
    """
    Plot a rose plot of the backazimuths
    Input:
    - BAZ: numpy array, backazimuths in radians
    - v_app: numpy array, apparent velocities
    """
    # force square figure and square axes looks better for polar, IMO

    BAZ = np.array(BAZ)
    v_app = np.array(v_app)

    # Turn list of arrays into one array
    BAZ = np.concatenate(BAZ)
    # v_app = np.concatenate(v_app)

    BAZ = BAZ[~np.isnan(BAZ)]
    v_app = v_app[~np.isnan(v_app)]

    BAZ = np.deg2rad(BAZ)

    # Calculate the density of the backazimuths
    N = len(BAZ)
    bins = 90
    theta = np.linspace(-np.pi, np.pi, bins, endpoint=False)
    density = np.zeros(bins)
    for i in range(bins-1):
        density[i] = np.sum((BAZ > theta[i]) & (BAZ < theta[i+1]))
    

    fig = figure(figsize=(8,8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    N = 90
    theta = BAZ
    radii = v_app
    width = np.pi/bins
    bars = ax.bar(theta, radii, width=width, bottom=0.0)

    for r,bar in zip(radii, bars):
        theta_bar = np.linspace(-np.pi, np.pi, bins)
        # Get density of backazimuths
        ind = np.argmin(np.abs(theta_bar - bar.xy[0]))
        bar.set_facecolor('black')
        bar.set_alpha(density[ind]/np.max(density)*0.7+0.3)


    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # ax.set_rmax(10)
    # plt.show()
    plt.savefig('images/rose_plot.png', dpi=600)
    plt.close()

if __name__=="__main__":
    BAZ = np.array([np.pi/2, np.pi/2, np.pi/4, -np.pi/7, np.pi/9, np.pi/2, np.nan])
    v_app = np.array([2, 3, 3, 4, 5, 6, np.nan])
    rose_plot(BAZ, v_app)