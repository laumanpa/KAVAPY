from GPS_utils import GPS_utils
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Converting Geodetic coordinates to ENU (East North Up) Coordinate System (Local coordinate system)
    """
    Converter = GPS_utils()

    # Set ENU origin
    Converter.setENUorigin(ref_lat, ref_lon, ref_alt)

    #Preallocate arrays
    x = np.array(lat.copy())
    y = np.array(lat.copy())
    z = np.array(alt.copy())
    
    # Convert coordinates
    for i in range(0,len(lat)):
        [x[i], y[i], z[i]] = Converter.geo2enu(lat[i], lon[i], alt[i])

    return x, y, z

def calc_dist(xm,ym,zm):
    """
    Calculate Distance between Stations in x and y direction
    Input:
        xm: Latitude (Array)
        yx: Longitude(Array)
        zm: Altitude (Array)
    Output:
        dist_x: Distance between Stations in x direction
        dist_y: Distance between Stations in y direction
    """
    # Transform geodetic to local coordinates
    [x, y, _] = geodetic_to_enu(xm, ym, zm, xm[0], ym[0], zm[0])
    
    dist_x = np.zeros([len(x),len(x)])
    dist_y = np.zeros([len(y),len(y)])
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            dist_x[j, i] = (x[j]-x[i])/1000
            dist_y[j, i] = (y[j]-y[i])/1000
    
    return dist_x, dist_y


def max_rolling1(a, window,axis =1):
    """""
    Function for computing the rolling maximum
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.max(rolling,axis=axis)

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

    # Padding array, so result has same length as input
    x = np.pad(x,(0, window-1),'edge')

    return func(sliding_window_view(x, window),-1)



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
    
    if ~isinstance(plot_flag,bool):
        plot_flag = False
    
    if len(values) == 0:
        return np.nan
    values = np.array(values)
    
    dimflag = values.ndim == 1
    
    if dimflag:
        num = 1
    else:
        num = np.shape(values)[0]
    
    most_frequent_value = np.zeros(num)
    
    for i in range(0, num):
        
        if dimflag:
            values_temp = values
        else:
            values_temp = values[i,:]
        
        # Remove nan values
        values_temp = values_temp[~np.isnan(values_temp)]

        # Check if all values are nan
        if len(values_temp)==0:
            most_frequent_value[i] = np.nan
            continue
        elif len(values_temp)==1:
            most_frequent_value[i] = values_temp
            continue

        # Check, if all entries are the same
        if np.all(values_temp == values_temp[0]):
            most_frequent_value[i] = values_temp[0]
            continue
            
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

def pick_events(BAZ):
    """"
    Function, for finding the indices of the start and end points of time spans, where values exist
    Input:
        BAZ: Array of backazimuths, with nans in between the spans (array)
    Output:
        span_indices: List of tuples with start and end index of the span (list)
    """
    BAZ = np.insert(BAZ, 0, np.nan)
    BAZ[-1] = np.nan
    borders = np.isnan(BAZ).astype(int)
    borders = np.diff(borders)
    # Find the indices of -1 and 1
    indices_minus_one = np.where(borders == -1)[0]
    indices_one = np.where(borders == 1)[0]
    # Initialize an empty list to store the start and end indices of spans
    span_indices = []
    BAZ_mode= []

    # Iterate over the indices of -1 and find the corresponding start and end indices
    for index_minus_one in indices_minus_one:
        # Find the index of the next 1 after the current -1
        next_one_after_minus_one = indices_one[indices_one > index_minus_one][0]
        
        # Append the start and end indices to the list
        span_indices.append((index_minus_one, next_one_after_minus_one))

        # Find mode of span

        mode = estimate_mode(BAZ[index_minus_one:next_one_after_minus_one])
        BAZ_mode.append(mode)

    return span_indices, BAZ_mode

    


if __name__=="__main__":
    BAZ = np.array([4,5,6,7,np.nan,np.nan, 8,3,5,np.nan, 4,3,2,np.nan,np.nan])
    print(BAZ)
    spans = pick_events(BAZ)
    print(spans)