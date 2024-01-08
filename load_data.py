import os
import numpy as np
from obspy import read, Stream
from obspy.signal.filter import bandpass
from datetime import datetime, timedelta
import config
import load_data as ld

def get_data(folder_path, freqmin, freqmax, param):
    # Initialize an empty stream to store the traces
    stream = Stream()

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pri0"):
            file_path = os.path.join(folder_path, filename)

            # Read the MiniSEED file
            st = read(file_path)

            # Extract the Z trace with an ending of ".pri0"
            z_trace = None
            for trace in st:
                z_trace = trace.copy()

            if z_trace is not None:
                # Apply bandpass filtering
                z_trace.data = bandpass(z_trace.data, freqmin, freqmax, z_trace.stats.sampling_rate, corners=2)

                # Append the filtered trace to the stream
                stream += z_trace

    # Extract traces from the stream
    traces = [trace.data for trace in stream]

    # Convert the list of traces into a NumPy array
    numpy_array = np.array(traces)

    # Extract station names
    stations = [trace.id[1:6] for trace in stream]
    

    # Sort parameters
    Ind = get_index(param["stations"], stations)

    param["stations"] = [item for index, item in sorted(zip(Ind, param["stations"]), key=custom_sort) if not np.isnan(index)]
    param["xm"] = [item for index, item in sorted(zip(Ind, param["xm"]), key=custom_sort) if not np.isnan(index)]
    param["ym"] = [item for index, item in sorted(zip(Ind, param["ym"]), key=custom_sort) if not np.isnan(index)]
    param["zm"] = [item for index, item in sorted(zip(Ind, param["zm"]), key=custom_sort) if not np.isnan(index)]
    param["sr"] = stream[0].stats.sampling_rate

    # Find indices of entries in list2 that are missing in list1
    missing_indices = [stations.index(entry) for entry in stations if entry not in param["stations"]]

    # Delete wrong traces
    for i in range(0,len(missing_indices)):
        numpy_array = np.delete(numpy_array, missing_indices[i], axis=0)

    # Generate corresponding datetime array
    t = generate_datetime_array(config.date, config.year, np.size(numpy_array,1))

    # Cut data according to time interval
    [numpy_array, t] = cut_data(numpy_array,t, config.start_time, config.duration_minutes, param["sr"])

        
    return numpy_array, t, param


def custom_sort(item):
    index, value = item
    # Use float('inf') for np.nan to ensure it is placed at the end
    return (float('inf') if np.isnan(index) else index, value)

def get_parameters(folder_program, str_param=None):
    """"
    Reading Parameters from parameter file. If no name is given, the user has to choose a file
    Input:
        folder_program: The folder in which the program runs (str)
        str_param: Name of the parameter file (str)
    """
    # Select text file with station parameters.
    if str_param is None or str_param == '':
        str0 = input('Enter streaming parameter file name: ')
        str_param = os.path.join(folder_program, 'Streaming_Parameters', f'{str0}.txt')
    else:
        str_param = os.path.join(folder_program, 'Streaming_Parameters', f'{str_param}.txt')

    # Read the text file
    with open(str_param, 'r') as fid:
        # Skip the first line
        fid.readline()

        # Read the second line for var1
        var1 = [float(x) for x in fid.readline().split()]

        # Read the rest of the lines for var2
        var2 = []
        for line in fid:
            var2.extend(line.split())
    num = int(len(var2)/8)
    [IP, station, network, channel0, location, xm, ym, zm] = [[None]*num, [None]*num, [None]*num, [None]*num, 
                                                            [None]*num, [None]*num,[None]*num,[None]*num]
    # Define variables
    lp, hp, corr_win, corr_res, shift_size = var1
    for i in range(0, num):
        IP[i], station[i], network[i], channel0[i], location[i], xm[i], ym[i], zm[i] = var2[i*8:(i+1)*8]
    
    nroot = 0

    return {
        'IP': IP,
        'network': network,
        'hp': hp,
        'lp': lp,
        'stations': station,
        'nroot': nroot,
        'channel0': channel0,
        'str_param': str_param,
        'xm': np.array(xm).astype(float),
        'ym': np.array(ym).astype(float),
        'zm': np.array(zm).astype(float),
        'location': location,
        'corr_win': corr_win,
        'corr_res': corr_res,
        'shift_size': shift_size
    }

def get_index(list1, list2):
    """"
    Get Index of entries in list one, in list two
    """
    
    # Create a dictionary to map items in list2 to their indices
    index_dict = {item: index for index, item in enumerate(list2)}

    # Get the indices for each item in list1, or use np.nan if not present
    indices = [index_dict.get(item, np.nan) for item in list1]

    return indices


def generate_datetime_array(start_date_str, year_str, array_length):
    """"
    Generate one day long datetime array with specific length
    Input:
        start_date_str: Start date in Format "dd_mm" (str)
        year_str: the year (str)
        array_length: length of the output array (int)
    Output:
        datetime_array (array)
    """
    # Parse the start date string in the format "dd_MM" and set the year to 2023
    start_date = datetime.strptime(start_date_str + "_" + year_str, "%d_%m_%Y")

    # Generate a datetime array with a specified length and duration of one day
    datetime_array = [start_date + timedelta(days=1/array_length*i) for i in range(array_length)]

    return datetime_array

def cut_data(d, t, start_time, duration, sr):
    time = datetime.strptime(start_time, "%H:%M:%S")
    seconds = int((time - datetime(1900, 1, 1)).total_seconds())
    start_index = int(seconds*sr)
    end_index = start_index + int(duration*60*sr)
    d_cut = d[:,start_index:end_index]
    t_cut = t[start_index:end_index]
    return d_cut, t_cut


if __name__ == "__main__":
    # # Example usage
    # folder_path = "/home/patrick/Daten Salomonen/DATA-Mseed/Events/10_03"
    # freqmin = 4.5  # Minimum frequency for bandpass filter
    # freqmax = 80.0  # Maximum frequency for bandpass filter

    # stream = get_data(folder_path, freqmin, freqmax)
    

#    [IP, network, hp, lp, station, nroot, channel0, str_param, xm, ym, zm, location, corr_win, corr_res, shift_size] = get_parameters('/home/patrick/Processing_Seedlink', str_param='Test_case_gross2')

#    print(network) 
    
    ## Example lists (replace these with your actual lists)
    #list1 = ["apple", "banana", "orange", "grape"]
    #list2 = ["orange", "banana", "apple"]

    #get_index(list1, list2)
    param = ld.get_parameters(config.folder_program, config.parameter_file)
    [data, t, param] = ld.get_data(config.folder_data, config.freqmin, config.freqmax, param)
    
    
    