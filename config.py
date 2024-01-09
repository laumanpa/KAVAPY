import numpy as np
# Data processing
freqmin = 4.5 # Highpass Frequency
freqmax = 80  # Lowpass frequency

# Choose date
# date = "26_03" 
# year = "2023"
# start_time = "22:57:10"
date = "05_04" 
year = "2023"
start_time = "08:00:00"
duration_minutes = 3

# set folders
distro = "Linux" # "Windows" or "Linux"
if distro == "Linux":
    folder_program = '/home/patrick/OneDrive/Rümpker/Salomonen/KAVAPY'
    parameter_file = 'Test_case_gross2'
    folder_data = '/home/patrick/Daten Salomonen/DATA-Mseed/Events/' + date
elif distro == "Windows":
    folder_program = 'C:\\Users\\lauma\\OneDrive\\Rümpker\\Salomonen\\KAVAPY'
    parameter_file = 'Test_case_gross2'
    folder_data = 'E:\\Daten_Salomonen\\DATA-Mseed\\Events\\' + date

# Picking parameters
c_median_threshold = 0.1
amp_threshold = 500
rmse_threshold = 0.05 #0.054
std_threshold = 200
std_window = 20
mode_window = 40
processing_window = 300 # in samples
corr_res = 10 #20
min_length = 1 # minimal length of span window
v_app_min = 0 # minimal apparent velocity
v_app_max = 9 # maximal apparent velocity
max_delay = 2 # maximal delay in seconds

# Inversion parameters
# Choose inversion method
inversion_method = "Tikhonov" # "Huber" or "Tikhonov"

# Plotting parameters
fit_plot_flag = False