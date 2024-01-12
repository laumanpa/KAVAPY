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
distro = "Windows" # "Windows" or "Linux"
if distro == "Linux":
    folder_program = '/home/patrick/OneDrive/Rümpker/Salomonen/KAVAPY'
    parameter_file = 'Test_case_gross2'
    folder_data = '/home/patrick/Daten Salomonen/DATA-Mseed/Events/' + date
elif distro == "Windows":
    folder_program = 'C:\\Users\\lauma\\OneDrive\\Rümpker\\Salomonen\\KAVAPY'
    parameter_file = 'Test_case_gross2'
    folder_data = 'E:\\Daten_Salomonen\\DATA-Mseed\\Events\\' + date

# Picking parameters
c_median_threshold = 0.25
amp_threshold = 500
rmse_threshold = 0.05 #0.054
std_threshold = 200
std_window = 20
mode_window = 40
processing_window_seconds = 0.5 # in seconds
corr_res = 20 #20
min_length = 1 # minimal length of span window
v_app_min = 0 # minimal apparent velocity
v_app_max = 20 # maximal apparent velocity
max_delay = 2 # maximal delay in seconds

# Inversion parameters
# Choose inversion method
execution_mode = "processing" # "processing" or "testing_inversion" or "testing_all"
inversion_method = "Huber" # "Huber" or "Tikhonov"

# Huber parameters
Huber_epsilon = 1.35 

# Tikhonov parameters
Tikhonov_alpha = 'gcv' # Const value for alpha, or method for alpha calculation ('gcv' or 'l_curve')
Tikhonov_k = 1.0

# Plotting parameters
fit_plot_flag = True