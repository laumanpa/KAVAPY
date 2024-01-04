# Data processing
freqmin = 4.5 # Highpass Frequency
freqmax = 80  # Lowpass frequency

# Choose date
date = "26_03" 
year = "2023"
start_time = "22:57:25"
duration_minutes = 3

# set folders
folder_program = '/home/patrick/OneDrive/Rümpker/Salomonen/KAVAPY'
parameter_file = 'Test_case_gross2'
folder_data = '/home/patrick/Daten Salomonen/DATA-Mseed/Events/' + date

# Picking parameters
c_median_threshold = 0.2
amp_threshold = 500
rmse_threshold = 0.5
std_threshold = 0.2
std_window = 20
mode_window = 10
processing_window = 300
corr_res = 10 #20
min_length = 20 # minimal length of span window