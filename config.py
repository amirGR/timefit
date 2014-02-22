import numpy as np

random_seed = 0 # None means initialize using time or /dev/urandom

fontsize = 18
n_sigmoid_points_to_plot = 100

default_figure_size_x = 18.5
default_figure_size_y = 10.5
default_figure_facecolor = 0.85 * np.ones(3)
default_figure_dpi = 100

b_verbose_optmization = False
b_allow_less_restarts = True
b_minimal_restarts = False
if b_minimal_restarts:
    n_optimization_restarts = 2
    n_max_optimization_attempt_factor = 10
else:
    n_optimization_restarts = 10
    n_max_optimization_attempt_factor = 20
all_fits_n_jobs = -2 #1
all_fits_verbose = 70 #0

# theta = [a,h,mu,w]
theta_prior_mean = np.array([5, 5, 30, 2.5])
theta_prior_sigma = np.array([5, 5, 30, 2.5])
n_folds_for_hadas_fit = 10

# these two settings must change together
from sklearn.metrics import r2_score as score
score_type = 'R2'

sorted_regions = [
    'DFC','OFC','VFC','MFC',
    'M1C','S1C','IPC',
    'ITC','STC','A1C','V1C',
    'AMY', 'CBC','HIP', 'MD','STR',
];

html_table_threshold_score = 0.3

pathways = {
    'test' : ['HTR2A', 'HTR2B'],
    'serotonin': [
        'HTR2A', 'HTR2B', 'HTR2C', 
        'HTR1A', 'HTR1B', 'HTR1D', 'HTR1E', 'HTR1F', 
        'HTR3A', 'HTR3B', 'HTR3C', 'HTR3D', 'HTR3E', 
        'HTR4', 'HTR5A', 'HTR6', 'HTR7', 
        'TPH2', 'TPH1', 'DDC', 
        'SLC18A1', 'SLC18A2', 'SLC6A4', 
        'MAOA', 'MAOB',
    ],
    'cannabinoids': [
        'CNR1','CNR2','CALB1','CALB2','DAGLA','DAGLB',
        'NAPEPLD','GDE1','ABHD4','PTPN22', 'MGLL','ABHD6','FAAH',
    ],
    
}