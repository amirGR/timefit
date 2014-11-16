import numpy as np

verbosity = 0
random_seed = 1 # None means initialize using time or /dev/urandom

fontsize = 30
xtick_fontsize = 30
ytick_fontsize = 30
equation_fontsize = 36
minimal_annotation_fontsize = 10
default_figure_size_x = 18.5
default_figure_size_x_square = 12.5
default_figure_size_y = 10.5
default_figure_facecolor = 0.85 * np.ones(3)
default_figure_dpi = 100

n_curve_points_to_plot = 200

b_verbose_optmization = False
b_allow_less_restarts = True
b_minimal_restarts = False
minimization_tol = None
if b_minimal_restarts:
    n_optimization_restarts = 2
else:
    n_optimization_restarts = 10
job_batch_size = 16
job_big_key_size = 10000
job_big_batch_size = 128
parallel_n_jobs = -2 #1
parallel_run_locally = False # disable parallelization for debugging

n_folds = 30 # 0 is LOO

log_scale_x0 = -38.0/52

fitter_scaling_percentiles = (10,90)

n_parameter_estimate_bootstrap_samples = 30

# these two settings must change together
from sklearn.metrics import r2_score as score
score_type = 'R2'

sorted_regions = {
    'kang2011' : [
        'DFC','OFC','VFC','MFC',
        'M1C','S1C','IPC',
        'ITC','STC','A1C','V1C',
        'AMY', 'CBC','HIP', 'MD','STR',
    ],
    'colantuoni2011' : ['PFC'],
    'brainSpan2014': [
        'DFC','OFC','VFC','MFC',
        'M1C','S1C','IPC',
        'ITC','STC','A1C','V1C',
        'AMY', 'CBC','HIP', 'MD','STR',
    ],
}

pathways = {
    'test' : ['HTR2A', 'HTR1E'],
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
    'dopamine' : [
        'ADCY5', 'AKT1', 'AKT2', 'AKT3', 'ARNTL', 'ARRB2', 'ATF2', 'ATF4', 'ATF6B', 
        'CACNA1A', 'CACNA1B', 'CACNA1C', 'CACNA1D', 'CALM1', 'CALML3', 'CALML5', 'CALML6', 
        'CALY', 'CAMK2A', 'CAMK2B', 'CAMK2D', 'CAMK2G', 'CLOCK', 'COMT', 'CREB1', 'CREB3', 
        'CREB3L1', 'CREB3L2', 'CREB3L3', 'CREB3L4', 'CREB5', 'DDC', 
        'DRD1', 'DRD2', 'DRD3', 'DRD4', 'DRD5', 'FOS', 'GNAI1', 'GNAI2', 'GNAI3', 'GNAL', 'GNAO1', 
        'GNAQ', 'GNAS', 'GNB1', 'GNB2', 'GNB3', 'GNB4', 'GNB5', 'GNG10', 'GNG11', 'GNG12', 'GNG13', 
        'GNG2', 'GNG3', 'GNG4', 'GNG5', 'GNG7', 'GNG8', 'GNGT1', 'GNGT2', 
        'GRIA1', 'GRIA2', 'GRIA3', 'GRIA4', 'GRIN2A', 'GRIN2B', 'GSK3A', 'GSK3B', 'ITPR1', 'ITPR2', 'ITPR3', 
        'KCNJ3', 'KCNJ5', 'KCNJ6', 'KCNJ9', 'KIF5A', 'KIF5B', 'KIF5C', 'MAOA', 'MAOB', 
        'MAPK10', 'MAPK11', 'MAPK12', 'MAPK13', 'MAPK14', 'MAPK8', 'MAPK9', 
        'PLCB1', 'PLCB2', 'PLCB3', 'PLCB4', 'PPP1CA', 'PPP1CB', 'PPP1CC', 'PPP1R1B', 'PPP2CA', 
        'PPP2CB', 'PPP2R1A', 'PPP2R1B', 'PPP2R2A', 'PPP2R2B', 'PPP2R2C', 'PPP2R2D', 'PPP2R3A', 
        'PPP2R3B', 'PPP2R3C', 'PPP2R5A', 'PPP2R5B', 'PPP2R5C', 'PPP2R5D', 'PPP2R5E', 'PPP3CA', 
        'PPP3CB', 'PPP3CC', 'PRKACA', 'PRKACB', 'PRKACG', 'PRKCA', 'PRKCB', 'PRKCG', 'PRKX', 
        'SCN1A', 'SLC18A1', 'SLC18A2', 'SLC6A3', 'TH'
    ],
    '17pathways': '17pathways.txt', # references a file you should hopefully have in the data directory
    '17full': '17pathways-full.txt', # references a file you should hopefully have in the data directory
}