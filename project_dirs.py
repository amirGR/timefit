from os.path import dirname, join, abspath

def base_dir():
    code_dir = dirname(__file__)
    return abspath(join(code_dir,'..'))

def code_dir():
    return join(base_dir(), 'code')

def resources_dir():
    return join(base_dir(), 'code', 'resources')
    
def data_dir():
    return join(base_dir(), 'data')

def pathways_dir():
    return join(data_dir(), 'pathways')

def pathway_lists_dir():
    return join(data_dir(), 'pathways', 'pathway_lists')

def cache_dir():
    return join(base_dir(), 'cache')

def results_dir():
    return join(base_dir(), 'results')
    
def priors_dir():
    return join(code_dir(), 'priors')
    
def templates_dir():
    return join(code_dir(), 'templates')

def fit_results_relative_path(d,fitter): # d can be data or dataset
    s = d.pathway
    if d.age_scaler is not None:
        s = '{}-{}'.format(d.age_scaler.cache_name(),s)
    if d.age_restriction:
        s = '{}-{}'.format(d.age_restriction,s)
    if d.is_shuffled:
        s = 'shuffled-{}'.format(s)
    fitname = 'fits-{}-{}'.format(s, fitter.cache_name())
    return join(d.name, fitname)
