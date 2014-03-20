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

def cache_dir():
    return join(base_dir(), 'cache')

def results_dir():
    return join(base_dir(), 'results')

def fit_results_relative_path(data,fitter):
    s = data.pathway
    if data.age_scaler is not None:
        s = '{}-{}'.format(data.age_scaler.cache_name(),s)
    if data.postnatal_only:
        s = 'postnatal-{}'.format(s)
    fitname = 'fits-{}-{}'.format(s, fitter.cache_name())
    return join(data.dataset, fitname)
