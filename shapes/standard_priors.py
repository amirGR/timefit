from priors import NormalPrior, GammaPrior
from os.path import isfile, splitext, basename

########################################################################
# Sigma priors
########################################################################
def get_sigma_prior(name):
    if name is None:
        return None,None
    if name in sigma_priors:
        return name, sigma_priors[name]
    if isfile(name):
        path = name
        name = splitext(basename(path))[0] # use the file's basename (without extension) as the priors name
        with open(path) as f:
            priors = eval(f.read())
        return name,priors
    raise Exception('Unknown prior: {}'.format(name))

sigma_priors = {
    'gamma' : GammaPrior(2.61,1.15,0.65),
    'normal': NormalPrior(mu=5,sigma=5),
}

########################################################################
# Theta priors
########################################################################
def get_theta_priors(name):
    if name is None:
        return None,None
    if name in theta_priors:
        return name, theta_priors[name]
    if isfile(name):
        path = name
        name = splitext(basename(path))[0] # use the file's basename (without extension) as the priors name
        with open(path) as f:
            priors = eval(f.read())
        return name,priors
    raise Exception('Unknown prior: {}'.format(name))

def standard_poly_priors(n):
    return [NormalPrior(0,1) for _ in range(n+1)]

theta_priors = {
    #########################
    # Sigmoid
    #########################
    'sigmoid_wide' : [
        NormalPrior(mu=8, sigma=10), # baseline
        NormalPrior(mu=8, sigma=8), # height
        NormalPrior(mu=30, sigma=50), # onset
        NormalPrior(mu=0, sigma=30), # width
    ],
    
    'sigmoid_kang_serotonin' : [
        GammaPrior(alpha=1.87, beta=1.135, mu=3.65), # baseline
        NormalPrior(mu=0.28, sigma=0.84), # height
        GammaPrior(alpha=1.41, beta=0.055, mu=-4.31), # onset
        NormalPrior(mu=2.5, sigma=2.5), # width
    ],

    #########################
    # Poly
    #########################
    'poly0' : standard_poly_priors(0),
    'poly1' : standard_poly_priors(1),
    'poly2' : standard_poly_priors(2),
    'poly3' : standard_poly_priors(3),
}

