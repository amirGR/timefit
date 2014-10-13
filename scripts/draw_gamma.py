import setup
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from shapes.priors import GammaPrior

def draw_gamma(alpha, beta, mu, xmin=-0.1, xmax=1):
    fig = plt.figure()
    x = np.linspace(xmin, xmax, 200)
    gamma = GammaPrior(alpha,beta,mu)
    y = gamma.rv.pdf(x)
    plt.plot(x,y,'g',linewidth=3)
    plt.xlabel('x',fontsize=cfg.fontsize)
    plt.ylabel('p(x)',fontsize=cfg.fontsize)
    return fig

draw_gamma(alpha=1.3, beta=1.1, mu=0.0)
