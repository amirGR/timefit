import setup
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nanmean, nanstd
import config as cfg
from utils.misc import disable_all_warnings
from load_data import load_data
from plots import save_figure
from scalers import LogScaler
from dev_stages import dev_stages

scaler = LogScaler()

def plot_means(dataset):
    min_age = min(dataset.ages)
    max_age = max(dataset.ages)
    min_expression = np.nanmin(dataset.expression.flat)
    max_expression = np.nanmax(dataset.expression.flat)
    
    center = np.empty(dataset.ages.shape)
    std_plus = np.empty(dataset.ages.shape)
    std_minus = np.empty(dataset.ages.shape)
    for i,age in enumerate(dataset.ages):
        a = dataset.expression[i,:,:].flat
        c = nanmean(a)
        s = nanstd(a)
        center[i] = c
        std_plus[i] = c + s
        std_minus[i] = c - s

    fig = plt.figure()
    ax = fig.add_axes([0.08,0.15,0.85,0.8])

    ax.set_ylabel('expression level', fontsize=cfg.fontsize)
    ax.set_xlabel('age', fontsize=cfg.fontsize)
    ax.set_title('Mean expression across all genes - {}'.format(dataset.name), fontsize=cfg.fontsize)
        
    # set the development stages as x labels
    stages = [stage.scaled(scaler) for stage in dev_stages]
    ax.set_xticks([stage.central_age for stage in stages])
    ax.set_xticklabels([stage.short_name for stage in stages], fontsize=cfg.xtick_fontsize, fontstretch='condensed', rotation=90)    
    ax.set_xlim([min_age, max_age])
    
    # mark birth time with a vertical line
    ymin, ymax = ax.get_ylim()
    birth_age = scaler.scale(0)
    ax.plot([birth_age, birth_age], [ymin, ymax], '--', color='0.85')
    
    
    ax.plot([min_age, max_age], [min_expression, min_expression], '--g')
    ax.plot([min_age, max_age], [max_expression, max_expression], '--g')
    ax.plot(dataset.ages, center, 'bx')
    ax.plot(dataset.ages, std_plus, 'g-')
    ax.plot(dataset.ages, std_minus, 'g-')
    
    save_figure(fig,'mean-expression-{}.png'.format(dataset.name), under_results=True)
    
def main():
    data = load_data().scale_ages(scaler)
    for ds in data.datasets:
        plot_means(ds)

if __name__ == '__main__':
    disable_all_warnings()   
    main()
