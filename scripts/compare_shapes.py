import setup
from os.path import join
import matplotlib.pyplot as plt
import config as cfg
from utils.misc import disable_all_warnings, ensure_dir
from project_dirs import results_dir
from all_fits import get_all_fits
from command_line import get_common_parser, process_common_inputs, get_data_from_args, get_fitter_from_args
from shapes.shape import allowed_shape_names
from shapes.priors import get_allowed_priors
from scalers import allowed_scaler_names
from plots import save_figure

def print_diff_points(data1, fitter1, data2, fitter2, n):
    fits1 = get_all_fits(data1,fitter1)
    fits2 = get_all_fits(data2,fitter2)

    diffs = [(fits1[k].LOO_score-fits2[k].LOO_score, k) for k in fits1.iterkeys()]
    diffs.sort()
    
    print 'Top {} fits where {} > {}:'.format(n, fitter1.shape, fitter2.shape)
    for diff,k in diffs[-n:]:
        g,r = k
        score1 = fits1[k].LOO_score
        score2 = fits2[k].LOO_score
        print '\t{}@{}: diff={:.2g}, {}={:.2g}, {}={:.2g}'.format(g,r,diff,fitter1.shape,score1,fitter2.shape,score2)

    print 'Top {} fits where {} < {}:'.format(n, fitter1.shape, fitter2.shape)
    for diff,k in diffs[:n]:
        g,r = k
        score1 = fits1[k].LOO_score
        score2 = fits2[k].LOO_score
        print '\t{}@{}: diff={:.2g}, {}={:.2g}, {}={:.2g}'.format(g,r,diff,fitter1.shape,score1,fitter2.shape,score2)

def plot_comparison_scatter(data1, fitter1, data2, fitter2):
    fits1 = get_all_fits(data1,fitter1)
    fits2 = get_all_fits(data2,fitter2)

    pairs = [(fits1[k].LOO_score, fits2[k].LOO_score) for k in fits1.iterkeys()]
    scores1,scores2 = zip(*pairs)
    
    fig = plt.figure()
    plt.scatter(scores1, scores2, alpha=0.5)
    plt.plot([-1, 1], [-1, 1],'k--')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    ttl1 = r'Comparison of scores using {} vs. {}'.format(fitter1.shape,fitter2.shape)
    ttl2 = r'{}, {}'.format(data1.dataset, data1.pathway)
    plt.title('\n'.join([ttl1, ttl2]), fontsize=cfg.fontsize)
    plt.xlabel('R2 for {}'.format(fitter1.shape), fontsize=cfg.fontsize)
    plt.ylabel('R2 for {}'.format(fitter2.shape), fontsize=cfg.fontsize)    
    return fig

if __name__ == '__main__':
    disable_all_warnings()
    parser = get_common_parser()
    parser.add_argument('--shape2', required=True, help='The shape to compare against', choices=allowed_shape_names())
    parser.add_argument('--scaling2', help='The scaling used when fitting shape2. Default: none', choices=allowed_scaler_names())
    parser.add_argument('--sigma_prior2', help='Prior to use for 1/sigma when fitting shape2. Default: None', choices=get_allowed_priors(is_sigma=True))
    parser.add_argument('--priors2', help='The priors used for theta when fitting shape2. Default: None', choices=get_allowed_priors())
    parser.add_argument('--filename', help='Where to save the figure. Default: results/comparison.png')
    parser.add_argument('--show', help='Show figure and wait before exiting', action='store_true')
    parser.add_argument('--ndiffs', type=int, default=5, help='Number of top diffs to show. Default=5.')
    args = parser.parse_args()
    data1, fitter1 = process_common_inputs(args)    
    data2 = get_data_from_args(args.dataset, args.pathway, args.postnatal, args.scaling2)
    fitter2 = get_fitter_from_args(args.shape2, args.priors2, args.sigma_prior2)

    print_diff_points(data1,fitter1,data2,fitter2,args.ndiffs)

    fig = plot_comparison_scatter(data1,fitter1,data2,fitter2)

    filename = args.filename    
    if filename is None:
        ensure_dir(results_dir())
        filename = join(results_dir(), 'shape_comparison.png')
    print 'Saving figure to {}'.format(filename)
    save_figure(fig, filename)    

    if args.show:
        plt.show(block=True)
