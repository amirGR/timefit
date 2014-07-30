import setup
import re
import sys
from sklearn.datasets.base import Bunch
from utils.misc import disable_all_warnings
from all_fits import get_all_fits, save_as_mat_files
from fit_score import loo_score
from command_line import get_common_parser, process_common_inputs
from plots import save_fits_and_create_html

def do_fits(data, fitter, k_of_n):
    print """
==============================================================================================
==============================================================================================
==== Computing Fits with {}
==============================================================================================
==============================================================================================
""".format(fitter)
    fits = get_all_fits(data, fitter, k_of_n)    
    return fits
    
def create_html(data, fitter, fits, html_dir, k_of_n, use_correlations, show_onsets):
    print """
==============================================================================================
==============================================================================================
==== Writing HTML
==============================================================================================
==============================================================================================
"""
    if show_onsets:
        kw = dict(
            extra_top_links = [ 
                ('onsets.html','Onset Times'),
            ],
        )
    else:
        kw = None
    save_fits_and_create_html(data, fitter,         
        fits = fits,
        basedir = html_dir, 
        k_of_n = k_of_n, 
        use_correlations = use_correlations,
        html_kw = kw,
    )

    if show_onsets:
        def get_onset_time(fit):
            a,h,mu,w = fit.theta
            age = data.age_scaler.unscale(mu)
            txt = '{:.2g}'.format(age)
            if fit.LOO_score > 0.2: # don't use correlations even if we have them. we want to know if the transition itself is significant in explaining the data
                cls = 'positiveTransition' if h*w > 0 else 'negativeTransition'
            else:
                cls = ''
            return txt,cls

        kw = dict(
            filename = 'onsets',
            ttl = 'Onset times',
            top_text = """\
All onset times are in years.

red = strong positive transition.
blue = strong negative transition.
""",
            show_R2 = False,
            extra_fields_per_fit = [get_onset_time],
            b_R2_dist = False, 
        )

        save_fits_and_create_html(data, fitter,         
            fits = fits,
            basedir = html_dir, 
            k_of_n = k_of_n, 
            use_correlations = use_correlations,
            html_kw = kw,
        )

    else:
        kw = None

def save_mat_file(data, fitter, fits):
    print """
==============================================================================================
==============================================================================================
==== Saving matlab file(s)
==============================================================================================
==============================================================================================
"""
    save_as_mat_files(data, fitter, fits)

def add_predictions_using_correlations(data, fitter, fits):
    for r in data.region_names:
        print 'Analyzing correlations for region {}...'.format(r)
        series = data.get_several_series(data.gene_names,r)
        ds_fits = fits[data.get_dataset_for_region(r)]
        def cache(iy,ix):
            g = series.gene_names[iy]
            fit = ds_fits[(g,r)]
            if ix is None:
                return fit.theta
            else:
                theta,sigma = fit.LOO_fits[ix]
                return theta    
        preds,_ = fitter.fit_multiple_series_with_cache(series.ages, series.expression, cache)
        for iy,g in enumerate(series.gene_names):
            fit = ds_fits[(g,r)]
            y_real = series.expression[:,iy]
            y_preds = preds[:,iy]
            fit.with_correlations = Bunch(
                LOO_predictions = y_preds,
                LOO_score = loo_score(y_real, y_preds),
            )
    return fits

def parse_k_of_n(s):
    """Parse a string that looks like "3/5" and return tuple (3,5)"""
    if s is None:
        return None
    m = re.match('(\d+)/(\d+)',s)
    if m is None:
        print '{} is not a valid part description. Format is k/n.'.format(s)
        sys.exit(-1)
    return tuple(int(x) for x in m.groups())

if __name__ == '__main__':
    disable_all_warnings()
    NOT_USED = (None,)
    parser = get_common_parser()
    parser.add_argument('--part', help='Compute only part of the genes. format: <k>/<n> e.g. 1/4. (k=1..n)')
    parser.add_argument('--html', nargs='?', metavar='DIR', default=NOT_USED, help='Create html for the fits. Optionally override output directory.')
    parser.add_argument('--mat', action='store_true', help='Save the fits also as matlab .mat file.')
    parser.add_argument('--correlations', action='store_true', help='Use correlations between genes for prediction')
    parser.add_argument('--onset', action='store_true', help='Show onset times and not R2 scores in HTML table')
    args = parser.parse_args()
    if args.part is not None and args.mat:
        print '--mat cannot be used with --part'
        sys.exit(-1)
    if args.correlations:
        if args.part:
            print '--correlations cannot be used with --part'
            sys.exit(-1)
        if args.mat:
            print '--correlations not compatible with --mat'
            sys.exit(-1)
        if args.html == NOT_USED:
            print '--correlations only currently makes sense with --html (since fits are not saved)'
            sys.exit(-1)
    if args.onset and args.html == NOT_USED:
        print '--onset should only be used with --html'
        sys.exit(-1)
    k_of_n = parse_k_of_n(args.part)
    data, fitter = process_common_inputs(args)
    fits = do_fits(data, fitter, k_of_n)
    if args.correlations:
        add_predictions_using_correlations(data, fitter, fits)
    if args.html != NOT_USED:
        create_html(data, fitter, fits, args.html, k_of_n, use_correlations=args.correlations, show_onsets=args.onset)
    if args.mat:
        save_mat_file(data,fitter,fits)
