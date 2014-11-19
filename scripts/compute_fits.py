import setup
import re
import sys
from sklearn.datasets.base import Bunch
from utils.misc import disable_all_warnings, covariance_to_correlation
from all_fits import get_all_fits, save_as_mat_files
from fit_score import loo_score
from command_line import get_common_parser, process_common_inputs
from plots import save_fits_and_create_html
from sigmoid_change_distribution import add_change_distributions, compute_dprime_measures_for_all_pairs, export_timing_info_for_all_fits, compute_fraction_of_change


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
    
def create_html(data, fitter, fits, html_dir, k_of_n, use_correlations, correlations, show_onsets, show_change_distributions, no_legend):
    print """
==============================================================================================
==============================================================================================
==== Writing HTML
==============================================================================================
==============================================================================================
"""
    figure_kw = dict(
        show_legend = not no_legend,
    )
    basic_kw = dict(
        fits = fits,
        basedir = html_dir, 
        k_of_n = k_of_n, 
        use_correlations = use_correlations,
        correlations = correlations,
        show_change_distributions = show_change_distributions,
    )
    
    if show_onsets:
        html_kw = dict(
            extra_top_links = [ 
                ('onsets.html','Onset Times'),
                ('onsets-tooltips.html','Onset Times (with tooltips)'),
            ],
        )
    else:
        html_kw = None
    save_fits_and_create_html(data, fitter, html_kw=html_kw, figure_kw=figure_kw, **basic_kw)

    if show_onsets:
        for tooltips in [False, True]:
            bin_edges = fits.change_distribution_params.bin_edges
            R2_color_threshold = 0.2
            def get_change_distribution_info(fit):
                x_median, x_from, x_to = fit.change_distribution_spread
                childhood = [0,12]
                adolescence = [12,24]
                if data.age_scaler is None:
                    age = x_median
                else:
                    age = data.age_scaler.unscale(x_median)
                    x_from = data.age_scaler.unscale(x_from)
                    x_to = data.age_scaler.unscale(x_to)
                    childhood = [data.age_scaler.scale(x) for x in childhood]
                    adolescence = [data.age_scaler.scale(x) for x in adolescence]
                pct_childhood = 100.0 * compute_fraction_of_change(fit.change_distribution_weights, bin_edges, *childhood)
                pct_adolescence = 100.0 * compute_fraction_of_change(fit.change_distribution_weights, bin_edges, *adolescence)
                if tooltips:
                    txt = '<div title="0-12 years: {pct_childhood:.2g}%\n12-24 years: {pct_adolescence:.2g}%">{age:.2g} </br> <small>({x_from:.2g},{x_to:.2g})</small></div>'.format(**locals())
                else:
                    txt = '{age:.2g} </br> <small>({x_from:.2g},{x_to:.2g}) <br/> [{pct_childhood:.2g}%, {pct_adolescence:.2g}%] </small>'.format(**locals())
                if fit.LOO_score > R2_color_threshold: # don't use correlations even if we have them. we want to know if the transition itself is significant in explaining the data
                    cls = 'positiveTransition' if fitter.shape.is_positive_transition(fit.theta) > 0 else 'negativeTransition'
                else:
                    cls = ''
                return txt,cls
    
            percentage_location = 'tooltips for cell entries' if tooltips else 'square brackets'
            top_text = """\
    All onset times are in years. <br/>
    The main number is the median age. The two numbers (age1,age2) beneath the onset age are the range where most of the transition occurs. </br>
    The two percentages in {percentage_location} are the fraction of the change that happens during ages 0-12 years and during adolescence (12-24 years) respectively. </br>
    The onset age and range are estimated using bootstrap samples and may differ from the onset and width of the single best fit as displayed in the figure. 
    </p>
    <p>
    red = strong positive transition (R2 > {R2_color_threshold} and expression level increases with age) </br>
    blue = strong negative transition (R2 > {R2_color_threshold} and expression level decreases with age) </br>
    (for assessing transition strength, R2 above is LOO R2 without using correlations between genes)
    </p>
    """.format(**locals())
            if use_correlations:
                top_text += """
    <p>Click on a region name to see the correlation matrix for that region.</p>
    """
    
            html_kw = dict(
                filename = 'onsets-tooltips' if tooltips else 'onsets',
                ttl = 'Onset times',
                top_text = top_text,
                show_R2 = False,
                extra_fields_per_fit = [get_change_distribution_info],
                b_R2_dist = False, 
            )
            save_fits_and_create_html(data, fitter, only_main_html=True, html_kw=html_kw, figure_kw=figure_kw, **basic_kw)

def save_mat_file(data, fitter, fits, has_change_distributions):
    print """
==============================================================================================
==============================================================================================
==== Saving matlab file(s)
==============================================================================================
==============================================================================================
"""
    save_as_mat_files(data, fitter, fits, has_change_distributions)

def add_predictions_using_correlations(data, fitter, fits):
    correlations = {} # {region -> sigma}
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
        preds,_,sigma = fitter.fit_multiple_series_with_cache(series.ages, series.expression, cache)
        correlations[r] = covariance_to_correlation(sigma)
        for iy,g in enumerate(series.gene_names):
            fit = ds_fits[(g,r)]
            y_real = series.expression[:,iy]
            y_preds = preds[:,iy]
            fit.with_correlations = Bunch(
                LOO_predictions = y_preds,
                LOO_score = loo_score(y_real, y_preds),
            )
    return fits, correlations

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
    parser.add_argument('--onset', action='store_true', help='Show onset times and not R2 scores in HTML table (sigmoid only)')
    parser.add_argument('--dont_show_change_dist', action='store_true', help="Don't show change distribution in the figures (only relevant for sigmoids and together with --html)")
    parser.add_argument('--no_legend', action='store_true', help="Don't show the legend in the figures (only relevant together with --html)")
    parser.add_argument('--change_dist', action='store_true', help='Compute change distributions and related measures (sigmoid only)')
    args = parser.parse_args()
    if args.part is not None and args.mat:
        print '--mat cannot be used with --part'
        sys.exit(-1)
    is_sigmoid = args.shape in ['sigmoid','sigslope']
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
    if args.onset and not is_sigmoid:
        print '--onset can only be used with sigmoid fits'
        sys.exit(-1)
    if args.change_dist and not is_sigmoid:
        print '--change_dist can only be used with sigmoid fits'
        sys.exit(-1)
    if args.onset and args.html == NOT_USED:
        print '--onset should only be used with --html'
        sys.exit(-1)
    k_of_n = parse_k_of_n(args.part)
    data, fitter = process_common_inputs(args)
    fits = do_fits(data, fitter, k_of_n)
    if args.correlations:
        fits, correlations = add_predictions_using_correlations(data, fitter, fits)
    else:
        correlations = None
    has_change_distributions = is_sigmoid
    if has_change_distributions:
        print 'Computing change distributions...'
        add_change_distributions(data, fitter, fits)
        if args.change_dist:
            print 'Computing region pair timing measures...'
            compute_dprime_measures_for_all_pairs(data, fitter, fits)
            export_timing_info_for_all_fits(data, fitter, fits)
    if args.html != NOT_USED:
        create_html(data, fitter, fits, args.html, k_of_n, 
                    use_correlations=args.correlations, 
                    correlations=correlations, 
                    show_onsets=args.onset,
                    show_change_distributions = has_change_distributions and not args.dont_show_change_dist,
                    no_legend = args.no_legend,
                    )
    if args.mat:
        save_mat_file(data, fitter, fits, has_change_distributions)
