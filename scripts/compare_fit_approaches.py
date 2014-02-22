import setup
from load_data import load_data
import sigmoid_fit as new_fits
import sigmoid_fit_hadas as old_fits
from sigmoid_fit import loo_score

def compare(gene,region):
    data = load_data()   
    series = data.get_one_series(gene,region)
    x = series.ages
    y = series.expression
    
    print 'Fitting using old method...'
    old_preds = old_fits.fit_sigmoid_loo(x,y)
    
    print 'Fitting using new method...'
    new_preds = new_fits.fit_sigmoid_loo(x,y)
    
    print 'Comparing...'
    old_score = loo_score(y,old_preds)
    new_score = loo_score(y,new_preds)
    print 'Old score = {}'.format(old_score)
    print 'New score = {}'.format(new_score)

import sys
if __name__ == '__main__':
    try:
        progname, gene, region = sys.argv
    except:
        progname = sys.argv[0]
        print 'Usage: {} <gene> <region>'.format(progname)
    else:
        compare(gene,region)

