import setup
import argparse
import cPickle as pickle


def slim(filename):
    print 'Reading file {}...'.format(filename)
    with open(filename) as f:
        dct = pickle.load(f)
        
    print 'Removing LOO fits. Processing {} fits...'.format(len(dct))
    n_removed = 0
    for k,v in dct.iteritems():
        if hasattr(v, 'LOO_fits'):
            n_removed += 1
            del v.LOO_fits
        v.pop('LOO_fits',None)
    print 'Removed {} objects'.format(n_removed)
        
    print 'Writing back to {}...'.format(filename)
    with open(filename,'w') as f:
        pickle.dump(dct,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='The filename to remove LOO fit parameters from')
    args = parser.parse_args()
    slim(args.file)
