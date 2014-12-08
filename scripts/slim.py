import setup
import os
import argparse
import cPickle as pickle
from os.path import join

def format_file_size(num):
    units = ['','K','M','G', 'T']
    for unit in units[:-1]:
        if abs(num) < 1024.0:
            return "{:3.1f} {}B".format(num,unit)
        num /= 1024.0
    return "{.1f} {}B".format(num, units[-1])
    
def size_str(filename):
    statinfo = os.stat(filename)
    size = statinfo.st_size
    return format_file_size(size)
    
def slim(filename):
    orig_size = size_str(filename)
    print 'Reading file {} ({})...'.format(filename, orig_size)
    with open(filename) as f:
        dct = pickle.load(f)
        
    print 'Removing LOO fits. Processing {} fits...'.format(len(dct))
    n_removed = 0
    for k,v in dct.iteritems():
        if hasattr(v, 'LOO_fits'):
            n_removed += 1
            del v.LOO_fits
        try:
            v.pop('LOO_fits',None)
        except:
            pass # not a dictionary
    print 'Removed {} objects'.format(n_removed)
        
    print 'Writing back to {}...'.format(filename)
    with open(filename,'w') as f:
        pickle.dump(dct,f)

    new_size = size_str(filename)
    print 'Size reduction: {} -> {}'.format(orig_size, new_size)

def slim_dir(dirname):
    filenames = [f for f in os.listdir(dirname) if f.endswith('.pkl')]
    for filename in filenames:
        slim(join(dirname,filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='The filename to remove LOO fit parameters from')
    parser.add_argument('--dir', help='Process all pkl files in this directory')
    args = parser.parse_args()
    if args.dir is not None:
        slim_dir(args.dir)
    elif args.file is not None:
        slim(args.file)
