import setup
import urllib2
import re
from utils.parallel import Parallel
from utils.misc import retry

@retry(3) # reading from the website can fail
def get_age(_id):
    print 'Getting age for ID={}'.format(_id)
    url = 'http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM{}'.format(_id)
    html = urllib2.urlopen(url).read()
    m = re.search('age:\s*([-+.\d]+)',html)
    age = float(m.group(1))
    return age
    
def get_all_colantuoni_ages():  
    ids = range(749899,750168) # copied from main accession page
    pool = Parallel(get_age, verbosity=1)
    ages = pool(pool.delay(_id) for _id in ids)
    return ages
    
if __name__ == '__main__':
    ages = get_all_colantuoni_ages()
