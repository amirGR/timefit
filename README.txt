This is a quick-start guide for the project, including how to get the code and run basic tasks.

I'm assuming you're running on cortex. If not, you should make sure to have the following installed:
* git (on windows I recommend gitextensions)
* python 2.7+ with the following libraries: numpy, scipy, matplotlib, sklearn, jinja2
 (on windows a great way to start is to install winpython)
 
===================================================
1) Getting the code and linking to the data files
===================================================
a) create a directory for the project (e.g. ~/projects/pyfit):
>> mkdir ~/projects/pyfit
>> cd ~/projects/pyfit

b) get the code from github to a subdirectory called "code":
>> git clone https://github.com/ronniemaor/HTR.git code

c) create a sibling data directory which links to the data files:
>> ln -s /cortex/ronniemaor/HTR data

===================================================
2) Creating a single fit
===================================================
The script do_one_fit.py fits a shape for a single gene/region:
>> cd <pyfit-dir>/code/scripts
>> python do_one_fit.py

This should create the file <pyfit-dir>/results/fit.png

For more options:
>> python do_one_fit.py --help

===================================================
3) Creating all fits and html files for a dataset/pathway/shape
===================================================
This is handled by the script compute_fits.py. To see the options:
>> cd <pyfit-dir>/code/scripts
>> python compute_fits.py --help

A good way to check you can run this:
>> python compute_fits.py --pathway test --shape poly1 --html ~/www/pyfit 
This will fit a 1st order polynomial on a "test" pathway containing two genes. 
It will create the fits under a cache directory and then use them to create the html files.
Assuming you're running on cortex, you can see the html at http://chechiklab.biu.ac.il/~yourlogin/pyfit/poly1/fits.html

Specifying a pathway:
======================
Some options worth noting when specifying a pathway (set of genes):
 * You can use one of the preconfigured sets, like 'serotonin'
 * 'all' will use all the genes in the dataset
 * Using any other string will try to load the gene names from a file at that path. 
   It will try the path as is, and relative to the data directory.
   Accepted formats:
     * Files ending in .mat that contain a matlab file with one variable that is a cell array of strings
	 * Files not ending in .mat are expected to be text files with gene names separated by whitespace (and possibly commas)

Note on parallelization:
=========================
Currently for each gene, regions are fit in parallel using N-1 processes, where N is the number of cores on your machine.
If you're fitting many genes, use the "--part k/n" option to split the work on several machines, e.g.
ctx03>> python compute_fits.py --pathway test --shape poly1 --part 1/3
ctx04>> python compute_fits.py --pathway test --shape poly1 --part 2/3
ctx05>> python compute_fits.py --pathway test --shape poly1 --part 3/3
Each of these will compute part of the genes and write the fits to files like e.g. <base filename>.pkl.2_of_3
Once you later run without --part, the package will automatically consolidate all the parts into <base filename>.

Specifying command-line arguments:
==================================
The programs provide a standard options parser, so the normal things will work and you can get help by running with --help.
Some things that are semi or non standard are worth mentioning:
* You can read arguments from a text file by using @, e.g.:
>> python compute_fits.py -v @my_fits.args 
will read a set of arguments from my_fits.args. 
Most options acccept one value. If they are specified twice then the last value specified is used. This behavior allows you to use the
file as defaults and override some of the values on the command line.
* lines starting with # in the text file are ignored (treated as comments).
