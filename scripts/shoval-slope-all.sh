#!/bin/sh

python compute_fits.py @shoval-slope.args --priors sigslope$1 --scaling log --correlations
python compute_fits.py @shoval-slope.args --priors sigslope$1 --scaling log 
python compute_fits.py @shoval-slope.args --priors sigslope$1 --scaling log --from_age postnatal --correlations
python compute_fits.py @shoval-slope.args --priors sigslope$1 --scaling log --from_age postnatal
python compute_fits.py @shoval-slope.args --priors sigslope$1 --from_age postnatal --correlations
python compute_fits.py @shoval-slope.args --priors sigslope$1 --from_age postnatal

