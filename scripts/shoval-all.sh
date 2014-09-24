#!/bin/sh

python compute_fits.py @shoval.args --scaling log --correlations
python compute_fits.py @shoval.args --scaling log 
python compute_fits.py @shoval.args --scaling log --from_age postnatal --correlations
python compute_fits.py @shoval.args --scaling log --from_age postnatal
python compute_fits.py @shoval.args --from_age postnatal --correlations
python compute_fits.py @shoval.args --from_age postnatal

