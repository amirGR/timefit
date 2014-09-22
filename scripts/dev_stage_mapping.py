from __future__ import print_function

import setup
from os.path import join
from dev_stages import dev_stages
from scalers import LogScaler
from project_dirs import results_dir


filename = join(results_dir(),'dev-stages.txt')
with open(filename,'w') as f:
    scaler = LogScaler()
    header = '{:<30} {:<8} {:<10} {:<10}'.format('Full Name', 'Label', 'Age', 'Log Scale')
    print(header, file=f)
    print(len(header)*'-', file=f)
    for stage in dev_stages:
        name = stage.name
        short_name = stage.short_name
        age = stage.central_age
        log_age = stage.scaled(scaler).central_age
        print('{:<30} {:<8} {:<10.3g} {:<10.3g}'.format(name, short_name, age, log_age), file=f)
