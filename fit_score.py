import numpy as np
import config as cfg

def loo_score(y_real,y_pred):
    """Compute score of LOO predictions. 
       For this purpose we ignore the fits taken with the first and last points left out,
       because these fits are then evaluated outside the range they were trained which can
       cause bad overfitting for functions like a sigmoid when the data is basically flat.
       This type of overfitting doesn't affect the fit on the whole data if we only consider the
       fit within the range it was trained on (which we do), so excluding the first and last points 
       should give a better estimate of the generalization error.
       Also ignore any NaNs in both sequences.
    """
    y_real = y_real[1:-1]
    y_pred = y_pred[1:-1]
    valid = ~np.isnan(y_real) & ~np.isnan(y_pred)
    y_real = y_real[valid]
    y_pred = y_pred[valid]
    if len(y_real) < 3:
        return None
    return cfg.score(y_real, y_pred)
