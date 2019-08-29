from scipy.stats import norm
import numpy as np


def _pick_maximum(acq, n_pick=1, **kwargs):
    if not n_pick == 1:
        print('Error')
        return None
    return [np.nanargmax(acq)]


def expected_improvement(gp, X, **kwargs):
    """
    Parameters
    ----------

    # gp: sklearn.gaussian_process.GaussianProcessRegressor object

    # X: numpy.ndarray, shape of (n_candidates, n_params)

    # kwargs: Additional arguments
    """
    greater_is_better = kwargs.get('greater_is_better', True)

    mu, sigma = gp.predict(X, return_std=True)
    if len(mu.shape) > 1:
        if not mu.shape[1] == 1:
            print('Error')
            return None
        mu = mu.reshape(-1)

    with np.errstate(divide='ignore'):
        if greater_is_better:
            z = (mu - np.max(gp.y_train_)) / sigma
        else:
            z = (np.min(gp.y_train_) - mu) / sigma
        ei = sigma * z * norm.cdf(z) + sigma * norm.pdf(z)

        ei[sigma <= 0.0] = np.nan

    return ei  # (n_candidates, )
