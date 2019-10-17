from scipy.stats import norm
from scipy.special import erfc
import numpy as np


def expected_improvement(gp, X, kwargs={}):
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


def _gradient_gp_mean(gp, x):
    x = x.reshape(1, -1)
    xX = np.concatenate([x, gp.X_train_], axis=0)

    y_pred = gp.predict(gp.X_train_)
    K, grad_K = gp.kernel_(xX, eval_gradient=True)
    grad_K = grad_K[0, 1:, :]
    K_inv = np.linalg.inv(K[1:, 1:])

    return grad_K.T.dot(K_inv).dot(y_pred)  # shape: (D, 1)


def _local_penalizer(X, loc, gp, L):
    loc = loc.reshape(1, -1)  # Argument for `gp.predict` must be a 2d array

    # M = max(gp.y_train_)
    M = max(gp.predict(X))
    d = np.linalg.norm(X - loc, axis=1)
    loc_mu, loc_std = gp.predict(loc, return_std=True)

    z = (L * d - M + loc_mu) / np.sqrt(2 * loc_std**2)

    return 0.5 * erfc(-z)  # shape: (N, )


def pick_argmax_acq(gp, X, kwargs={}):
    acq_func = kwargs.get('acq_func', expected_improvement)
    n_pick = kwargs.get('n_pick', 1)

    acq = acq_func(gp, X, kwargs)

    if n_pick == 1:
        return [np.nanargmax(acq)]

    selected = []
    for _ in range(n_pick):
        # M-step
        idx = np.nanargmax(acq)
        selected.append(idx)
        # P-step
        L = np.linalg.norm(_gradient_gp_mean(gp, X[idx]))
        acq *= _local_penalizer(X, X[idx], gp, L)

    return selected
