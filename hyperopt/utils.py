import scipy as sp
import numpy as np


def bound_to_dist(param_bounds):
    param_dists = {}

    for name, bound in param_bounds.items():
        if isinstance(bound[0], int):
            dist = sp.stats.randint(bound[0], bound[1] + 1)
        elif isinstance(bound[0], float):
            dist = sp.stats.uniform(bound[0], bound[1] - bound[0])
        param_dists[name] = dist

    return param_dists


def dist_to_bound(param_dists):
    param_bounds = {}

    for name, dist in param_dists.items():
        if isinstance(dist.dist, sp.stats._discrete_distns.randint_gen):
            left = dist.args[0]
            right = dist.args[1] - 1
        elif isinstance(dist.dist, sp.stats._continuous_distns.uniform_gen):
            left = dist.args[0]
            right = left + dist.args[1]
        param_bounds[name] = (left, right)

    return param_bounds


def _sort_params(params):
    return [{key: d[key] for key in sorted(d)} for d in params]


def _params_to_2darray(params):
    n_params = len(params[0])
    params = np.array([list(x.values()) for x in params])
    return params.reshape(-1, n_params)


def _acq_pick(surrogate_model, X_candidate, acq_func, pick_func,
              n_pick=1, acq_kwargs={}, pick_kwargs={}):
    acq = acq_func(surrogate_model, X_candidate, **acq_kwargs)
    return pick_func(acq, n_pick, **pick_kwargs)
