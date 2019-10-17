import numpy as np
import matplotlib.pyplot as plt

from .utils import dist_to_bound


def vis_tuning_progress(cv_results, greater_is_better=False,
                        valid_score=None, n_initial_points=0):

    score = np.array(cv_results['mean_test_score'])
    iteration = np.array(range(1, len(score) + 1))

    if greater_is_better:
        best_score = np.maximum.accumulate(score)
    else:
        best_score = np.minimum.accumulate(score)

    if valid_score is not None:
        if greater_is_better:
            valid_mask = score > valid_score
        else:
            valid_mask = score < valid_score
        iteration = iteration[valid_mask]
        score = score[valid_mask]
        best_score = best_score[valid_mask]

    plt.plot(iteration, score, 'xkcd:silver')
    plt.plot(iteration, best_score, 'xkcd:goldenrod')

    if n_initial_points:
        plt.axvline(x=n_initial_points)

    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.show()


def vis_acquisition(scv, param, step, figsize=(10, 10)):
    """
    Parameters
    ----------

    scv :

    param : string, name of the parameter you want to visualize

    step : integer

    figsize (optional) : tuple of length 2
    """
    plt.figure(figsize=figsize)

    # observations
    X, y = scv._Xy_observations(scv.cv_results_)
    X_obs = X[:step]
    y_obs = y[:step]
    param_pos = sorted(scv.param_distributions).index(param)
    X_obs = X_obs[:, [param_pos]]

    # plot observations
    plt.subplot(211)
    alpha = scv.gp_params.get('alpha', 1e-10)
    plt.errorbar(X_obs.reshape(-1), y_obs, yerr=np.sqrt(alpha),
        fmt='k.', label='Observation')

    # points for drawing curve
    param_bounds = dist_to_bound(scv.param_distributions)
    left, right = param_bounds[param]
    xnew = np.linspace(left, right, 1000)

    # derive GP & acquisition
    gp = scv._gp_fit(X_obs, y_obs)
    mu, sigma = gp.predict(xnew.reshape(-1, 1), return_std=True)
    mu = mu.reshape(-1)

    acq_func = scv.acq_kwargs['acq_func']
    pick_func = scv.acq_kwargs['pick_func']
    acq = acq_func(gp, xnew.reshape(-1, 1), scv.acq_kwargs)
    selected = pick_func(gp, xnew.reshape(-1, 1), scv.acq_kwargs)

    # plot GP
    plt.subplot(211)
    plt.plot(xnew, mu, 'C0', label='GP estimate')
    plt.fill(np.concatenate([xnew, xnew[::-1]]),
        np.concatenate([mu - 1.96 * sigma, (mu + 1.96 * sigma)[::-1]]),
        alpha=0.1, fc='C0', ec='None', label='GP uncertainty')
    for s in selected:
        plt.axvline(xnew[s], linestyle='--', color='C3')
    plt.ylabel('CV score')
    plt.legend(loc='lower right')

    # plot acquisition
    plt.subplot(212)
    plt.plot(xnew, acq, color='C0')
    for s in selected:
        plt.axvline(xnew[s], linestyle='--', color='C3')
    plt.ylabel('Acquisition')
    plt.xlabel(param)
