from pickle import dump
from collections.abc import Mapping, Iterable

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler

from .acquisition import _pick_maximum, expected_improvement
from .utils import dist_to_bound, _sort_params, _params_to_2darray, _acq_pick


class ParameterAcquirer:

    def __init__(self, surrogate_model, param_distributions,
                 n_candidates, acq_func, acq_kwargs={},
                 n_pick=1, pick_func=_pick_maximum, pick_kwargs={},
                 random_state=None):
        if not isinstance(param_distributions, (Mapping, Iterable)):
            raise TypeError('Parameter distribution is not a dict or '
                            'a list ({!r})'.format(param_distributions))

        dist = param_distributions
        if not isinstance(dist, dict):
            raise TypeError('Parameter distribution is not a '
                            'dict ({!r})'.format(dist))
        for key in dist:
            if (not isinstance(dist[key], Iterable)
                    and not hasattr(dist[key], 'rvs')):
                raise TypeError('Parameter value is not iterable '
                                'or distribution (key={!r}, value={!r})'
                                .format(key, dist[key]))

        self.surrogate_model = surrogate_model
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates
        self.acq_func = acq_func
        self.acq_kwargs = acq_kwargs
        self.n_pick = n_pick
        self.pick_func = pick_func
        self.pick_kwargs = pick_kwargs
        self.random_state = random_state

    def __iter__(self):
        sampler = ParameterSampler(self.param_distributions,
                                   self.n_candidates,
                                   self.random_state)
        candidates = list(sampler)
        X_candidate = _params_to_2darray(candidates)

        idx = _acq_pick(self.surrogate_model, X_candidate,
                        self.acq_func, self.pick_func, self.n_pick,
                        self.acq_kwargs, self.pick_kwargs)

        for i in idx:
            yield candidates[i]

    def __len__(self):
        return self.n_pick


class EnhancedSearchCV(BaseSearchCV):

    def __init__(self, estimator, scoring=None, n_jobs=None, iid='deprecated',
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=True,
                 random_state=None, n_iter=10, file=None):
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state
        self.n_iter = n_iter
        self._n_iter = n_iter
        self.file = file

        self._X_init = []
        self._y_init = []

        super().__init__(
            estimator=estimator, scoring=scoring, n_jobs=n_jobs, iid=iid,
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
            error_score=error_score, return_train_score=return_train_score)

    def _save_cv_results(self, results, file):
        if file is None:
            return

        with open(file, 'wb') as f:
            dump(results, f)

    def set_initial_points(self, X, y):
        """

        Parameters
        ----------

        X : list of dictionary, each dict has shape of {'param1': value1, 'param2': value2, ...}

        y : array-like, shape = (n_points, ) or (n_points, n_score_metric)
        """

        if not isinstance(X, list):
            print('Error: `X` must be a list.')
            return
        else:
            for x in X:
                if not isinstance(x, dict):
                    print('Error: all elements of `X` must be dictionaries.')
                    return

        if not isinstance(y, (list, np.ndarray, pd.core.series.Series)):
            print('Error: all elements of `X` must be dictionaries.')
            return

        n_init = len(X)
        if not n_init == len(y):
            print('Error: length of X and y must be equal.')
            return

        self._n_iter -= n_init

        X = _sort_params(X)
        self._X_init = X
        self._y_init = y
        self._n_init = n_init


class BayesianSearchCV(EnhancedSearchCV):

    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 n_jobs=None, iid='deprecated', refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score=np.nan,
                 return_train_score=False, file=None,
                 greater_is_better=False, n_initial_points=10,
                 n_candidates=1000, acq_func='ei', acq_kwargs={},
                 n_pick=1, pick_func=_pick_maximum, pick_kwargs={},
                 gp_params={}):
        if acq_func == 'ei':
            acq_func = expected_improvement

        acq_kwargs['greater_is_better'] = greater_is_better

        self.param_distributions = param_distributions
        self.greater_is_better = greater_is_better
        self.n_initial_points = n_initial_points
        self._n_initial_points = n_initial_points

        self.n_candidates = n_candidates
        self.acq_func = acq_func
        self.acq_kwargs = acq_kwargs

        self.n_pick=n_pick
        self.pick_func=pick_func
        self.pick_kwargs=pick_kwargs

        self.gp_params = gp_params

        super().__init__(
            estimator=estimator, scoring=scoring, n_jobs=n_jobs, iid=iid,
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
            error_score=error_score, return_train_score=return_train_score,
            random_state=random_state, n_iter=n_iter, file=file)

    def _run_search(self, evaluate_candidates):
        results = {}

        # adjust number of iterations
        n_init = len(self._X_init)
        self._n_initial_points -= n_init
        self._n_iter -= max(0, self._n_initial_points)

        # Randomly sample initial points
        for _ in range(self._n_initial_points):
            results = evaluate_candidates(ParameterSampler(
                self.param_distributions, 1,
                random_state=self.random_state))

            self._save_cv_results(results, self.file)

        # Bayesian optimization
        for _ in range(self._n_iter):
            X_obs, y_obs = self._Xy_observations(results)
            gp = self._gp_fit(X_obs, y_obs)

            results = evaluate_candidates(ParameterAcquirer(
                surrogate_model=gp,
                param_distributions=self.param_distributions,
                n_candidates=self.n_candidates,
                acq_func=self.acq_func,
                acq_kwargs=self.acq_kwargs,
                n_pick=self.n_pick,
                pick_func=self.pick_func,
                pick_kwargs=self.pick_kwargs,
                random_state=self.random_state))

            self._save_cv_results(results, self.file)

    def _Xy_observations(self, cv_results):
        X_obs = self._X_init.copy()
        y_obs = self._y_init.copy()
        if cv_results:
            X_obs.extend(cv_results['params'])
            y_obs.extend(cv_results['mean_test_score'])
        X_obs = _params_to_2darray(_sort_params(X_obs))
        y_obs = np.array(y_obs)
        return (X_obs, y_obs)

    def _gp_fit(self, X, y):
        matern = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=matern,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=self.random_state)
        gp.set_params(**self.gp_params)
        gp.fit(X, y)
        return gp

    def vis_acquisition(self, param, step, figsize=(10, 10)):
        """
        Parameters
        ----------

        param : string, name of the parameter you want to visualize

        step : integer

        figsize (optional) : tuple of length 2
        """
        plt.figure(figsize=figsize)

        # observations
        X, y = self._Xy_observations(self.cv_results_)
        X_obs = X[:step]
        y_obs = y[:step]
        param_pos = sorted(self.param_distributions).index(param)
        X_obs = X_obs[:, [param_pos]]

        # plot observations
        plt.subplot(211)
        alpha = self.gp_params.get('alpha', 1e-10)
        plt.errorbar(X_obs.reshape(-1), y_obs, yerr=np.sqrt(alpha),
            fmt='k.', label='Observation')

        # points for drawing curve
        param_bounds = dist_to_bound(self.param_distributions)
        left, right = param_bounds[param]
        xnew = np.linspace(left, right, 1000)

        # derive GP & acquisition
        gp = self._gp_fit(X_obs, y_obs)
        mu, sigma = gp.predict(xnew.reshape(-1, 1), return_std=True)
        mu = mu.reshape(-1)

        acq = self.acq_func(gp, xnew.reshape(-1, 1), **self.acq_kwargs)

        # plot GP
        plt.subplot(211)
        plt.plot(xnew, mu, 'C0', label='GP estimate')
        plt.fill(np.concatenate([xnew, xnew[::-1]]),
            np.concatenate([mu - 1.96 * sigma, (mu + 1.96 * sigma)[::-1]]),
            alpha=0.1, fc='C0', ec='None', label='GP uncertainty')
        plt.axvline(xnew[np.nanargmax(acq)], linestyle='--', color='C3')
        plt.ylabel('CV score')
        plt.legend(loc='lower right')

        # plot acquisition
        plt.subplot(212)
        plt.plot(xnew, acq, color='C0')
        plt.axvline(xnew[np.nanargmax(acq)], linestyle='--', color='C3')
        plt.ylabel('Acquisition')
        plt.xlabel(param)


class RandomSearchCV(EnhancedSearchCV):

    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True, random_state=None, file=None):

        self.param_distributions = param_distributions

        super().__init__(
            estimator=estimator, scoring=scoring, n_jobs=n_jobs, iid=iid,
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
            error_score=error_score, return_train_score=return_train_score,
            random_state=random_state, n_iter=n_iter, file=file)

    def _run_search(self, evaluate_candidates):
        for _ in range(self._n_iter):
            results = evaluate_candidates(ParameterSampler(
                self.param_distributions, 1,
                random_state=self.random_state))

            self._save_cv_results(results, self.file)
