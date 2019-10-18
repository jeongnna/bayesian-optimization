from pickle import dump
from collections.abc import Mapping, Iterable

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler

from .acquisition import expected_improvement, pick_argmax_acq
from .utils import dist_to_bound, _sort_params, _params_to_2darray


class ParameterAcquirer:

    def __init__(self, surrogate, param_distributions,
                 acq_kwargs={}, random_state=None):
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

        self.surrogate = surrogate
        self.param_distributions = param_distributions
        self.acq_kwargs = acq_kwargs
        self.random_state = random_state

        self.pick_func = acq_kwargs.get('pick_func')

    def __iter__(self):
        sampler = ParameterSampler(self.param_distributions,
                                   self.acq_kwargs['n_candidates'],
                                   self.random_state)
        candidates = list(sampler)
        X_candidate = _params_to_2darray(candidates)

        #idx = _acq_pick(self.surrogate, X_candidate,
        #                self.acq_func, self.pick_func, self.n_pick,
        #                self.acq_kwargs, self.pick_kwargs)
        idx = self.pick_func(self.surrogate,
                               X_candidate,
                               self.acq_kwargs)

        for i in idx:
            yield candidates[i]

    def __len__(self):
        return self.acq_kwargs['n_pick']


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
                 n_initial_points=10, acq_kwargs={},
                 gp_params={}):

        self.param_distributions = param_distributions
        self.n_initial_points = n_initial_points
        self._n_initial_points = n_initial_points

        self.acq_kwargs = acq_kwargs
        self.acq_kwargs.setdefault('acq_func', expected_improvement)
        self.acq_kwargs.setdefault('pick_func', pick_argmax_acq)
        self.acq_kwargs.setdefault('n_candidates', 1000)
        self.acq_kwargs.setdefault('n_pick', 1)

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
        self._n_iter = self._n_iter // self.acq_kwargs['n_pick']  #!

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
                surrogate=gp,
                param_distributions=self.param_distributions,
                acq_kwargs=self.acq_kwargs,
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

    def _gp_fit(self, X, y, kernel=None):
        if kernel is None:
            kernel = Matern(nu=2.5)

        gp = GaussianProcessRegressor(kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=self.random_state)
        gp.set_params(**self.gp_params)
        gp.fit(X, y)

        return gp


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
