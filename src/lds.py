import sys

from argparse import ArgumentParser
from path import path
from time import time

import cPickle as pickle
import numpy as np
import numpy.random as rn

from copy import deepcopy
from sklearn.base import BaseEstimator
from pykalman import KalmanFilter

from run_pgds import get_train_forecast_split, get_chain_num, save_forecast_eval, save_smoothing_eval


STATE_VARS = ['Pi_KK',
              'Sigma_KK',
              'Theta_TK',
              'Phi_VK',
              'D_VV',
              'alpha_K',
              'Beta_KK',
              'delta_T',
              'psi_T']


class LDS(BaseEstimator):
    """Linear Dynamical System"""

    def __init__(self, n_components=5, stationary=True, seed=None):
        self.n_components = n_components
        self.stationary = stationary
        self.total_itns = 0
        if seed is None:
            self.seed = rn.randint(100000)
        else:
            self.seed = seed

    def get_total_itns(self):
        return self.total_itns

    def get_state(self):
        state = {}
        for s in STATE_VARS:
            if hasattr(self, s):
                state[s] = np.copy(getattr(self, s))
        return state

    def set_state(self, state):
        assert all(s in state.keys() for s in STATE_VARS)
        V, K = state['Phi_VK'].shape
        T = state['Theta_TK'].shape[0]
        self.n_features = V
        self.n_timesteps = T
        self.n_components = K
        for s in state.keys():
            setattr(self, s, deepcopy(state[s]))

    def reconstruct(self, subs=None, partial_state={}):
        given = 'Theta_TK' in partial_state.keys()
        Theta_TK = self.Theta_TK if not given else partial_state['Theta_TK']

        given = 'Phi_VK' in partial_state.keys()
        Phi_VK = self.Phi_VK if not given else partial_state['Phi_VK']

        rates_TV = np.dot(Theta_TK, Phi_VK.T)
        return rates_TV if subs is None else rates_TV[subs]

    def forecast(self, n_timesteps=1):
        S = n_timesteps
        K = self.n_components
        Theta_SK = np.zeros((S, K))
        Theta_SK[0] = np.dot(self.Pi_KK, self.Theta_TK[-1])
        for s in xrange(1, S):
            Theta_SK[s] = np.dot(self.Pi_KK, Theta_SK[s-1])
        rates_SV = np.dot(Theta_SK, self.Phi_VK.T)
        return rates_SV

    def fit(self, data, initialize=True, num_itns=10, verbose=False):
        assert isinstance(data, np.ndarray) or isinstance(data, np.ma.MaskedArray)
        assert data.ndim == 2
        self.n_timesteps, self.n_features = data.shape
        if initialize:
            self._init_latent_params()
        self._update(data, num_itns=num_itns, verbose=verbose)
        return self

    def _init_latent_params(self):
        V = self.n_features
        K = self.n_components
        self.Pi_KK = rn.uniform(-0.99, 0.99, size=(K, K))
        self.Phi_VK = rn.uniform(-0.99, 0.99, size=(V, K))
        self.Sigma_KK = None
        self.D_VV = None
        self.alpha_K = None
        self.Beta_KK = None
        self.delta_T = None
        self.psi_T = None

    def _update(self, data, num_itns=10, verbose=False):
        em_vars = ['transition_covariance',
                   'transition_matrices',
                   'observation_matrices',
                   'observation_covariance',
                   'initial_state_covariance',
                   'initial_state_mean']

        if not self.stationary:
            em_vars += ['transition_offsets', 'observation_offsets']

        kf = KalmanFilter(transition_matrices=self.Pi_KK,
                          transition_covariance=self.Sigma_KK,
                          observation_matrices=self.Phi_VK,
                          observation_covariance=self.D_VV,
                          initial_state_mean=self.alpha_K,
                          initial_state_covariance=self.Beta_KK,
                          transition_offsets=self.delta_T,
                          observation_offsets=self.psi_T,
                          random_state=self.seed)

        start = time()
        kf = kf.em(data, em_vars=em_vars, n_iter=num_itns)
        self.total_itns += num_itns
        if verbose:
            end = time() - start
            print '%fs: em' % end

        start = time()
        self.Theta_TK = kf.smooth(data)[0]
        if verbose:
            end = time() - start
            print '%fs: smoothing' % end

        self.Pi_KK = kf.transition_matrices
        self.Sigma_KK = kf.transition_covariance
        self.Phi_VK = kf.observation_matrices
        self.D_VV = kf.observation_covariance
        self.alpha_K = kf.initial_state_mean
        self.Beta_KK = kf.initial_state_covariance
        self.delta_T = kf.transition_offsets
        self.psi_T = kf.observation_offsets


def main():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-k', '--n_components', type=int, default=100)
    p.add_argument('--stationary', action="store_true", default=False)
    p.add_argument('-s', '--seed', type=int, default=None)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('-n', '--num_itns', type=int, default=10)
    args = p.parse_args()

    data_dict = np.load(args.data)
    data = data_dict['data']
    mask = data_dict['mask']
    data_TV, data_SV = get_train_forecast_split(data, mask)
    T, V = data_TV.shape
    if data_SV is not None:
        S = data_SV.shape[0]
        mask_TV = mask[:T]
    else:
        S = 0
        mask_TV = mask
    masked_data = np.ma.array(data_TV, mask=mask_TV)

    args.out.makedirs_p()

    model = LDS(n_components=args.n_components, stationary=args.stationary, seed=args.seed)
    model.fit(masked_data, initialize=True, num_itns=args.num_itns, verbose=args.verbose)

    chain = get_chain_num(args.out)
    with open(args.out.joinpath('%d_params.p' % chain), 'wb') as params_file:
        pickle.dump(model.get_params(), params_file)

    state_name = '%d_state.npz' % chain
    np.savez(args.out.joinpath(state_name), **model.get_state())

    eval_path = args.out.joinpath('%d_smoothing_eval.txt' % chain)
    pred_path = args.out.joinpath('%d_smoothed.npz' % chain)
    save_smoothing_eval(masked_data, model, eval_path, pred_path)

    if S > 0:
        eval_path = args.out.joinpath('%d_forecast_eval.txt' % chain)
        pred_path = args.out.joinpath('%d_forecast.npz' % chain)
        save_forecast_eval(data_SV, model, eval_path, pred_path)

if __name__ == '__main__':
    main()
