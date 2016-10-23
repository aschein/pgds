#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: extra_link_args = ['-lgsl', '-lgslcblas']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import sys
import numpy as np
cimport numpy as np
from libc.math cimport exp, log, log1p, expm1

from fatwalrus.mcmc_model cimport MCMCModel
from fatwalrus.sample cimport _sample_gamma, _sample_dirichlet, \
                              _sample_crt, _sample_trunc_poisson
# from impute import init_missing_data

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass

cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_rng_uniform(gsl_rng * r)
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])

cdef extern from "gsl/gsl_sf_psi.h" nogil:
    double gsl_sf_psi(double)


cdef class GPDPFA(MCMCModel):

    cdef:
        int T, V, K, P, binary, stationary, y_, l_
        double gam, beta, e, f, alpha, theta_, lambda_
        double[::1] lambda_K, delta_T, P_K, Theta_T, Theta_K, shp_V
        double[:,::1] Theta_TK, Phi_KV, zeta_TK
        int[::1] vals_P, L_K, Y_K
        int[:,::1] Y_TV, Y_TK, Y_KV, L_TK, subs_P2
        unsigned int[::1] N_K, N_V

    def __init__(self, int T, int V, int K, double e=0.1, double f=0.1, 
                 double alpha=0.01, int stationary=0, int binary=0,
                 object seed=None):

        super(GPDPFA, self).__init__(seed)
        
        self.T = self.param_list['T'] = T
        self.V = self.param_list['V'] = V
        self.K = self.param_list['K'] = K
        self.e = self.param_list['e'] = e
        self.f = self.param_list['f'] = f
        self.alpha = self.param_list['alpha'] = alpha
        self.stationary = self.param_list['stationary'] = stationary
        self.binary = self.param_list['binary'] = binary

        self.lambda_ = 0
        self.lambda_K = np.zeros(K)
        self.delta_T = np.zeros(T)
        self.Theta_TK = np.zeros((T, K))
        self.Theta_T = np.zeros(T)
        self.Theta_K = np.zeros(K)
        self.theta_ = 0
        self.Phi_KV = np.zeros((K, V))
        self.gam = 1.
        self.beta = 1.

        self.zeta_TK = np.zeros((T, K))
        self.l_ = 0
        self.L_K = np.zeros(K, dtype=np.int32)
        self.L_TK = np.zeros((T, K), dtype=np.int32)
        self.Y_TV = np.zeros((T, V), dtype=np.int32)
        self.Y_TK = np.zeros((T, K), dtype=np.int32)
        self.Y_KV = np.zeros((K, V), dtype=np.int32)
        self.Y_K = np.zeros(K, dtype=np.int32)
        self.y_ = 0
        self.P_K = np.zeros(K)
        self.N_K = np.zeros(K, dtype=np.uint32)
        self.N_V = np.zeros(V, dtype=np.uint32)
        self.shp_V = np.zeros(V)

        self.P = 0  # placeholder
        self.subs_P2 = np.zeros((self.P, 2), dtype=np.int32)
        self.vals_P = np.zeros(self.P, dtype=np.int32)
        

    def fit(self, data, num_itns=1000, verbose=True, initialize=True, burnin={}):
        if not isinstance(data, np.ma.core.MaskedArray):
            data = np.ma.array(data, mask=None)
        
        assert data.shape == (self.T-1, self.V)
        assert (data >= 0).all()
        if self.binary == 1:
            assert (data <= 1).all()

        filled_data = data.astype(np.int32).filled(fill_value=-1)
        subs = filled_data.nonzero()
        self.vals_P = filled_data[subs]  # missing values will be -1
        subs_P2 = np.array(zip(*subs), dtype=np.int32)
        subs_P2[:, 0] += 1  # time indices are offset by 1
        self.subs_P2 = subs_P2
        self.P = self.vals_P.shape[0]

        if self.binary == 0:
            Y_TV = np.zeros((self.T, self.V), dtype=np.int32)
            filled_data[filled_data == -1] = 0
            Y_TV[1:] = filled_data
            # Y_TV[1:] = init_missing_data(data)
            self.Y_TV = Y_TV
            self.y_ = np.sum(self.Y_TV)

        if initialize:
            self._init_state()

        self._update(num_itns=num_itns, verbose=int(verbose), burnin=burnin)

    def reconstruct(self, subs=(), partial_state={}):
        Theta_TK = np.array(self.Theta_TK)
        if 'Theta_TK' in partial_state.keys():
            Theta_TK = partial_state['Theta_TK']

        Phi_KV = np.array(self.Phi_KV)
        if 'Phi_KV' in partial_state.keys():
            Phi_KV = partial_state['Phi_KV']

        lambda_K = np.array(self.lambda_K)
        if 'lambda_K' in partial_state.keys():
            lambda_K = partial_state['lambda_K']

        if not subs:
            rates_TV = np.einsum('tk,k,kv->tv', Theta_TK[1:], lambda_K, Phi_KV)
            if self.binary == 1:
                rates_TV = -np.expm1(-rates_TV)
            return rates_TV

        else:
            Theta_PK = Theta_TK[1:][subs[0]]
            Phi_KP = Phi_KV[:, subs[1]]
            rates_P = np.einsum('pk,k,kp->p', Theta_PK, lambda_K, Phi_KP)
            if self.binary == 1:
                rates_P = -np.expm1(-rates_P)
            return rates_P

    def forecast(self, n_timesteps=1, n_chains=1, mode='arithmetic'):
        assert mode in ['arithmetic', 'geometric', 'sample']
        if mode != 'sample':
            assert n_chains == 1

        rates_CSV = np.zeros((n_chains, n_timesteps, self.V))
        self._forecast(rates_CSV=rates_CSV, mode=mode)
        return rates_CSV[0] if n_chains == 1 else rates_CSV

    cdef void _forecast(self, double[:,:,::1] rates_CSV, str mode='arithmetic'):
        cdef:
            int C, S, K, c, s, k, v
            double[:,::1] Theta_SK
            double[::1] delta_S, lambda_K
            double[:] phi_K
            double sca_s, shp_sk, mu_csv

        C = rates_CSV.shape[0]
        S = rates_CSV.shape[1]
        K = self.K

        lambda_K = self.lambda_K

        delta_S = np.zeros(S)
        if self.stationary == 1:
            delta_S[:] = self.delta_T[0]
        else:
            delta_S[:] = np.mean(self.delta_T[self.T-2:])

        Theta_SK = np.zeros((S, K))

        for c in range(C):
            for s in range(S):
                sca_s = 1. / delta_S[s]
                for k in range(K):
                    if s == 0:
                        shp_sk = self.Theta_TK[self.T-1, k]
                    else:
                        shp_sk = Theta_SK[s-1, k]
                    
                    if mode == 'arithmetic':
                        Theta_SK[s, k] = shp_sk * sca_s
                    elif mode == 'geometric':
                        Theta_SK[s, k] = exp(gsl_sf_psi(shp_sk) + log(sca_s))
                    else:
                        Theta_SK[s, k] = _sample_gamma(self.rng, shp_sk, sca_s)
                    
                for v in range(self.V):
                    phi_K = self.Phi_KV[:, v]
                    mu_csv = np.einsum('k,k,k->', Theta_SK[s], lambda_K, phi_K)
                    if self.binary == 1:
                        rates_CSV[c, s, v] = -expm1(-mu_csv)
                    else:
                        rates_CSV[c, s, v] = mu_csv

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.
        """
        variables = [('Y_KV', self.Y_KV, self._update_Y_TVK),
                     ('Y_TK', self.Y_TK, lambda x: None),
                     ('Phi_KV', self.Phi_KV, self._update_Phi_KV),
                     ('L_TK', self.L_TK, self._update_L_TK),
                     ('Theta_TK', self.Theta_TK, self._update_Theta_TK),
                     ('lambda_K', self.lambda_K, self._update_lambda_K),
                     ('beta', self.beta, self._update_beta),
                     ('gam', self.gam, self._update_gam)]

        if self.stationary == 0:
            variables += [('delta_T', self.delta_T, self._update_delta_T)]
        else:
            variables += [('delta_T', self.delta_T[0], self._update_delta_T)]

        return variables

    cdef void _print_state(self):
        cdef:
            int l
            double theta, lam

        l = np.sum(self.L_TK)
        theta = np.mean(self.Theta_TK)
        lam = np.mean(self.lambda_K)
        print 'ITERATION %d: total aux counts: %d\t \
               mean theta: %.4f\t mean lambda: %f\n' % \
               (self.total_itns, l, theta, lam)

    cdef void _init_state(self):
        cdef:
            double tmp_e, tmp_f, tmp_a

        tmp_e = self.e
        tmp_f = self.f
        tmp_a = self.alpha

        self.e = 10.
        self.f = 0.1
        self.alpha = 2.

        self._generate_state()

        self.e = tmp_e
        self.f = tmp_f
        self.alpha = tmp_a

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        cdef:
            int K, k, t
            double e, f, lambda_k, theta_tk, shp_tk
            gsl_rng * rng

        K = self.K
        e = self.e
        f = self.f
        rng = self.rng

        self.gam = _sample_gamma(rng, e, 1. / f)
        self.beta = _sample_gamma(rng, e, 1. / f)

        self.lambda_ = 0
        for k in range(K):
            lambda_k = _sample_gamma(rng, self.gam / K, 1. / self.beta)
            assert np.isfinite(lambda_k)
            self.lambda_K[k] = lambda_k
            self.lambda_ += lambda_k

        if self.stationary == 1:
            self.delta_T[:] = _sample_gamma(rng, e, 1. / f)
        else:
            for t in range(self.T):
                self.delta_T[t] = _sample_gamma(rng, e, 1. / f)

        self.theta_ = 0
        self.Theta_K[:] = 0
        self.Theta_T[:] = 0
        for t in range(self.T):
            for k in range(K):
                shp_tk = self.alpha if t == 0 else self.Theta_TK[t-1, k]
                theta_tk = _sample_gamma(rng, shp_tk, 1. / self.delta_T[t])
                self.Theta_TK[t, k] = theta_tk
                self.Theta_T[t] += theta_tk
                self.Theta_K[k] += theta_tk
                self.theta_ += theta_tk

        for k in range(self.K):
            _sample_dirichlet(rng, np.ones(self.V) * e, self.Phi_KV[k])
            assert self.Phi_KV[k, 0] >= 0

    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """
        cdef:
            int t, v, k
            unsigned int y_tk, y_tvk
            double mu_tk
            tuple subs

        self.y_ = 0
        self.Y_K[:] = 0
        self.Y_KV[:] = 0
        self.Y_TK[:] = 0
        self.Y_TV[:] = 0

        with nogil:
            for t in range(1, self.T):
                for k in range(self.K):
                    mu_tk = self.Theta_TK[t, k] * self.lambda_K[k]
                    y_tk = gsl_ran_poisson(self.rng, mu_tk)

                    if y_tk > 0:
                        self.Y_TK[t, k] = y_tk
                        self.Y_K[k] += y_tk
                        self.y_ += y_tk

                        gsl_ran_multinomial(self.rng,
                                            self.V,
                                            y_tk,
                                            &self.Phi_KV[k, 0],
                                            &self.N_V[0])
                        
                        for v in range(self.V):
                            y_tvk = self.N_V[v]
                            if y_tvk > 0:
                                self.Y_KV[k, v] += y_tvk
                                self.Y_TV[t, v] += y_tvk

        self.P = 0
        if self.y_ > 0:
            subs = np.nonzero(self.Y_TV)
            self.subs_P2 = np.array(zip(*subs), dtype=np.int32)
            self.P = self.subs_P2.shape[0]
            if self.binary == 0:
                self.vals_P = np.array(self.Y_TV)[subs]

        self._update_L_TK()

    cdef void _update_Y_TVK(self) nogil:
        cdef:
            int p, t, v, k, y_tv
            double mu_tv
            unsigned int y_tvk

        self.Y_TK[:] = 0
        self.Y_KV[:] = 0
        self.Y_K[:] = 0

        for p in range(self.P):
            t = self.subs_P2[p, 0]
            v = self.subs_P2[p, 1]
            y_tv = self.Y_TV[t, v]
            
            if (self.vals_P[p] == -1) or (self.binary == 1):
                self.y_ -= y_tv

                mu_tv = 0
                for k in range(self.K):
                    mu_tv += self.Theta_TK[t, k] * self.Phi_KV[k, v] * \
                             self.lambda_K[k]

                if (self.vals_P[p] == -1) and (self.total_itns > 0):
                    y_tv = gsl_ran_poisson(self.rng, mu_tv)
                else:
                    y_tv = _sample_trunc_poisson(self.rng, mu_tv)

                self.Y_TV[t, v] = y_tv
                self.y_ += y_tv

            if y_tv == 0:
                continue

            for k in range(self.K):
                self.P_K[k] = self.Theta_TK[t, k] * self.Phi_KV[k, v] * \
                              self.lambda_K[k]

            gsl_ran_multinomial(self.rng,
                                self.K,
                                <unsigned int> y_tv,
                                &self.P_K[0],
                                &self.N_K[0])

            for k in range(self.K):
                y_tvk = self.N_K[k]
                if y_tvk > 0:
                    self.Y_TK[t, k] += y_tvk
                    self.Y_KV[k, v] += y_tvk
                    self.Y_K[k] += y_tvk

    cdef void _update_L_TK(self):
        cdef:
            int t, k, m_tk, l_tk
            double tmp

        self.l_ = 0
        self.L_K[:] = 0
        self.L_TK[:] = 0
        self.zeta_TK[:] = 0

        for t in range(self.T-2, -1, -1):
            for k in range(self.K):
                m_tk = self.Y_TK[t+1, k] + self.L_TK[t+1, k]
                l_tk = _sample_crt(self.rng, m_tk, self.Theta_TK[t, k])
                assert l_tk >= 0
                if l_tk > 0:
                    self.L_TK[t, k] = l_tk
                    self.L_K[k] += l_tk
                    self.l_ += l_tk

                tmp = self.zeta_TK[t+1, k] + self.lambda_K[k]
                self.zeta_TK[t, k] = log1p(tmp / self.delta_T[t+1])

    cdef void _update_Theta_TK(self) nogil:
        cdef:
            int t, k, T
            double shp_tk, rte_tk, lambda_k, theta_tk
            int[:,::1] L_TK, Y_TK
            gsl_rng * rng

        rng = self.rng
        L_TK = self.L_TK
        Y_TK = self.Y_TK

        self.theta_ = 0
        self.Theta_K[:] = 0
        self.Theta_T[:] = 0

        for k in range(self.K):
            lambda_k = self.lambda_K[k]
            for t in range(self.T):
                if t == 0:
                    shp_tk = self.alpha + L_TK[t, k]
                    rte_tk = self.delta_T[t] + self.zeta_TK[t, k]
                else:
                    #  L_TK[-1] and zeta_TK[-1] should always be 0
                    shp_tk = self.Theta_TK[t-1, k] + L_TK[t, k] + Y_TK[t, k]
                    rte_tk = self.delta_T[t] + self.zeta_TK[t, k] + lambda_k
                theta_tk = _sample_gamma(rng, shp_tk, 1. / rte_tk)
                self.Theta_TK[t, k] = theta_tk
                self.Theta_T[t] += theta_tk
                self.Theta_K[k] += theta_tk
                self.theta_ += theta_tk

    cdef void _update_Phi_KV(self):
        cdef: 
            int k, v

        for k in range(self.K):
            for v in range(self.V):
                self.shp_V[v] = self.e + self.Y_KV[k, v]
            _sample_dirichlet(self.rng, self.shp_V, self.Phi_KV[k])
            assert self.Phi_KV[k, 0] >= 0

    cdef void _update_delta_T(self) nogil:
        cdef:
            int t, K
            double shp, rte

        K = self.K
        
        if self.stationary == 1:
            shp = self.e + self.theta_ - self.Theta_T[self.T-1] + self.alpha * K
            rte = self.f + self.theta_
            self.delta_T[:] = _sample_gamma(self.rng, shp, 1. / rte)
        else:
            for t in range(self.T):
                if t == 0:
                    shp = self.e + self.alpha * K
                else:
                    shp = self.e + self.Theta_T[t-1]
                rte = self.f + self.Theta_T[t]
                self.delta_T[t] = _sample_gamma(self.rng, shp, 1. / rte)

    cdef void _update_lambda_K(self):
        cdef:
            int k
            double shp_k, rte_k

        self.lambda_ = 0
        for k in range(self.K):
            shp_k = self.gam / self.K + self.Y_K[k]
            rte_k = self.beta + self.Theta_K[k] - self.Theta_TK[0, k]
            self.lambda_K[k] = _sample_gamma(self.rng, shp_k, 1. / rte_k)
            self.lambda_ += self.lambda_K[k]

    cdef void _update_beta(self) nogil:
        cdef: 
            double shp, rte

        shp = self.e + self.gam
        rte = self.f + self.lambda_
        self.beta = _sample_gamma(self.rng, shp, 1. / rte)

    cdef void _update_gam(self) nogil:
        cdef: 
            int k
            double shp, rte

        shp = self.e
        for k in range(self.K):
            shp += _sample_crt(self.rng, self.Y_K[k], self.gam / self.K)

        rte = 0
        for k in range(self.K):
            rte += log1p((self.Theta_K[k] - self.Theta_TK[0, k]) / self.beta)
        rte /= self.K
        rte += self.f

        self.gam = _sample_gamma(self.rng, shp, 1. / rte)
