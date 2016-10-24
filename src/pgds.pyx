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
from fatwalrus.sample cimport _sample_gamma, _sample_dirichlet, _sample_lnbeta,\
                              _sample_crt, _sample_trunc_poisson

from lambertw cimport _simulate_zeta
from impute import init_missing_data

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


cdef class PGDS(MCMCModel):

    cdef:
        int T, V, K, P, shrink, stationary, steady, binary, y_
        double tau, gam, beta, eps, theta_, nu_
        double[::1] nu_K, xi_K, delta_T, zeta_T, P_K, lnq_K, Theta_T, shp_V
        double[:,::1] Pi_KK, Theta_TK, shp_KK, Phi_KV, R_TK
        int[::1] Y_T, vals_P, L_K
        int[:,::1] Y_TV, Y_TK, Y_KV, L_TK, L_KK, H_KK, mask_TV, data_TV, subs_P2
        int[:,:,::1] L_TKK
        unsigned int[::1] N_K, N_V

    def __init__(self, int T, int V, int K, double eps=0.1, double gam=10.,
                 double tau=1., int shrink=1, int stationary=0, int steady=0,
                 int binary=0, object seed=None):

        super(PGDS, self).__init__(seed)
        
        self.T = self.param_list['T'] = T
        self.V = self.param_list['V'] = V
        self.K = self.param_list['K'] = K
        self.eps = self.param_list['eps'] = eps
        self.gam = self.param_list['gam'] = gam
        self.tau = self.param_list['tau'] = tau
        self.binary = self.param_list['binary'] = binary
        self.shrink = self.param_list['shrink'] = shrink
        self.stationary = self.param_list['stationary'] = stationary
        self.steady = self.param_list['steady'] = steady
        if steady == 1 and stationary == 0:
            raise ValueError('Steady-state only valid for stationary model.')

        self.beta = 1.
        self.nu_K = np.zeros(K)
        self.xi_K = np.zeros(K)
        self.Pi_KK = np.zeros((K, K))
        self.Theta_TK = np.zeros((T, K))
        self.Theta_T = np.zeros(T)
        self.theta_ = 0
        self.delta_T = np.zeros(T)
        self.Phi_KV = np.zeros((K, V))

        self.zeta_T = np.zeros(T)
        self.shp_KK = np.zeros((K, K))
        self.R_TK = np.zeros((T, K))
        self.H_KK = np.zeros((K, K), dtype=np.int32)
        self.L_K = np.zeros(K, dtype=np.int32)
        self.L_KK = np.zeros((K, K), dtype=np.int32)
        self.L_TK = np.zeros((T, K), dtype=np.int32)
        self.L_TKK = np.zeros((T, K, K), dtype=np.int32)
        self.Y_TV = np.zeros((T, V), dtype=np.int32)
        self.Y_TK = np.zeros((T, K), dtype=np.int32)
        self.Y_KV = np.zeros((K, V), dtype=np.int32)
        self.Y_T = np.zeros(T, dtype=np.int32)
        self.y_ = 0
        self.P_K = np.zeros(K)
        self.lnq_K = np.zeros(K)
        self.N_K = np.zeros(K, dtype=np.uint32)
        self.N_V = np.zeros(V, dtype=np.uint32)
        self.shp_V = np.zeros(V)

        self.data_TV = np.zeros((T, V), dtype=np.int32)
        self.mask_TV = np.zeros((T, V), dtype=np.int32)

        self.P = 0  # placeholder
        self.subs_P2 = np.zeros((self.P, 2), dtype=np.int32)
        self.vals_P = np.zeros(self.P, dtype=np.int32)

        

    def fit(self, data, num_itns=1000, verbose=True, initialize=True, burnin={}):
        if not isinstance(data, np.ma.core.MaskedArray):
            data = np.ma.array(data, mask=None)
        
        assert data.shape == (self.T, self.V)
        assert (data >= 0).all()
        if self.binary == 1:
            assert (data <= 1).all()

        filled_data = data.astype(np.int32).filled(fill_value=-1)
        subs = filled_data.nonzero()
        self.vals_P = filled_data[subs]  # missing values will be -1
        self.subs_P2 = np.array(zip(*subs), dtype=np.int32)
        self.P = self.vals_P.shape[0]

        if self.binary == 0:
            # filled_data[filled_data == -1] = 0
            # self.Y_TV = np.ascontiguousarray(filled_data, dtype=np.int32)
            self.Y_TV = np.ascontiguousarray(init_missing_data(data))
            self.Y_T = np.sum(self.Y_TV, axis=1, dtype=np.int32)
            self.y_ = np.sum(self.Y_T)

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

        delta_T = np.array(self.delta_T)
        if 'delta_T' in partial_state.keys():
            delta_T = partial_state['delta_T']

        if not subs:
            rates_TV = np.einsum('tk,t,kv->tv', Theta_TK, delta_T, Phi_KV)
            if self.binary == 1:
                rates_TV = -np.expm1(-rates_TV)
            return rates_TV

        else:
            Theta_PK = Theta_TK[subs[0]]
            delta_P = delta_T[subs[0]]
            Phi_KP = Phi_KV[:, subs[1]]
            rates_P = np.einsum('pk,kp,p->p', Theta_PK, Phi_KP, delta_P)
            if self.binary == 1:
                rates_P = -np.expm1(-rates_P)
            return rates_P

    cdef void _forecast(self, double[:,:,::1] rates_CSV, str mode='arithmetic'):
        cdef:
            int C, S, K, c, s, k, v
            double[:,::1] Theta_SK
            double[::1] delta_S, shp_K
            double rte, delta, tau, mu_csv

        C = rates_CSV.shape[0]
        S = rates_CSV.shape[1]
        K = self.K

        delta_S = np.zeros(S)
        if self.stationary == 1:
            delta_S[:] = self.delta_T[0]
        else:
            delta_S[:] = np.mean(self.delta_T[self.T-2:])

        Theta_SK = np.zeros((S, K))

        tau = self.tau
        rte = 1. / tau
        for c in range(C):
            for s in xrange(S):
                if s == 0:
                    shp_K = tau * np.dot(self.Theta_TK[self.T-1], self.Pi_KK)
                else:
                    shp_K = tau * np.dot(Theta_SK[s-1], self.Pi_KK)
                
                delta = delta_S[s]
                for k in range(K):

                    if mode == 'arithmetic':
                        Theta_SK[s, k] = shp_K[k] / rte
                    elif mode == 'geometric':
                        Theta_SK[s, k] = exp(gsl_sf_psi(shp_K[k]) - log(rte))
                    else:
                        Theta_SK[s, k] = _sample_gamma(self.rng, shp_K[k], rte)

                for v in range(self.V):
                    mu_csv = delta * np.dot(Theta_SK[s], self.Phi_KV[:, v])
                    if self.binary == 1:
                        rates_CSV[c, s, v] = -expm1(-mu_csv)
                    else:
                        rates_CSV[c, s, v] = mu_csv

    def forecast(self, n_timesteps=1, n_chains=1, mode='arithmetic'):
        assert mode in ['arithmetic', 'geometric', 'sample']
        if mode != 'sample':
            assert n_chains == 1

        rates_CSV = np.zeros((n_chains, n_timesteps, self.V))
        self._forecast(rates_CSV=rates_CSV, mode=mode)
        return rates_CSV[0] if n_chains == 1 else rates_CSV

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.
        """
        variables = [('Y_KV', self.Y_KV, self._update_Y_TVK),
                     ('Y_TK', self.Y_TK, lambda x: None),
                     ('L_TKK', self.L_TKK, self._update_L_TKK)]

        if self.stationary == 0:
            variables += [('delta_T', self.delta_T, self._update_delta_T)]
        else:
            variables += [('delta_T', self.delta_T[0], self._update_delta_T)]

        variables += [('Phi_KV', self.Phi_KV, self._update_Phi_KV),
                      ('Theta_TK', self.Theta_TK, self._update_Theta_TK),
                      ('Pi_KK', self.Pi_KK, self._update_Pi_KK)]

        if self.shrink == 0:
            variables += [('beta', self.beta, self._update_beta),
                          ('nu_K', self.nu_K, self._update_nu_K)]
        else:
            variables += [('beta', self.beta, self._update_beta),
                          ('H_KK', self.H_KK, self._update_H_KK_and_lnq_K),
                          ('lnq_K', self.lnq_K, lambda x: None),
                          ('nu_K', self.nu_K, self._update_nu_K),
                          ('xi_K', self.xi_K[0], self._update_xi_K)]
        return variables

    cdef void _init_state(self):
        cdef:
            int K, k, k2, t
            double eps, tau, shape, theta_tk, nu_k
            gsl_rng * rng

        K = self.K
        eps = self.eps
        tau = self.tau
        rng = self.rng

        self.delta_T[:] = 1.
        self._update_zeta_T()

        self.beta = 1.
        self.tau = 1.

        self.xi_K[:] = 1.

        self.nu_ = 0
        for k in range(K):
            nu_k = _sample_gamma(rng, self.gam / float(K), 1. / self.beta)
            self.nu_K[k] = nu_k
            self.nu_ += nu_k

        for k in range(K):
            for k2 in range(K):
                self.shp_KK[k, k2] = _sample_gamma(rng, 1., 1.)
            _sample_dirichlet(rng, self.shp_KK[k], self.Pi_KK[k])
            assert np.isfinite(self.Pi_KK[k]).all()
            assert self.Pi_KK[k, 0] >= 0

        self.theta_ = 0
        self.Theta_T[:] = 0
        for t in range(self.T):
            for k in range(self.K):
                theta_tk = _sample_gamma(rng, 1., 1.)
                self.Theta_TK[t, k] = theta_tk
                self.Theta_T[t] += theta_tk
                self.theta_ += theta_tk

        for k in range(self.K):
            _sample_dirichlet(rng, np.ones(self.V), self.Phi_KV[k])
            assert self.Phi_KV[k, 0] >= 0

    cdef void _print_state(self):
        cdef:
            int l
            double theta, delta

        l = np.sum(self.L_KK)
        theta = np.mean(self.Theta_TK)
        delta = np.mean(self.delta_T)
        print 'ITERATION %d: total aux counts: %d\t \
               mean theta: %.4f\t mean delta: %f\n' % \
               (self.total_itns, l, theta, delta) 

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        cdef:
            int K, k, k1, k2, t
            double eps, tau, shape, theta_tk, nu_k
            gsl_rng * rng

        K = self.K
        eps = self.eps
        tau = self.tau
        rng = self.rng

        self.beta = _sample_gamma(rng, eps, 1. / eps)

        self.nu_ = 0
        for k in range(K):
            nu_k = _sample_gamma(rng, self.gam / K, 1. / self.beta)
            assert np.isfinite(nu_k)
            self.nu_K[k] = nu_k
            self.nu_ += nu_k

        self.xi_K[:] = 1
        if self.shrink == 1:
            self.xi_K[:] = _sample_gamma(rng, eps, 1. / eps)
            assert np.isfinite(self.xi_K).all()
        
        for k in range(K):
            self.shp_KK[k, :] = self.eps
            if self.shrink == 1:
                self.shp_KK[k, k] = self.nu_K[k] * self.xi_K[k]
                for k2 in range(K):
                    if k == k2:
                        continue
                    self.shp_KK[k, k2] = self.nu_K[k] * self.nu_K[k2]
            _sample_dirichlet(rng, self.shp_KK[k], self.Pi_KK[k])
            assert np.isfinite(self.Pi_KK[k]).all()
            assert self.Pi_KK[k, 0] >= 0

        self.theta_ = 0
        self.Theta_T[:] = 0
        for k in range(K):
            shape = tau * self.nu_K[k]
            theta_tk = _sample_gamma(rng, shape, 1. / tau)
            self.Theta_TK[0, k] = theta_tk
            self.Theta_T[0] += theta_tk
            self.theta_ += theta_tk

        for t in range(1, self.T):
            for k in range(K):
                shape = tau * np.dot(self.Theta_TK[t-1], self.Pi_KK[:, k])
                theta_tk = _sample_gamma(rng, shape, 1. / tau)
                self.Theta_TK[t, k] = theta_tk
                self.Theta_T[t] += theta_tk
                self.theta_ += theta_tk

        for k in range(self.K):
            _sample_dirichlet(rng, np.ones(self.V) * eps, self.Phi_KV[k])
            assert self.Phi_KV[k, 0] >= 0

        if self.stationary == 0:
            for t in range(self.T):
                self.delta_T[t] = _sample_gamma(rng, eps, 1. / eps)
        else:
            self.delta_T[:] = _sample_gamma(rng, eps, 1. / eps)
        self._update_zeta_T()

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
        self.Y_T[:] = 0
        self.Y_TV[:] = 0
        self.Y_TK[:] = 0
        self.Y_KV[:] = 0

        for t in range(self.T):
            for k in range(self.K):
                mu_tk = self.Theta_TK[t, k] * self.delta_T[t]
                y_tk = gsl_ran_poisson(self.rng, mu_tk)

                if y_tk > 0:

                    self.Y_TK[t, k] = y_tk
                    self.Y_T[t] += y_tk
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

        if self.y_ > 0:
            subs = np.nonzero(self.Y_TV)
            self.subs_P2 = np.array(zip(*subs), dtype=np.int32)
            self.P = self.subs_P2.shape[0]
            if self.binary == 0:
                self.vals_P = np.array(self.Y_TV)[subs]
        else:
            self.P = 0

        self._update_L_TKK()
        self._update_H_KK_and_lnq_K()

    cdef void _update_zeta_T(self) nogil:
        cdef:
            int t
            double tmp

        if self.steady == 1:
            self.zeta_T[:] = _simulate_zeta(self.tau, self.tau, self.delta_T[1])
        else:
            self.zeta_T[self.T-1] = 0
            for t in range(self.T-2,-1,-1):
                tmp = (self.zeta_T[t+1] + self.delta_T[t+1]) / self.tau
                self.zeta_T[t] = self.tau * log1p(tmp)

    cdef void _update_Y_TVK(self) nogil:
        cdef:
            int p, t, v, k, y_tv
            double norm, u, mu_tv
            unsigned int y_tvk

        self.Y_TK[:] = 0
        self.Y_KV[:] = 0
        for p in range(self.P):
            t = self.subs_P2[p, 0]
            v = self.subs_P2[p, 1]
            y_tv = self.Y_TV[t, v]
            
            if (self.vals_P[p] == -1) or (self.binary == 1):
                self.y_ -= y_tv
                self.Y_T[t] -= y_tv

                mu_tv = 0
                for k in range(self.K):
                    mu_tv += self.Theta_TK[t, k] * self.Phi_KV[k, v]
                mu_tv *= self.delta_T[t]

                if (self.vals_P[p] == -1) and (self.total_itns > 500):
                    y_tv = gsl_ran_poisson(self.rng, mu_tv)
                else:
                    y_tv = _sample_trunc_poisson(self.rng, mu_tv)

                self.Y_TV[t, v] = y_tv
                self.Y_T[t] += y_tv
                self.y_ += y_tv

            if y_tv == 0:
                continue

            for k in range(self.K):
                self.P_K[k] = self.Theta_TK[t, k] * self.Phi_KV[k, v]

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

    cdef void _update_L_TKK(self):
        cdef:
            int t, k, k1, m, l_tk
            double norm, mu, zeta
            unsigned int l_tkk

        self.L_K[:] = 0
        self.L_KK[:] = 0
        self.L_TK[:] = 0
        self.L_TKK[:] = 0

        self.R_TK = self.tau * np.dot(self.Theta_TK, self.Pi_KK)

        with nogil:
            if self.steady == 1:
                zeta = self.zeta_T[self.T-1]
                for k in range(self.K):
                    mu = zeta * self.Theta_TK[self.T-1, k]
                    self.L_TK[self.T-1, k] = gsl_ran_poisson(self.rng, mu)

            for t in range(self.T-2, -1, -1):
                for k in range(self.K):
                    m = self.Y_TK[t+1, k] + self.L_TK[t+1, k]

                    if m == 0:  # l_tk = 0 if m = 0
                        continue

                    l_tk = _sample_crt(self.rng, m, self.R_TK[t, k])
                    # assert (l_tk >= 0) and (l_tk <= m)

                    if l_tk == 0:
                        continue

                    for k1 in range(self.K):
                        self.P_K[k1] = self.Theta_TK[t, k1] * self.Pi_KK[k1, k]
                    
                    gsl_ran_multinomial(self.rng,
                                        self.K,
                                        <unsigned int> l_tk,
                                        &self.P_K[0],
                                        &self.N_K[0])

                    for k1 in range(self.K):
                        l_tkk = self.N_K[k1]
                        if l_tkk > 0:
                            self.L_K[k1] += l_tkk
                            self.L_KK[k1, k] += l_tkk
                            self.L_TK[t, k1] += l_tkk
                            self.L_TKK[t, k1, k] = l_tkk

    cdef void _update_Theta_TK(self) nogil:
        cdef:
            int k, k1, t
            double shp, sca, tau, theta_tk

        tau = self.tau

        self.theta_ = 0
        self.Theta_T[:] = 0
        for t in range(self.T):
            sca = 1. / (tau + self.zeta_T[t] + self.delta_T[t])
           
            for k in range(self.K):
                if t == 0:
                    shp = tau * self.nu_K[k]
                else:
                    shp = 0
                    for k1 in range(self.K):
                        shp += self.Theta_TK[t-1, k1] * self.Pi_KK[k1, k]
                    shp *= tau
                shp += self.L_TK[t, k] + self.Y_TK[t, k]
                
                theta_tk = _sample_gamma(self.rng, shp, sca)
                self.Theta_TK[t, k] = theta_tk
                self.Theta_T[t] += theta_tk
                self.theta_ += theta_tk

    cdef void _update_Phi_KV(self) nogil:
        cdef: 
            int k, v
            double eps
            gsl_rng * rng

        eps = self.eps
        rng = self.rng

        for k in range(self.K):
            for v in range(self.V):
                self.shp_V[v] = eps + self.Y_KV[k, v]
            _sample_dirichlet(rng, self.shp_V, self.Phi_KV[k])
            # assert self.Phi_KV[k, 0] >= 0

    cdef void _update_delta_T(self) nogil:
        cdef:
            int t
            double shp, rte

        if self.stationary == 0:
            for t in range(self.T):
                shp = self.eps + self.Y_T[t]
                rte = self.eps + self.Theta_T[t]
                self.delta_T[t] = _sample_gamma(self.rng, shp, 1. / rte)
        else:
            shp = self.eps + self.y_
            rte = self.eps + self.theta_
            self.delta_T[:] = _sample_gamma(self.rng, shp, 1. / rte)

        self._update_zeta_T()

    cdef void _update_Pi_KK(self) nogil:
        cdef:
            int k, k2
            double nu_k

        for k in range(self.K):
            if self.shrink == 1:
                nu_k = self.nu_K[k]
                self.shp_KK[k, k] = nu_k * self.xi_K[k] + self.L_KK[k, k]
                for k2 in range(self.K):
                    if k == k2:
                        continue
                    self.shp_KK[k, k2] = nu_k * self.nu_K[k2] + self.L_KK[k, k2]
            else:
                for k2 in range(self.K):
                    self.shp_KK[k, k2] = self.eps + self.L_KK[k, k2]

        for k in range(self.K):
            _sample_dirichlet(self.rng, self.shp_KK[k], self.Pi_KK[k])
            # assert np.isfinite(self.Pi_KK[k]).all()

    cdef void _update_H_KK_and_lnq_K(self):
        cdef:
            int k, k2, l_k
            double nu_k, xi_k, tmp, r

        self.lnq_K[:] = 0
        for k in range(self.K):
            nu_k = self.nu_K[k]
            xi_k = self.xi_K[k]
            r = nu_k * xi_k
            self.H_KK[k, k] = _sample_crt(self.rng, self.L_KK[k, k], r)
            assert self.H_KK[k, k] >= 0

            for k2 in range(self.K):
                if k2 == k:
                    continue
                r = nu_k * self.nu_K[k2]
                self.H_KK[k, k2] = _sample_crt(self.rng, self.L_KK[k, k2], r)
                assert self.H_KK[k, k2] >= 0

            l_k = self.L_K[k]
            if l_k > 0:
                tmp = (xi_k  + self.nu_ - nu_k)
                self.lnq_K[k] = _sample_lnbeta(self.rng, nu_k * tmp, l_k)
                assert np.isfinite(self.lnq_K[k])
                assert self.lnq_K[k] <= 0

    cdef void _update_nu_K(self):
        cdef:
            int k, k2, m_k, l_0k
            double tau, zeta, gam_k, nu_k, shp, rte
            # list indices

        tau = self.tau
        zeta = tau * log1p((self.delta_T[0] + self.zeta_T[0]) / tau)
        gam_k = self.gam / self.K

        # indices = range(self.K)
        # np.random.shuffle(indices)

        for k in range(self.K):
            nu_k = self.nu_K[k]
            m_k = self.Y_TK[0, k] + self.L_TK[0, k]
            l_0k = _sample_crt(self.rng, m_k, tau * nu_k)
            assert l_0k >= 0
            
            shp = gam_k + l_0k
            rte = self.beta + zeta

            if self.shrink == 1:
                shp += self.H_KK[k, k]
                rte += (self.xi_K[k] + self.nu_ - nu_k) * (-self.lnq_K[k])
                for k2 in range(self.K):
                    if k2 == k:
                        continue
                    shp += self.H_KK[k, k2] + self.H_KK[k2, k]
                    rte += self.nu_K[k2] * (-self.lnq_K[k2])
                
            assert np.isfinite(shp) and np.isfinite(rte)
            assert shp >= 0 and rte >= 0
            
            self.nu_ -= nu_k
            nu_k = _sample_gamma(self.rng, shp, 1. / rte)
            self.nu_K[k] = nu_k
            self.nu_ += nu_k

    cdef void _update_xi_K(self) nogil:

        cdef:
            int k
            double shp, rte
        
        shp = rte = self.eps
        for k in range(self.K):
            shp += self.H_KK[k, k]
            rte -= self.nu_K[k] * self.lnq_K[k]
        # assert np.isfinite(shp) and np.isfinite(rte)

        self.xi_K[:] = _sample_gamma(self.rng, shp, 1. / rte)

    cdef void _update_beta(self) nogil:
        cdef: 
            double shp, rte

        shp = self.eps + self.gam
        rte = self.eps + self.nu_
        self.beta = _sample_gamma(self.rng, shp, 1. / rte)
