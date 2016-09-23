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
from time import time
from libc.math cimport log, log1p

from fatwalrus.mcmc_model cimport MCMCModel
from fatwalrus.sample cimport _sample_gamma, _sample_dirichlet, _sample_lnbeta, _sample_crt, _searchsorted
from lambertw cimport _simulate_zeta

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass

cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_rng_uniform(gsl_rng * r)
    double gsl_ran_gamma(gsl_rng * r, double a, double b)
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])

cdef class PGDS(MCMCModel):

    cdef:
        int T, V, K, shrink, stationary, steady
        double tau, gam, beta, eps, start_time
        double[::1] nu_K, xi_K, delta_T, zeta_T, P_K, Q_K
        double[:,::1] Pi_KK, Theta_TK, shp_KK, Phi_KV
        int[::1] Y_T
        int[:,::1] Y_TV, Y_TK, Y_KV, L_TK, L_KK
        int[:,:,::1] L_TKK
        unsigned int[::1] N_K, N_V

    def __init__(self, int T, int V, int K, double eps=0.1, double gam=10.,
                 double tau=1., int shrink=1, int stationary=0, int steady=0,
                 object seed=None):

        self.T = T
        self.V = V
        self.K = K
        self.eps = eps
        self.gam = gam
        self.tau = tau
        self.shrink = shrink
        self.stationary = stationary
        self.steady = steady
        if steady == 1 and stationary == 0:
            raise ValueError('Steady-state only valid for stationary model.')

        self.beta = 1.
        self.nu_K = np.zeros(K)
        self.xi_K = np.zeros(K)
        self.Pi_KK = np.zeros((K, K))
        self.Theta_TK = np.zeros((T, K))
        self.delta_T = np.zeros(T)
        self.Phi_KV = np.zeros((K, V))

        self.zeta_T = np.zeros(T)
        self.shp_KK = np.zeros((K, K))
        self.H_KK = np.zeros((K, K), dtype=np.int32)
        self.L_KK = np.zeros((K, K), dtype=np.int32)
        self.L_TK = np.zeros((T, K), dtype=np.int32)
        self.L_TKK = np.zeros((T, K, K), dtype=np.int32)
        self.Y_TV = np.zeros((T, V), dtype=np.int32)
        self.Y_TK = np.zeros((T, K), dtype=np.int32)
        self.Y_KV = np.zeros((K, V), dtype=np.int32)
        self.Y_T = np.zeros(T, dtype=np.int32)
        self.P_K = np.zeros(K)
        self.Q_K = np.zeros(K)
        self.N_K = np.zeros(K, dtype=np.uint32)
        self.N_V = np.zeros(V, dtype=np.uint32)

        self.start_time = time()

        super(PGDS, self).__init__(seed)

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.
        """
        variables = [('Y_KV', self.Y_KV, self._update_Y_TVK),
                     ('Y_TK', self.Y_TK, lambda x: None),
                     ('L_TKK', self.L_TKK, self._update_L_TKK),
                     ('Theta_TK', self.Theta_TK, self._update_Theta_TK),
                     ('Pi_KK', self.Pi_KK, self._update_Pi_KK),
                     ('Phi_KV', self.Phi_KV, self._update_Phi_KV),
                     ('nu_K', self.nu_K, self._update_nu_K),
                     ('beta', self.beta, self._update_beta)]

        if self.stationary == 0:
            variables += [('delta_T', self.delta_T, self._update_delta_T)]
        else:
            variables += [('delta_T', self.delta_T[0], self._update_delta_T)]

        if self.shrink == 1:
            variables += [('xi_K', self.xi_K[0], self._update_xi_K)]

        return variables

    cdef void _init_state(self):
        cdef:
            double tmp

        tmp = self.eps
        
        self.eps = 10.
        self._generate_state()
        
        self.eps = tmp

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        cdef:
            int K, k, k1, k2, t
            double eps, tau, shape
            gsl_rng * rng

        K = self.K
        eps = self.eps
        tau = self.tau
        rng = self.rng

        self.beta = _sample_gamma(rng, eps, 1. / eps)

        for k in range(K):
            self.nu_K[k] = _sample_gamma(rng, self.gam / K, 1. / self.beta)
            assert np.isfinite(self.nu_K[k])

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

        for k in range(K):
            shape = tau * self.nu_K[k]
            self.Theta_TK[0, k] = _sample_gamma(rng, shape, 1. / tau)
        for t in range(1, self.T):
            for k in range(K):
                shape = tau * np.dot(self.Theta_TK[t-1], self.Pi_KK[:, k])
                self.Theta_TK[t, k] = _sample_gamma(rng, shape, 1. / tau)

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
            unsigned int y_tk, y_tkv

        self.Y_T[:] = 0
        self.Y_TV[:] = 0
        self.Y_TK[:] = 0
        self.Y_KV[:] = 0

        for t in range(self.T):
            for k in range(self.K):
                mu = self.Theta_TK[t, k] * self.delta_T[t]
                y_tk = gsl_ran_poisson(self.rng, mu)
                self.Y_TK[t, k] = y_tk
                self.Y_T[t] += y_tk

                if y_tk > 0:
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
        self._update_L_TKK()

    cdef void _update_zeta_T(self):
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

    cdef void _update_Y_TVK(self):
        cdef:
            int t, v, k, _
            double norm
            unsigned int y_tv, y_tkv

        self.Y_TK[:] = 0
        self.Y_KV[:] = 0

        for t in range(self.T):
            for v in range(self.V):
                y_tv = self.Y_TV[t, v]
                
                if y_tv == 0:
                    continue

                if (y_tv < self.K) and (self.K < 40):  # use CDF/searchsorted method
                    self.P_K[0] = self.Theta_TK[t, 0] * self.Phi_KV[0, v]
                    for k in range(1, self.K):
                        self.P_K[k] = self.P_K[k-1] + self.Theta_TK[t, k] * self.Phi_KV[k, v]

                    norm = self.P_K[self.K-1]
                    for _ in range(y_tv):
                        k = _searchsorted(norm * gsl_rng_uniform(self.rng), self.P_K)
                        self.Y_TK[t, k] += 1
                        self.Y_KV[k, v] += 1

                else: # otherwise use conditional binom method via GSL's multinomial
                    for k in range(self.K):
                        self.P_K[k] = self.Theta_TK[t, k] * self.Phi_KV[k, v]

                    gsl_ran_multinomial(self.rng,
                                        self.K,
                                        y_tv,
                                        &self.P_K[0],
                                        &self.N_K[0])

                    for k in range(self.K):
                        y_tvk = self.N_K[k]
                        if y_tvk > 0:
                            self.Y_TK[t, k] += y_tvk
                            self.Y_KV[k, v] += y_tvk

    cdef void _update_L_TKK(self):
        cdef:
            int t, k, k1, m, _
            double norm, mu, zeta
            unsigned int l_tk, l_tkk
            list indices

        self.L_TK[:] = 0
        self.L_KK[:] = 0
        self.L_TKK[:] = 0

        if self.steady == 1:
            self._update_zeta_T()
            zeta = self.zeta_T[self.T-1]
            for k in range(self.K):
                mu = zeta * self.Theta_TK[self.T-1, k]
                self.L_TK[self.T-1, k] = gsl_ran_poisson(self.rng, mu)

        indices = range(self.K)
        np.random.shuffle(indices)
        for t in range(self.T-2, -1, -1):
            for k in indices:
                m = self.Y_TK[t+1, k] + self.L_TK[t+1, k]
                
                if m == 0:  # l_tk = 0 if m = 0
                    continue

                r = self.tau * np.dot(self.Theta_TK[t], self.Pi_KK[:, k])
                l_tk = _sample_crt(self.rng, m, r)
                assert l_tk >= 0

                if l_tk == 0:
                    continue

                if (l_tk < self.K) and (self.K < 40):  # use CDF/searchsorted method
                    self.P_K[0] = self.Theta_TK[t, 0] * self.Pi_KK[0, k]
                    for k1 in range(1, self.K):
                        self.P_K[k1] = self.P_K[k1-1] + self.Theta_TK[t, k1] * self.Pi_KK[k1, k]

                    norm = self.P_K[self.K-1]
                    for _ in range(l_tk):
                        k1 = _searchsorted(norm * gsl_rng_uniform(self.rng), self.P_K)
                        self.L_KK[k1, k] += 1
                        self.L_TK[t, k1] += 1
                        self.L_TKK[t, k1, k] += 1
                
                else:  # otherwise use conditional binom method via GSL's multinomial
                    for k1 in range(self.K):
                        self.P_K[k1] = self.Theta_TK[t, k1] * self.Pi_KK[k1, k]
                    
                    gsl_ran_multinomial(self.rng,
                                        self.K,
                                        l_tk,
                                        &self.P_K[0],
                                        &self.N_K[0])

                    for k1 in range(self.K):
                        l_tkk = self.N_K[k1]
                        if l_tkk > 0:
                            self.L_KK[k1, k] += l_tkk
                            self.L_TK[t, k1] += l_tkk
                            self.L_TKK[t, k1, k] = l_tkk

    cdef void _update_Theta_TK(self):
        cdef:
            int k, t
            double shape, scale, tau
            list indices

        self._update_zeta_T()

        tau = self.tau
        indices = range(self.K)
        np.random.shuffle(indices)
        for t in range(self.T):
            scale = 1. / (tau + self.zeta_T[t] + self.delta_T[t])
           
            for k in indices:
                if t == 0:
                    shape = tau * self.nu_K[k]
                else:
                    shape = tau * np.dot(self.Theta_TK[t-1], self.Pi_KK[:, k])
                shape += self.L_TK[t, k] + self.Y_TK[t, k]
                self.Theta_TK[t, k] = _sample_gamma(self.rng, shape, scale)

    cdef void _update_Phi_KV(self):
        cdef: 
            int k

        for k in range(self.K):
            _sample_dirichlet(self.rng,
                              np.add(self.eps, self.Y_KV[k]),
                              self.Phi_KV[k])
            assert self.Phi_KV[k, 0] >= 0

    cdef void _update_delta_T(self):
        cdef:
            int t
            double shape, scale

        if self.stationary == 0:
            for t in range(self.T):
                shape = self.eps + self.Y_T[t]
                scale = 1. / (self.eps + np.sum(self.Theta_TK[t, :]))
                self.delta_T[t] = _sample_gamma(self.rng, shape, scale)
        else:
            shape = self.eps + np.sum(self.Y_T)
            scale = 1. / (self.eps + np.sum(self.Theta_TK))
            self.delta_T[:] = _sample_gamma(self.rng, shape, scale)

    cdef void _update_Pi_KK(self):
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
            assert np.isfinite(self.Pi_KK[k]).all()

    cdef void _update_xi_K(self):

        cdef:
            int h, k, l_kk, l_k
            double eps, nu, c, nu_k, xi_k, lnq_k, tmp

        eps = self.eps
        nu = np.sum(self.nu_K)

        h = c = 0
        for k in range(self.K):
            nu_k = self.nu_K[k]
            xi_k = self.xi_K[k]

            l_kk = self.L_KK[k, k]
            if l_kk > 0:  # crt draw is 0 if l_kk = 0
                h += _sample_crt(self.rng, l_kk, nu_k * xi_k)

            l_k = np.sum(self.L_KK[k])
            if l_k > 0:  # q_k = 1 if l_k = 0, thus log(q_k) = 0 if l_k = 0
                lnq_k = _sample_lnbeta(self.rng, nu_k * (xi_k + nu - nu_k), l_k)
                c -= nu_k * lnq_k

        self.xi_K[:] = _sample_gamma(self.rng, eps + h, 1. / (eps + c))

    cdef void _update_nu_K(self):
        cdef:
            int k, k2, m_k, l_0k, l_kk, l_k, h_k
            double tau, gam_k, zeta, nu, nu_k, xi_k, shp, rte, c_k, tmp, r, lnq
            list indices

        self._update_zeta_T()

        tau = self.tau
        gam_k = self.gam / self.K
        zeta = tau * log1p((self.delta_T[0] + self.zeta_T[0]) / tau)
        nu = np.sum(self.nu_K)

        indices = range(self.K)
        np.random.shuffle(indices)
        for k in indices:
            nu_k = self.nu_K[k]
            m_k = self.Y_TK[0, k] + self.L_TK[0, k]
            l_0k = _sample_crt(self.rng, m_k, tau * nu_k)
            
            shp = gam_k + l_0k
            rte = self.beta + zeta
            
            if self.shrink == 1:
                xi_k = self.xi_K[k]
                l_kk = self.L_KK[k, k]
                l_k = np.sum(self.L_KK[k])
                
                h_k = c_k = 0

                if l_kk > 0:
                    h_k = _sample_crt(self.rng, l_kk, nu_k * xi_k)
                
                if l_k > 0:
                    tmp = (xi_k + nu - nu_k)
                    lnq = _sample_lnbeta(self.rng, nu_k * tmp, l_k)
                    assert np.isfinite(lnq)
                    c_k = -tmp * lnq
                
                for k2 in range(self.K):
                    if k2 == k:
                        continue
                    
                    r = nu_k * self.nu_K[k2]
                    h_k += _sample_crt(self.rng, self.L_KK[k, k2], r)
                    h_k += _sample_crt(self.rng, self.L_KK[k2, k], r)

                    l_k2 = np.sum(self.L_KK[k2])
                    if l_k2 > 0:
                        tmp = self.nu_K[k2] * (self.xi_K[k2] + nu - self.nu_K[k2])
                        lnq = _sample_lnbeta(self.rng, tmp, l_k2)
                        assert np.isfinite(lnq)
                        c_k -= self.nu_K[k2] * lnq

                shp += h_k
                rte += c_k
            assert np.isfinite(shp) and np.isfinite(rte)
            
            nu -= nu_k
            self.nu_K[k] = _sample_gamma(self.rng, shp, 1. / rte)
            nu += self.nu_K[k]


    cdef void _update_beta(self):
        cdef: 
            double shape, scale

        shape = self.eps + self.gam
        scale = 1. / (self.eps + np.sum(self.nu_K))
        self.beta = _sample_gamma(self.rng, shape, scale)


