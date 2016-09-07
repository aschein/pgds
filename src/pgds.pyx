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

from libc.math cimport sqrt, exp, log, log1p
from mcmc_model cimport MCMCModel
from sample cimport _sample_gamma, _sample_dirichlet, _sample_beta, _sample_crt
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
        int V, T, K, diagonal, stationary, steady, binary, shrink
        double tau, gam, beta, eps, alpha, start_time
        double[::1] nu_K, xi_K, delta_T, zeta_T, P_K, Q_K
        double[:,::1] Pi_KK, Theta_TK, Phi_KV, shp_KK
        int[::1] Y_T, L_K
        int[:,::1] Y_TV, Y_KV, Y_TK, data_TV, L_TK, L_KK, H_KK
        unsigned int[::1] N_K, N_V
        unsigned int[:,:,::1] L_TKK, Y_TVK

    def __init__(self, int V, int T, int K, double eps=0.1, double alpha=1.,
                 double gam=1., double tau = 1., int stationary=1, int steady=1,
                 int diagonal=0, int binary=0, int shrink=0, object seed=None):

        self.V = V
        self.T = T
        self.K = K
        self.eps = eps
        self.alpha = alpha
        self.gam = gam
        self.tau = tau
        self.stationary = stationary
        self.steady = steady
        self.diagonal = diagonal
        self.shrink = shrink
        self.binary = binary

        self.beta = 1.
        self.nu_K = np.ones(K)
        self.xi_K = np.ones(K)
        self.delta_T = np.zeros(T)
        self.Pi_KK = np.ones((K, K))
        self.Theta_TK = np.ones((T, K))
        self.Phi_KV = np.ones((K, V))
        self.L_TKK = np.zeros((T, K, K), dtype=np.uint32)
        self.Y_TVK = np.zeros((T, V, K), dtype=np.uint32)

        self.zeta_T = np.zeros(T)
        self.shp_KK = np.ones((K, K))
        self.L_KK = np.zeros((K, K), dtype=np.int32)
        self.L_TK = np.zeros((T, K), dtype=np.int32)
        self.Y_TV = np.zeros((T, V), dtype=np.int32)
        self.Y_TK = np.zeros((T, K), dtype=np.int32)
        self.Y_KV = np.zeros((K, V), dtype=np.int32)
        self.Y_T = np.zeros(T, dtype=np.int32)
        self.P_K = np.zeros(K)
        self.N_K = np.zeros(K, dtype=np.uint32)
        self.N_V = np.zeros(V, dtype=np.uint32)

        self.Q_K = np.zeros(K)
        self.L_K = np.zeros(K, dtype=np.int32)
        self.H_KK = np.zeros((K, K), dtype=np.int32)

        self.data_TV = np.zeros((T, V), dtype=np.int32)

        self.start_time = time()

        super(PGDS, self).__init__(seed)

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.
        """
        return [('Y_TK', self.Y_TK, self._update_Y_TVK),
                ('Y_KV', self.Y_KV, lambda x: None),
                ('L_TKK', self.L_TKK, self._update_L_TKK),
                ('Phi_KV', self.Phi_KV, self._update_Phi_KV),
                ('Theta_TK', self.Theta_TK, self._update_Theta_TK),
                ('Pi_KK', self.Pi_KK, self._update_Pi_KK),
                ('delta_T', self.delta_T, self._update_delta_T),
                ('xi_K', self.xi_K, self._update_xi_K),
                ('nu_K', self.nu_K, self._update_nu_K),
                ('beta', self.beta, self._update_beta),
                ('gam', self.gam, self._update_gam),
                ('tau', self.tau, self._update_tau)]

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        cdef:
            int t, k1, k2, k, K
            double shape, scale, delta, eps
            gsl_rng * rng

        K = self.K
        eps = self.eps
        rng = self.rng

        for k in range(self.K):
            _sample_dirichlet(rng, np.ones(self.V) * eps, self.Phi_KV[k])
            assert self.Phi_KV[k, 0] >= 0
        
        if self.diagonal:
            for k1 in range(K):
                for k2 in range(K):
                    if k1 == k2:
                        self.Pi_KK[k1, k2] = 1
                    else:
                        self.Pi_KK[k1, k2] = 0
        else:
            if self.shrink:
                for k in range(K):
                    self.xi_K[k] = _sample_gamma(rng, eps, 1./eps)
                    self.nu_K[k] = _sample_gamma(rng, self.gam/K, 1./self.beta)
                    assert np.isfinite(self.nu_K).all()

                for k1 in range(K):
                    for k2 in range(K):
                        if k1 == k2:
                            self.shp_KK[k1, k2] = self.nu_K[k1] * self.xi_K[k1]
                        else:
                            self.shp_KK[k1, k2] = self.nu_K[k1] * self.nu_K[k2]
            else:
                self.shp_KK[:] = eps

            for k in range(K):
                _sample_dirichlet(rng, self.shp_KK[k], self.Pi_KK[k])
                assert self.Pi_KK[k, 0] >= 0

        self.tau = _sample_gamma(rng, 1., 1.)

        shape = self.tau * self.alpha
        scale = 1. / self.tau
        for k in range(K):
            self.Theta_TK[0, k] = _sample_gamma(rng, shape, scale)

        for t in range(1, self.T):
            for k in range(K):
                shape = self.tau * np.dot(self.Theta_TK[t-1], self.Pi_KK[:, k])
                self.Theta_TK[t, k] = _sample_gamma(rng, shape, scale)

        if self.stationary:
            delta = _sample_gamma(rng, eps, 1. / eps)
            if self.steady:
                self.delta_T[:] = delta
            else:
                self.delta_T[1:] = delta
        else:
            for t in range(1, self.T):
                self.delta_T[t] = _sample_gamma(rng, eps, 1. / eps)


    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """
        cdef:
            int t, v, k
            double mu, delta
            unsigned int y, y_tk, y_tkv

        self.Y_T[:] = 0
        self.Y_TV[:] = 0
        self.Y_TK[:] = 0
        self.Y_KV[:] = 0
        self.Y_TVK[:] = 0
        self.data_TV[:] = 0

        for t in range(self.T):
            if (t == 0) and not (self.stationary and self.steady):
                continue
            delta = self.delta_T[t]
            for k in range(self.K):
                mu = self.Theta_TK[t, k] * delta
                self.Y_TK[t, k] = y_tk = gsl_ran_poisson(self.rng, mu)
                if y_tk > 0:
                    self.Y_T[t] += y_tk
                    gsl_ran_multinomial(self.rng,
                                        self.V,
                                        y_tk,
                                        &self.Phi_KV[k, 0],
                                        &self.N_V[0])
                    
                    for v in range(self.V):
                        y_tvk = self.N_V[v]
                        if y_tvk > 0:
                            self.Y_TVK[t, v, k] = y_tvk
                            self.Y_KV[k, v] += y_tvk
                            self.Y_TV[t, v] += y_tvk
                            if t > 0:
                                self.data_TV[t, v] += y_tvk

    cdef void _init_state(self):
        """
        Initialize internal state.
        """

        pass

    cdef void _print_state(self):
        """
        Print internal state.
        """

        pass

    cdef void _update_Y_TVK(self):
        cdef:
            int t, v, y_tv
            unsigned int y_tvk

        self.Y_TK[:] = 0
        self.Y_KV[:] = 0
        self.Y_TVK[:] = 0

        for t in range(self.T):
            for v in range(self.V):
                
                y_tv = self.Y_TV[t, v]
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
                        self.Y_TVK[t, v, k] = y_tvk

    cdef void _update_L_TKK(self):
        cdef:
            double mu, r, norm, zeta
            int t, k, m, k1, k2, l_tk
            unsigned int l_tkk

        self.L_KK[:] = 0
        self.L_TK[:] = 0
        self.L_TKK[:] = 0

        if self.stationary and self.steady:
            self._update_zeta_T()
            zeta = self.zeta_T[0]
            assert zeta == self.zeta_T[self.T-1]

            for k in range(self.K):
                mu = zeta * self.Theta_TK[self.T-1, k]
                self.L_TK[self.T-1, k] = gsl_ran_poisson(self.rng, mu)

        for t in range(self.T-2, -1, -1):
            for k2 in range(self.K):
                m = self.Y_TK[t+1, k2] + self.L_TK[t+1, k2]
                
                norm = 0
                for k1 in range(self.K):
                    self.P_K[k1] = self.Theta_TK[t, k1] * self.Pi_KK[k1, k2]
                    norm += self.P_K[k1]
                r = self.tau * norm
    
                l_tk = _sample_crt(self.rng, m, r)
                assert l_tk >= 0

                if l_tk > 0:
                    for k1 in range(self.K):
                        self.P_K[k1] /= norm
                    assert np.allclose(np.sum(self.P_K), 1)

                    gsl_ran_multinomial(self.rng,
                                        self.K,
                                        <unsigned int> l_tk,
                                        &self.P_K[0],
                                        &self.N_K[0])
                    assert np.sum(self.N_K) == l_tk

                    for k1 in range(self.K):
                        l_tkk = self.N_K[k1]
                        self.L_TKK[t, k1, k2] = l_tkk
                        self.L_TK[t, k1] += l_tkk
                        self.L_KK[k1, k2] += l_tkk

        assert np.sum(self.L_TKK) == np.sum(self.L_KK) 
        if self.stationary == 0 or self.steady == 0:
            assert np.sum(self.L_TK[self.T-1]) == 0 
            assert np.sum(self.L_TKK[self.T-1]) == 0
            assert np.sum(self.L_TKK) == np.sum(self.L_TK)

    cdef void _update_zeta_T(self):
        cdef:
            int t, T
            double tmp, tau
            double[::1] delta_T

        T = self.T
        tau = self.tau
        delta_T = self.delta_T

        if self.stationary == 1 and self.steady == 1:
            self.zeta_T[:] = _simulate_zeta(self.tau, self.tau, self.delta_T[0])
        else:
            assert self.zeta_T[self.T-1] == 0  # dummy variable, always zero
            for t in range(self.T-2,-1,-1):
                tmp = (self.zeta_T[t+1] + self.delta_T[t+1]) / self.tau
                self.zeta_T[t] = self.tau * log1p(tmp)

    cdef void _update_Theta_TK(self):
        cdef:
            int k, t
            double shape, scale

        self._update_zeta_T()

        if self.delta_T[0] == 0:
            assert np.sum(self.Y_TK[0]) == 0

        if self.zeta_T[self.T-1] == 0:
            assert np.sum(self.L_TK[self.T-1]) == 0

        scale = 1. / (self.tau + self.zeta_T[0] + self.delta_T[0])
        for k in range(self.K):
            shape = self.tau * self.alpha + self.L_TK[0, k] + self.Y_TK[0, k]
            self.Theta_TK[0, k] = _sample_gamma(self.rng, shape, scale)
        
        for t in range(1, self.T):
            for k in range(self.K):
                shape = self.L_TK[t, k] + self.Y_TK[t, k] + \
                        self.tau * np.dot(self.Theta_TK[t-1], self.Pi_KK[:, k])
                scale = 1. / (self.tau + self.zeta_T[t] + self.delta_T[t])
                self.Theta_TK[t, k] = _sample_gamma(self.rng, shape, scale)

    cdef void _update_Phi_KV(self):
        cdef: 
            int k
            double eps
            gsl_rng * rng

        eps = self.eps
        rng = self.rng        
        for k in range(self.K):
            _sample_dirichlet(rng, np.add(eps, self.Y_KV[k]), self.Phi_KV[k])
            assert self.Phi_KV[k, 0] >= 0

    cdef void _update_Pi_KK(self):
        cdef:
            int k1, k2

        if not self.diagonal:
            if self.shrink:
                for k1 in range(self.K):
                    for k2 in range(self.K):
                        if k1 == k2:
                            self.shp_KK[k1, k2] = self.nu_K[k1] * self.xi_K[k1]
                        else:
                            self.shp_KK[k1, k2] = self.nu_K[k1] * self.nu_K[k2]
            else:
                assert self.shp_KK[0, 0] == self.eps  # all should be eps

            for k1 in range(self.K):
                _sample_dirichlet(self.rng,
                                  np.add(self.shp_KK[k1], self.L_KK[k1]),
                                  self.Pi_KK[k1])
                assert self.Pi_KK[k1, 0] >= 0
                assert np.isfinite(self.Pi_KK[k1]).all()

    cdef void _update_delta_T(self):
        cdef:
            int t
            double shape, scale

        if self.stationary:
            if self.steady:
                shape = self.eps + np.sum(self.Y_T)
                scale = 1. / (self.eps + np.sum(self.Theta_TK))
                self.delta_T[:] = _sample_gamma(self.rng, shape, scale)

            else:
                shape = self.eps + np.sum(self.Y_T[1:])
                scale = 1. / (self.eps + np.sum(self.Theta_TK[1:]))
                self.delta_T[1:] = _sample_gamma(self.rng, shape, scale)
        else:
            for t in range(1, self.T):
                shape = self.eps + self.Y_T[t]
                scale = 1. / (self.eps + np.sum(self.Theta_TK[t, :]))
                self.delta_T[t] = _sample_gamma(self.rng, shape, scale)

    cdef void _update_beta(self):
        cdef: 
            double shape, scale

        shape = self.eps + self.gam
        scale = 1. / (self.eps + np.sum(self.nu_K))
        self.beta = _sample_gamma(self.rng, shape, scale)

    cdef void _update_xi_K(self):
        cdef:
            int m, h_k, l_k, k, k2
            double r, shape, scale, shape1, shape2, nu, nu_k, xi_k, q_k, w_k
            double[::1] shp_K
            int[:, ::1] L_KK

        L_KK = np.sum(self.L_TKK[1:], axis=0, dtype=np.int32)

        nu = np.sum(self.nu_K)
        for k in range(self.K):
            xi_k = self.xi_K[k]
            nu_k = self.nu_K[k]

            shape1 = nu_k * (xi_k + nu - nu_k)
            shape2 = np.sum(L_KK[k])
            w_k = _sample_beta(self.rng, shape1, shape2)
            assert w_k > 0

            m = L_KK[k, k]
            r = nu_k * xi_k
            h_k = _sample_crt(self.rng, m, r)

            shape = self.eps + h_k
            scale = 1. / (self.eps - nu_k * log(w_k))
            self.xi_K[k] = xi_k = _sample_gamma(self.rng, shape, scale)

            # for k2 in range(self.K):
            #     if k2 == k:
            #         continue

            #     m = L_KK[k, k2]
            #     r = nu_k * self.nu_K[k2]
            #     h_k += _sample_crt(self.rng, m, r)

    cdef void _update_nu_K(self):
        pass

    cdef void _update_gam(self):
        pass

    cdef void _update_tau(self):
        pass


    cpdef void alt_schein(self, int num_samples):
        



