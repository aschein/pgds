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

from libc.math cimport sqrt, exp, log
from mcmc_model cimport MCMCModel

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
        int V, T, K, diagonal, stationary, steady, binary
        double tau, gam, beta, epsilon, alpha, start_time
        double[::1] nu_K, xi_K, delta_T
        double[:,::1] Lambda_KK, Theta_TK, Phi_KV
        int[::1] Y_T
        int[:,::1] Y_KV, Y_TK, data_TV
        int[:,:,::1] L_TKK

    def __init__(self, int V, int T, int K,
                 double epsilon=0.1, double alpha=1.0, int diagonal=0,
                 int stationary=1, int steady=1, int binary=0, seed=None):

        self.V = V
        self.T = T
        self.K = K
        self.epsilon = epsilon
        self.alpha = alpha
        self.diagonal = diagonal
        self.stationary = stationary
        self.steady = steady
        self.binary = binary

        self.tau = 1.
        self.gam = 1.
        self.beta = 1.
        self.nu_K = np.empty(K)
        self.xi_K = np.empty(K)
        self.delta_T = np.empty(T)
        self.Lambda_KK = np.empty((K, K))
        self.Theta_TK = np.empty((T, K))
        self.Phi_KV = np.empty((K, V))
        self.L_TKK = np.empty((T, K, K), dtype=np.int32)
        self.Y_TVK = np.empty((T, V, K), dtype=np.int32)

        self.data_TV = np.empty((T, V), dtype=np.int32)

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
                ('Lambda_KK', self.Lambda_KK, self._update_Lambda_KK),
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

        pass

    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """

        pass

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
