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
from numpy.random import randint


cdef class Sampler:
    """
    Wrapper for a gsl_rng object that exposes all sampling methods to Python.

    Useful for testing or writing pure Python programs.
    """
    def __init__(self, object seed=None):

        self.rng = gsl_rng_alloc(gsl_rng_mt19937)

        if seed is None:
            seed = randint(0, sys.maxint) & 0xFFFFFFFF
        gsl_rng_set(self.rng, seed)

    def __dealloc__(self):
        """
        Free GSL random number generator.
        """

        gsl_rng_free(self.rng)

    cpdef double gamma(self, double a, double b):
        return _sample_gamma(self.rng, a, b)

    cpdef double beta(self, double a, double b):
        return _sample_beta(self.rng, a, b)

    cpdef void dirichlet(self, double[::1] alpha, double[::1] out):
        _sample_dirichlet(self.rng, alpha, out)

    cpdef int categorical(self, double[::1] dist):
        return _sample_categorical(self.rng, dist)

    cpdef int searchsorted(self, double val, double[::1] arr):
        return _searchsorted(val, arr)

    cpdef int crt(self, int m, double r):
        return _sample_crt(self.rng, m, r)

    cpdef int sumcrt(self, int[::1] M, double[::1] R):
        return _sample_sumcrt(self.rng, M, R)

    cpdef int sumlog(self, int n, double p):
        return _sample_sumlog(self.rng, n, p)

    cpdef int truncated_poisson(self, double mu):
        return _sample_truncated_poisson(self.rng, mu)
