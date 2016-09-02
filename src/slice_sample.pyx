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
from libc.math cimport log


cdef class SliceSampler:
    """
    Interface for a model-like object that implements slice-sampling. 

    _slice_sample(...) calls _logprob(x) (which must be implemented).

    This is intended as an instructive example of how to implement a nogil
    slice-sampler not (necessarily) an interface to inherit.
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

    cdef double _logprob(self, double x, int which_var=0) nogil:
        """
        Log-probability function f(x).

        Arguments:
            x -- Value of parameter.
            which_var -- Identifies which parameter is being sampled and thus 
                         which version of f(x) should be output.
        """
        pass

    cdef double _slice_sample(self,
                              double x_init,
                              double x_min,
                              double x_max,
                              int which_var=0,
                              int max_iter=1000) nogil:
        """
        Slice sample x given a log-probability function f(x) which represents a 
        closure over data sufficient statistics and hyperparameters.  

        Calls self._logprob(double x).

        Arguments:
            x_init -- Current value of parameter to be slice-sampled.
            x_min -- Min allowable value of x
            x_max -- Max allowable value of x
            which_var -- Identifies which parameter is being sampled.  This is
                         passed to _logprob(...) which implements different 
                         log-probability functions for the different parameters.
            max_iter -- Max number of iterations before terminating.

        """
        cdef:
            int _
            double z_init, z, e, u, diff, x

        z_init = self._logprob(x_init, which_var)
        z = z_init - gsl_ran_exponential(self.rng, 1.)
        for _ in range(max_iter):
            diff = x_max - x_min
            if diff < 0:
                return x_min - 1

            x = gsl_rng_uniform(self.rng) * diff + x_min

            if self._logprob(x, which_var) >= z:
                return x

            elif diff == 0:
                return x_min - 1

            if x < x_init:
                x_min = x
            else:
                x_max = x

        return x

cdef class PoissonSliceSampler(SliceSampler):
    """
    Simple example of a model-like object that inherits SliceSampler and
    implements the _lobprob(...) function.
    """
    def __init__(self, seed=None):
        self.data_size = 1
        self.data_sum = 1
        super(PoissonSliceSampler, self).__init__(seed)

    cdef double _logprob(self, double x, int which_var=0) nogil:
        """
        Log-probability:

                    f(x) = Pois(data | x) Gamma(x | 1, 1).
        """
        # return self.data_sum * log(x) - self.data_size * x - x
        return log(gsl_ran_poisson_pdf(self.data_sum, self.data_size * x)) + \
               log(gsl_ran_gamma_pdf(x, 1, 1))

    cpdef double slice_sample(self,
                              int[::1] data,
                              double x_init=1,
                              list window=[0, 1e7]):
        """
        Wrapper for _slice_sample(...) which takes data as an argument, uses a
        list variable to represent the bounds, and exposes everything to Python.

        This method first updates the model's data sufficient statistics which
        are called by self._logprob(...) before it calls _slice_sample(...).

        Arguments:
            data -- Array of counts that are assumed drawn iid from Pois(x).
            x_init -- Current value of rate parameter to be slice-sampled.
            window -- The bounds [x_min, x_max]
        """

        cdef:
            double x_min, x_max

        x_min = window[0]
        x_max = window[1]

        self.data_size = data.shape[0]
        self.data_sum = np.sum(data)

        return self._slice_sample(x_init=x_init,
                                  x_min=x_min,
                                  x_max=x_max)

cdef class GammaSliceSampler(SliceSampler):
    """
    Simple example two-parameter model that inherits SliceSampler and implements
    the _lobprob(...) function as well as a wrapper to _slice_sample(...).
    """
    def __init__(self, init_shape=1, init_scale=1, seed=None):
        self.shape = init_shape
        self.scale = init_scale
        self.data_size = 1
        self.data_sum = 1
        super(GammaSliceSampler , self).__init__(seed)

    cdef double _logprob(self, double x, int which_var=0) nogil:
        """
        If the shape parameter x is being slice-sampled, then this returns:
                        
                        f(x) = Gamma(data | x, scale) Gamma(x | 1, 1)

        If the scale parameter x is being slice-sample, then this returns:

                        f(x) = Gamma(data | shape, x) Gamma(x | 1, 1)
        """
        cdef:
            double out
        if which_var == 0:
            out = log(gsl_ran_gamma_pdf(self.data_sum,
                                        self.data_size * x, 
                                        self.scale))
        else:
            out = log(gsl_ran_gamma_pdf(self.data_sum,
                                        self.data_size * self.shape, 
                                        x))
        return out + log(gsl_ran_gamma_pdf(x, 1, 1))

    cpdef double slice_sample(self,
                              double[::1] data,
                              double x_init=1,
                              list window=[0, 1e7],
                              int which_var=0):
        """
        Wrapper for _slice_sample(...) which takes data as an argument, uses a
        list variable to represent the bounds, and exposes everything to Python.

        This method first updates the model's data sufficient statistics which
        are called by self._logprob(...) before it calls _slice_sample(...).

        Arguments:
            data -- Array of doubles assumed drawn iid from Gamma(shape, scale).
            x_init -- Current value of rate parameter to be slice-sampled.
            window -- The bounds [x_min, x_max]
            max_iter -- Max number of iterations before terminating.
        """

        cdef:
            double x_min, x_max

        x_min = window[0]
        x_max = window[1]

        self.data_size = data.shape[0]
        self.data_sum = np.sum(data)

        return self._slice_sample(x_init=x_init,
                                  x_min=x_min,
                                  x_max=x_max,
                                  which_var=which_var)
