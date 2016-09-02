#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: extra_link_args = ['-lgsl', '-lgslcblas']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    void gsl_rng_set(gsl_rng * r, unsigned long int)
    void gsl_rng_free(gsl_rng * r)

cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_rng_uniform(gsl_rng * r)
    double gsl_ran_exponential(gsl_rng * r, double mu)
    double gsl_ran_gamma_pdf(double x, double a, double b)
    double gsl_ran_poisson_pdf(unsigned int k, double mu)


cdef class SliceSampler:
    """
    Interface for a model-like object that implements slice-sampling. 

    _slice_sample(...) calls _logprob(...) (which must be implemented).

    This is intended as an instructive example of how to implement a nogil
    slice-sampler not (necessarily) an interface to inherit.
    """
    cdef:
        gsl_rng *rng

    cdef double _logprob(self, double x, int which_var=?) nogil

    cdef double _slice_sample(self,
                              double x_init,
                              double x_min,
                              double x_max,
                              int which_var=?,
                              int max_iter=?) nogil

cdef class PoissonSliceSampler(SliceSampler):
    """
    Simple example one-parameter model that inherits SliceSampler and implements
    the _lobprob(...) function as well as a wrapper to _slice_sample(...).
    """
    cdef:
        int data_size, data_sum

    cpdef double slice_sample(self,
                              int[::1] data,
                              double x_init=?,
                              list window=?)

cdef class GammaSliceSampler(SliceSampler):
    """
    Simple example two-parameter model that inherits SliceSampler and implements
    the _lobprob(...) function as well as a wrapper to _slice_sample(...).
    """
    cdef:
        double shape, scale
        int data_size, data_sum

    cpdef double slice_sample(self,
                              double[::1] data,
                              double x_init=?,
                              list window=?,
                              int which_var=?)


