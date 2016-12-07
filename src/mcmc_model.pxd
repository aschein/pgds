#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: language = c
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


cdef class MCMCModel:
    cdef:
        gsl_rng *rng
        int total_itns, print_every
        dict param_list

    cdef list _get_variables(self)
    cdef void _generate_state(self)
    cdef void _generate_data(self)
    cdef void _init_state(self)
    cdef void _print_state(self)
    cdef void _update(self, int num_itns, int verbose, dict burnin)
    cpdef void update(self, int num_itns, int verbose, dict burnin=?)
    cdef void _test(self,
                    int num_samples,
                    str method=?,
                    dict var_funcs=?,
                    dict burnin=?)
    cdef void _calc_funcs(self, int n, dict var_funcs, dict out)
    cpdef void geweke(self, int num_samples, dict var_funcs=?, dict burnin=?)
    cpdef void schein(self, int num_samples, dict var_funcs=?, dict burnin=?)
