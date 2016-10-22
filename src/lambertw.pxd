#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#distutils: extra_link_args = ['-lgsl', '-lgslcblas']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

from libc.math cimport exp, log1p

cdef extern from "gsl/gsl_sf_lambert.h" nogil:
    double gsl_sf_lambert_Wm1(double x)

cpdef double calculate_zeta(double a, double b, double c) nogil

cdef inline double _calculate_zeta(double a, double b, double c) nogil:
    return -a * gsl_sf_lambert_Wm1(-b / (a * exp((b + c) / a))) - b - c

cpdef double simulate_zeta(double a, double b, double c) nogil

cdef inline double _simulate_zeta(double a, double b, double c) nogil:
    cdef:
        int _
        double zeta
  
    zeta = 1.
    for _ in range(35):
        zeta = a * log1p((zeta + c) / b)
    return zeta
