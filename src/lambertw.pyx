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

cpdef double calculate_zeta(double a, double b, double c) nogil:
    return _calculate_zeta(a, b, c)

cpdef double simulate_zeta(double a, double b, double c) nogil:
    return _simulate_zeta(a, b, c)

