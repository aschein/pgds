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

cimport numpy as np
from libc.math cimport log, log1p, exp, M_E, tgamma, sqrt

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
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)
    double gsl_ran_gamma(gsl_rng * r, double a, double b)
    unsigned int gsl_ran_logarithmic (const gsl_rng * r, double p)
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])
    double gsl_ran_ugaussian (const gsl_rng * r)

DEF EPS = 1e-300

cdef inline double _sample_gamma(gsl_rng * rng, double a, double b) nogil:
    """
    Simulate a Gamma random variate.

    When a is large, this is a thin wrapper to gsl_ran_gamma.
    
    When a is small, this calls _sample_ln_gamma_small_shape, and
    exponentiates the result.

    This method clips the shape parameter at EPS. 
    It also clips the result at EPS: the result is always >= EPS.

    Arguments:
        rng -- Pointer to a GSL random number generator object
        a -- Shape parameter (a > 0)
        b -- Scale parameter (b > 0)
    """
    cdef double g

    if a < EPS:
        a = EPS

    if a > 0.75:
        g = gsl_ran_gamma(rng, a, b)
    else:
        g = exp(_sample_lngamma_small_shape(rng, a, b))
    
    return g if g > EPS else EPS

cdef inline double _sample_lngamma(gsl_rng * rng, double a, double b) nogil:
    """
    Simulate a log-Gamma random variate.

    When a is large, this calls gsl_ran_gamma and returns the log of it.
    
    When a is small, this is a thin wrapper to _sample_ln_gamma_small_shape.

    This method clips the shape parameter at EPS.

    Arguments:
        rng -- Pointer to a GSL random number generator object
        a -- Shape parameter (a > 0)
        b -- Scale parameter (b > 0)
    """
    cdef double g

    if a < EPS:
        a = EPS

    if a > 0.75:
        g = gsl_ran_gamma(rng, a, b)
        return log(g) if g > EPS else log(EPS)
    else:
        return _sample_lngamma_small_shape(rng, a, b)

cdef inline double _sample_lngamma_small_shape(gsl_rng * rng,
                                               double a,
                                               double b) nogil:
    """
    Implements the algorithm described by Liu, Martin and Syring (2015) for 
    simulating Gamma random variates with small shape parameters (a < 1).

    Do not call this with a > 1 (the expected number of iterations is large).

    Arguments:
        rng -- Pointer to a GSL random number generator object
        a -- Shape parameter (a > 0)
        b -- Scale parameter (b > 0)

    References:
    [1] C. Liu, R. Martin & N. Syring (2013).
        Simulating from a gamma distribution with small shape parameter
    """
    cdef: 
        double lam, w, lnr, lnu, z, lnh, lneta

    lam = 1. / a - 1.
    lnw = log(a) - log1p(-a) - 1
    lnr = -log1p(exp(lnw))

    while 1:
        lnu = log(gsl_rng_uniform(rng))
        if lnu <= lnr:
            z = lnr - lnu
        else:
            z = log(gsl_rng_uniform(rng)) / lam
        
        lnh = -z - exp(-z / a)
        if z >= 0:
            lneta = -z
        else:
            lneta = lnw + log(lam) + lam * z

        if (lnh - lneta) > log(gsl_rng_uniform(rng)):
            return log(b) - z / a


cdef inline double _sample_lnbeta(gsl_rng * rng, double a, double b) nogil:
    """
    Sample a log-Beta by sampling 2 log-Gamma random variates.

    Arguments:
        rng -- Pointer to a GSL random number generator object
        a -- First shape parameter (a > 0)
        b -- Second shape parameter (b > 0)
    """
    cdef:
        double lng1, lng2, c, lse

    lng1 = _sample_lngamma(rng, a, 1.)
    lng2 = _sample_lngamma(rng, b, 1.)

    c = lng1 if lng1 > lng2 else lng2
    lse = c + log(exp(lng1 - c) + exp(lng2 - c))  # logsumexp

    return lng1 - lse

cdef inline double _sample_beta(gsl_rng * rng, double a, double b) nogil:
    """
    Sample a Beta by exponentiating a log-Beta.

    Arguments:
        rng -- Pointer to a GSL random number generator object
        a -- First shape parameter (a > 0)
        b -- Second shape parameter (b > 0)
    """
    return exp(_sample_lnbeta(rng, a, b))

cdef inline void _sample_lndirichlet(gsl_rng * rng,
                                     double[::1] alpha,
                                     double[::1] out) nogil:
    """
    Sample a K-dimensional log-Dirichlet by sampling K log-Gamma random variates

    Arguments:
        rng -- Pointer to a GSL random number generator object
        alpha -- Concentration parameters (alpha[k] > 0 for k = 1...K)
        out -- Output array (same size as alpha)
    """
    cdef:
        int K, k
        double c, lng, lse

    K = alpha.shape[0]
    if out.shape[0] != K:
        out[0] = -1
        return

    c = out[0] = _sample_lngamma(rng, alpha[0], 1.)
    for k in range(1, K):
        out[k] = lng = _sample_lngamma(rng, alpha[k], 1.)
        if lng > c:
            c = lng

    lse = 0  # logsumexp to compute the normalizer
    for k in range(K):
        lse += exp(out[k] - c)
    lse = c + log(lse)

    for k in range(K):
        out[k] = out[k] - lse

cdef inline void _sample_dirichlet(gsl_rng * rng,
                                   double[::1] alpha,
                                   double[::1] out) nogil:
    """
    Sample a K-dimensional Dirichlet by exponentiating a log-Dirichlet.

    Arguments:
        rng -- Pointer to a GSL random number generator object
        alpha -- Concentration parameters (alpha[k] > 0 for k = 1...K)
        out -- Output array (same size as alpha)
    """
    cdef:
        int K, k

    K = alpha.shape[0]
    if out.shape[0] != K:
        out[0] = -1
        return

    _sample_lndirichlet(rng, alpha, out)

    for k in range(K):
        out[k] = exp(out[k])

cdef inline int _sample_categorical(gsl_rng * rng, double[::1] dist) nogil:
    """
    Uses the inverse CDF method to return a sample drawn from the
    specified (unnormalized) discrete distribution.

    TODO: Use searchsorted to reduce complexity from O(K) to O(logK).
          This requires creating a CDF array which requires GIL, if using numpy.

    Arguments:
        rng -- Pointer to a GSL random number generator object
        dist -- (unnormalized) distribution
    """

    cdef:
        int k, K
        double r

    K = dist.shape[0]

    r = 0.0
    for k in range(K):
        r += dist[k]

    r *= gsl_rng_uniform(rng)

    for k in range(K):
        r -= dist[k]
        if r <= 0.0:
            return k

    return -1

cdef inline int _searchsorted(double val, double[::1] arr) nogil:
    """
    Find first element of a sorted array that is greater than a given value.

    Arguments:
        val -- Given value to search for
        arr -- Sorted (ascending order) array
    """
    cdef:
        int imin, imax, imid

    imin = 0
    imax = arr.shape[0] - 1
    while (imin < imax):
        imid = (imin + imax) / 2
        if arr[imid] < val:
            imin = imid + 1
        else:
            imax = imid
    return imin

cdef inline int _sample_crt(gsl_rng * rng, int m, double r) nogil:
    """
    Sample a Chinese Restaurant Table (CRT) random variable [1].

    l ~ CRT(m, r) can be sampled as the sum of indep. Bernoullis:

            l = \sum_{n=1}^m Bernoulli(r/(r+n-1))

    where m >= 0 is integer and r >=0 is real.

    Arguments:
        rng -- Pointer to a GSL random number generator object
        m -- First parameter of the CRT (m >= 0)
        r -- Second parameter of the CRT (r >= 0)

    References:
    [1] M. Zhou & L. Carin (2012). Negative Binomial Count and Mixture Modeling.
    """
    cdef:
        int l, n
        double u, p

    if m < 0 or r < 0:
        return -1  # represents an error

    if m == 0:
        return 0

    if m == 1:
        return 1

    if r <= 1e-50:
        return 1

    l = 0
    for n in range(m):
        p = r / (r + n)
        u = gsl_rng_uniform(rng)
        if p > u:
            l += 1
    return l

cdef inline int _sample_sumcrt(gsl_rng * rng, int[::1] M, double[::1] R) nogil:
    """
    Sample the sum of K independent CRT random variables.

        l ~ \sum_{k=1}^K CRT(m_k, r_k)

    Arguments:
        rng -- Pointer to a GSL random number generator object
        M -- Array of first parameters
        R -- Array of second parameters (same size as M)
    """
    cdef:
        int l, lk, K, k 

    K = M.shape[0]

    l = 0
    for k in range(K):
        lk = _sample_crt(rng, M[k], R[k])
        if lk == -1:
            return -1
        else:
            l += lk

cdef inline int _sample_sumlog(gsl_rng * rng, int n, double p) nogil:
    """
    Sample a SumLog random variable defined as the sum of n iid Logarithmic rvs:

        y ~ \sum_{i=1}^n Logarithmic(p)

    Arguments:
        rng -- Pointer to a GSL random number generator object
        n -- Parameter for number of iid Logarithmic rvs
        p -- Probability parameter of the Logarithmic distribution

    TODO: For very large p (e.g., =0.99999) this sometimes fails and returns 
          negative integers.  Figure out why gsl_ran_logarithmic is failing.
    """
    cdef:
        int i, out

    if p <= 0 or p >= 1 or n < 0:
        return -1  # this represents an error

    if n == 0:
        return 0

    out = 0
    for i in range(n):
        out += gsl_ran_logarithmic(rng, p)
    return out


cdef inline int _sample_trunc_poisson(gsl_rng * rng, double mu) nogil:
    """
    Sample a truncated Poisson random variable as described by Zhou (2015) [1].

    Arguments:
        rng -- Pointer to a GSL random number generator object
        mu -- Poisson rate parameter

    References:
    [1] Zhou, M. (2015). Infinite Edge Partition Models for Overlapping 
        Community Detection and Link Prediction.
    """
    cdef:
        unsigned int x
        double u

    if mu >= 1:
        while 1:
            x = gsl_ran_poisson(rng, mu)
            if x > 0:
                return x
    else:
        while 1:
            x = gsl_ran_poisson(rng, mu) + 1
            u = gsl_rng_uniform(rng)
            if x < 1. / u:
                return x


cdef inline void _sample_multinomial(gsl_rng * rng,
                                    unsigned int N,
                                    double[::1] p,
                                    unsigned int[::1] out) nogil:
    """
    Wrapper for gsl_ran_multinomial.
    """
    cdef:
        size_t K

    K = p.shape[0]
    gsl_ran_multinomial(rng, K, N, &p[0], &out[0])


cdef class Sampler:
    """
    Wrapper for a gsl_rng object that exposes all sampling methods to Python.

    Useful for testing or writing pure Python programs.
    """
    cdef:
        gsl_rng *rng

    cpdef double gamma(self, double a, double b)
    cpdef double lngamma(self, double a, double b)
    cpdef double beta(self, double a, double b)
    cpdef double lnbeta(self, double a, double b)
    cpdef void dirichlet(self, double[::1] alpha, double[::1] out)
    cpdef void lndirichlet(self, double[::1] alpha, double[::1] out)
    cpdef int categorical(self, double[::1] dist)
    cpdef int searchsorted(self, double val, double[::1] arr)
    cpdef int crt(self, int m, double r)
    cpdef int sumcrt(self, int[::1] M, double[::1] R)
    cpdef int sumlog(self, int n, double p)
    cpdef int trunc_poisson(self, double mu)
    cpdef void multinomial(self,
                           unsigned int N,
                           double[::1] p,
                           unsigned int[::1] out)
