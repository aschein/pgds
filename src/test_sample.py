from sample import *
import numpy as np
import numpy.random as rn
import scipy.stats as st
from itertools import product
from collections import defaultdict


def test_gamma(sampler, n_samples=100000, rtol=0.1):
    vals = [1e-2, 1, 5, 10, 25, 100, 1e5]
    for shape, scale in product(vals, vals):
        X = rn.gamma(shape, scale, size=n_samples)
        Y = np.asarray([sampler.gamma(shape, scale) for _ in xrange(n_samples)])

        for p in [25, 50, 75]:
            assert np.allclose(np.percentile(X, p), np.percentile(Y, p), rtol=rtol)


def test_beta(sampler, n_samples=100000, rtol=0.1):
    vals = [1e-1, 1, 5, 10, 25, 100, 1e5]
    for shape1, shape2 in product(vals, vals):
        X = rn.beta(shape1, shape2, size=n_samples)
        Y = np.asarray([sampler.beta(shape1, shape2) for _ in xrange(n_samples)])

        for p in [25, 50, 75]:
            if not np.allclose(np.percentile(X, p), np.percentile(Y, p), rtol=rtol):
                print shape1, shape2, p
                print np.percentile(X, p), np.percentile(Y, p)
                sys.exit()


def test_dirichlet(sampler, n_samples=10000, rtol=0.1):
    vals = [0.5, 1, 5, 10, 25, 100, 1e5]
    sizes = [10, 100, 10000]
    for concen, size in product(vals, sizes):
        print concen, size
        alpha = np.ones(size) * concen
        dictX = defaultdict(list)
        dictY = defaultdict(list)
        for _ in xrange(n_samples):
            X = rn.dirichlet(alpha)
            dictX['entropy'].append(st.entropy(X))
            dictX['geomean'].append(np.exp(np.log(X + 1e-300).mean()))

            Y = np.zeros(size)
            sampler.dirichlet(alpha, Y)
            assert np.allclose(Y.sum(), 1.)
            dictY['entropy'].append(st.entropy(Y))
            dictY['geomean'].append(np.exp(np.log(Y + 1e-300).mean()))

        for k in dictX.keys():
            for p in [25, 50, 75]:
                if not np.allclose(np.percentile(dictX[k], p),
                                   np.percentile(dictY[k], p),
                                   rtol=rtol):
                    print concen, size, k, p
                    print np.percentile(dictX[k], p), np.percentile(dictY[k], p)
                    sys.exit()


if __name__ == '__main__':
    s = Sampler(seed=None)
    test_gamma(s)
    test_beta(s)
    test_dirichlet(s)
