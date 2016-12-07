import numpy as np
import numpy.random as rn
import scipy.stats as st

from pgds import PGDS
from IPython import embed

if __name__ == '__main__':
    V = 3
    T = 4
    K = 5
    eps = 0.75
    gam = 30.
    tau = 0.75
    shrink = 0
    stationary = 1
    steady = 1
    binary = 0

    seed = rn.randint(10000)
    print seed

    model = PGDS(T=T, V=V, K=K, eps=eps, gam=gam, tau=tau,
                 shrink=shrink, stationary=stationary, steady=steady,
                 binary=binary, seed=seed)

    burnin = {'Y_KV': 0,
              'Y_TK': 0,
              'L_TKK': 0,
              'H_KK': 0,
              'lnq_K': 0,
              'Theta_TK': 0,
              'Phi_KV': 0,
              'delta_T': 0,
              'Pi_KK': 0,
              'nu_K': 0,
              'beta': 0,
              'xi_K': np.inf}

    entropy_funcs = {'Entropy min': lambda x: np.min(st.entropy(x)),
                     'Entropy max': lambda x: np.max(st.entropy(x)),
                     'Entropy mean': lambda x: np.mean(st.entropy(x)),
                     'Entropy var': lambda x: np.var(st.entropy(x))}

    var_funcs = {'Pi_KK': entropy_funcs,
                 'Phi_KV': entropy_funcs}

    model.schein(30000, var_funcs=var_funcs, burnin=burnin)
    # model.geweke(200000, var_funcs=var_funcs, burnin=burnin)
