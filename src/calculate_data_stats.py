import numpy as np
from path import path
from IPython import embed

ICEWS_DIR = path('/mnt/nfs/work1/wallach/aschein/data/icews/matrices')
GDELT_DIR = path('/mnt/nfs/work1/wallach/aschein/data/gdelt/matrices')
STOU_DIR = path('/mnt/nfs/work1/wallach/aschein/data/stou')
NIPS_DIR = path('/mnt/nfs/work1/wallach/aschein/data/nips-data')
DBLP_DIR = path('/mnt/nfs/work1/wallach/aschein/data/dblp')

icews_years = [2001, 2004, 2007, 2010]
icews_datasets = ['%d-%d-3D' % (year, year + 2) for year in icews_years]
icews_datasets = [ICEWS_DIR.joinpath('undirected', s) for s in icews_datasets]

gdelt_years = [2001, 2002, 2003, 2004, 2005]
gdelt_datasets = ['%d-D' % year for year in gdelt_years]
gdelt_datasets = [GDELT_DIR.joinpath('directed', s) for s in gdelt_datasets]

text_datasets = [NIPS_DIR, STOU_DIR, DBLP_DIR]

if __name__ == '__main__':
    for data_dir in gdelt_datasets + icews_datasets + text_datasets:
        Y_TV = np.load(data_dir.joinpath('masked_subset_3.npz'))['data']
        Y_TV = Y_TV.astype(float)  # necessary to cast from unsigned ints to do np.diff
        assert Y_TV.shape[1] == 1000

        mu_V = Y_TV.mean(axis=0)
        D_TV = np.abs(np.diff(Y_TV, axis=0))
        B_V = D_TV / mu_V

        print B_V.mean(), B_V.std(), data_dir

    embed()
