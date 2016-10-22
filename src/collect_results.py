import os
import sys
import numpy as np

from glob import glob
from path import path
from argparse import ArgumentParser
from collections import defaultdict

RESULTS_DIR = path('/mnt/nfs/work1/wallach/aschein/results/NIPS16/camera_ready')

GDELT_RESULTS_DIR = RESULTS_DIR.joinpath('gdelt/directed')


def print_averaged_results(pattern='smoothing_eval.txt'):
    print pattern
    results = get_results(pattern)
    for k, v in results.iteritems():
        print '%s: %f +- %f' % (k, np.mean(v), np.std(v))


def get_results(pattern='smoothing_eval.txt'):
    """
    Finds all results files in a directory and its sub-directories.

    This will aggregate all the results (e.g., for averaging).

    Returns a dict where keys are error metrics and vals are lists of result values.
    """
    results = defaultdict(list)

    for eval_file in glob(pattern):
        with open(eval_file) as f:
            lines = [line.rstrip() for line in f.readlines()]
            header = lines[0]
            eval_lines = lines[1:]

        col_names = header.split()
        assert col_names[0] == 'ITN'
        metric_names = col_names[1:]

        for line in eval_lines:
            metric_vals = [float(m) for m in line.split()[1:]]
            for k, v in zip(metric_names, metric_vals):
                results[k].append(v)
    return results


def foo():
    dataset_dirs = [RESULTS_DIR.joinpath(x) for x in ['gdelt/directed',  'icews/undirected']]

    for dataset_dir in dataset_dirs:
        for pred_type in ['smoothing', 'forecast']:
            name = 'gdelt' if 'gdelt' in dataset_dir else 'icews'
            name += '-%s' % pred_type
            print name
            print 'MODEL\tMRE\tMAE\tRMSE'
            pattern = dataset_dir.joinpath('*/masked_subset_[1|2]/K_100/pgds/*%s_eval.txt' % pred_type)
            results = get_results(pattern)
            print 'pgds\t%f\t%f\t%f' % (np.mean(results['MRE']),
                                        np.mean(results['MAE']),
                                        np.mean(results['RMSE']))
            for K in [5, 10, 25, 50, 100]:
                pattern = dataset_dir.joinpath('*/masked_subset_[1|2]/K_%d/lds/*%s_eval.txt' % (K, pred_type))
                results = get_results(pattern)
                print 'lds%d\t%f\t%f\t%f' % (K, np.mean(results['MRE']),
                                                np.mean(results['MAE']),
                                                np.mean(results['RMSE']))
            print

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-p', '--pattern', type=str, required=False)
    args = p.parse_args()

    # foo(RESULTS_DIR.joinpath('icews/undirected/2001-D/*/K_10/lds/smoothing_eval.txt'))
    # print_averaged_results(args.pattern)
    foo()
