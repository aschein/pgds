import os
import sys
import numpy as np

from glob import glob
from path import path
from argparse import ArgumentParser
from collections import defaultdict

RESULTS_DIR = path('/mnt/nfs/work1/wallach/aschein/results/NIPS16/camera_ready')


def print_results(results):
    for k, v in results.iteritems():
        print '%s: %f +- %f' % (k, np.mean(v), np.std(v))


def get_chain_results(pattern='*_smoothing_eval.txt'):
    """
    Finds all chain-specific results files in a directory and its sub-directories.

    This will aggregate all the results (e.g., for averaging).

    Returns a dict where keys are error metrics and vals are lists of result values.
    """
    results = defaultdict(list)

    for eval_file in glob(pattern):
        with open(eval_file) as f:
            lines = [line.rstrip() for line in f.readlines()]
            header = lines[0]
            eval_lines = [x for x in lines if x.split()[0] != 'ITN']       # dont include header lines
            eval_lines = [x for x in eval_lines if int(x.split()[0]) > 0]  # dont include initialization error

        col_names = header.split()
        assert col_names[0] == 'ITN'
        metric_names = col_names[1:]

        for line in eval_lines:
            metric_vals = [float(m) for m in line.split()[1:]]
            for k, v in zip(metric_names, metric_vals):
                results[k].append(v)
    return results


def get_averaged_results(pattern='avg_smoothing_eval.txt'):
    """
    Finds all averaged results files in a directory and its sub-directories.

    This will aggregate all the results (e.g., for averaging).

    Returns a dict where keys are error metrics and vals are lists of result values.
    """
    results = defaultdict(list)

    for eval_file in glob(pattern):
        with open(eval_file) as f:
            lines = [line.rstrip() for line in f.readlines()]
            header = lines[0]
            eval_lines = [x for x in lines if 'MAE' not in x]       # dont include header lines

        col_names = header.split()
        metric_names = col_names

        for line in eval_lines:
            metric_vals = [float(m) for m in line.split()[1:]]
            for k, v in zip(metric_names, metric_vals):
                results[k].append(v)
    return results


def foo(chain_results=False):
    dataset_dirs = [RESULTS_DIR.joinpath(x) for x in ['gdelt/directed',  'icews/undirected']]

    for dataset_dir in dataset_dirs:
        for pred_type in ['smoothing', 'forecast']:
            name = 'gdelt' if 'gdelt' in dataset_dir else 'icews'
            name += '-%s' % pred_type
            print name
            print 'MODEL\tMRE\t\tMAE\t\tRMSE'
            pattern = dataset_dir.joinpath('*/masked_subset_[1|2]/K_100/pgds')
            if chain_results:
                pattern = pattern.joinpath('[1-9]*_%s_eval.txt' % pred_type)
                results = get_chain_results(pattern)
            else:
                pattern = pattern.joinpath('avg_%s_eval.txt' % pred_type)
                results = get_averaged_results(pattern)
            print 'pgds\t%f\t%f\t%f' % (np.mean(results['MRE']),
                                        np.mean(results['MAE']),
                                        np.mean(results['RMSE']))
            for K in [5, 10, 25, 50, 100]:
                pattern = dataset_dir.joinpath('*/masked_subset_[1|2]/K_%d/lds' % K)
                if chain_results:
                    pattern = pattern.joinpath('[1-9]*_%s_eval.txt' % pred_type)
                    results = get_chain_results(pattern)
                else:
                    pattern = pattern.joinpath('avg_%s_eval.txt' % pred_type)
                    results = get_averaged_results(pattern)

                print 'lds%d\t%f\t%f\t%f' % (K,
                                             np.mean(results['MRE']),
                                             np.mean(results['MAE']),
                                             np.mean(results['RMSE']))
            print

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-p', '--pattern', type=str, required=False)
    p.add_argument('--chain_results', action="store_true", default=False)
    args = p.parse_args()

    foo(args.chain_results)


    # foo(RESULTS_DIR.joinpath('icews/undirected/2001-D/*/K_10/lds/smoothing_eval.txt'))
    # print_results(args.pattern)
    foo()
