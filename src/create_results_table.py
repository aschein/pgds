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
            metric_vals = [float(m) for m in line.split()]
            for k, v in zip(metric_names, metric_vals):
                results[k].append(v)
    return results


def foo():
    dataset_dirs = [RESULTS_DIR.joinpath(x) for x in ['gdelt/directed',  'icews/undirected', 'nips-data', 'dblp', 'stou']]
    data_names = ['gdelt', 'icews', 'nips', 'dblp', 'stou']

    for dataset_dir, data_name in zip(dataset_dirs, data_names):
        for pred_type in ['smoothing', 'forecast']:
            name = '%s-%s' % (data_name, pred_type)
            print name
            print 'MODEL\tMRE\t\tMAE\t\tRMSE'
            for version in ['pgds', 'gpdpfa', 'lds']:
                K = 25 if version == 'lds' else 100
                pattern = dataset_dir.joinpath('*/masked_subset_[1|2]/K_%d/%s' % (K, version))
                pattern = pattern.joinpath('avg_%s_eval.txt' % pred_type)
                results = get_averaged_results(pattern)
                print '%s\t%f\t%f\t%f' % (version,
                                          np.mean(results['MRE']),
                                          np.mean(results['MAE']),
                                          np.mean(results['RMSE']))
            print

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-p', '--pattern', type=str, required=False)
    args = p.parse_args()

    foo()
