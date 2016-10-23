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
    data_names = ['gdelt', 'icews', 'nips', 'dblp', 'stou']

    for data_name in data_names:
        dataset_dir = RESULTS_DIR.joinpath(data_name)
        if data_name == 'nips':
            dataset_dir = RESULTS_DIR.joinpath('nips-data')
        elif data_name == 'gdelt':
            dataset_dir = RESULTS_DIR.joinpath('gdelt/directed')
        elif data_name == 'icews':
            dataset_dir = RESULTS_DIR.joinpath('icews/undirected')

        for pred_type in ['smoothing', 'forecast']:
            print '%s-%s' % (data_name, pred_type)
            print 'MODEL\tMRE\t\tMAE\t\tRMSE'
            for version in ['pgds', 'gpdpfa', 'lds']:
                K = 25 if version == 'lds' else 100
                pattern = dataset_dir.joinpath('*/masked_subset_[1|2]/K_%d/%s' % (K, version))
                pattern = pattern.joinpath('avg_%s_eval.txt' % pred_type)
                results = get_averaged_results(pattern)

                if not results['MRE']:
                    print data_name, pred_type, version
                    sys.exit()

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
