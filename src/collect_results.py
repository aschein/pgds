import os
import sys
import numpy as np

from glob import glob
from path import path
from argparse import ArgumentParser
from collections import defaultdict

RESULTS_DIR = path('/mnt/nfs/work1/wallach/aschein/results/NIPS16/camera_ready')


def foo(out_dir, pattern='smoothing_eval.txt'):
    """
    Finds all results files in a directory and its sub-directories.

    Returns mean and std across each file for each error metric.
    """
    results = defaultdict(list)

    for eval_file in glob(pattern):
        print eval_file
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

    print out_dir
    for k, v in results.iteritems():
        print '%s: %f +- %f' % (k, np.mean(v), np.std(v))


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-p', '--pattern', type=str, required=True)
    args = p.parse_args()

    # foo(RESULTS_DIR.joinpath('icews/undirected/2001-D/*/K_10/lds/smoothing_eval.txt'))
    foo(args.pattern)