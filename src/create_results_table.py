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
            for version in ['pgds', 'gpdpfa', 'orig-gpdpfa', 'lds']:
                K = 25 if version == 'lds' else 100

                path_str = 'masked_subset_[3|4]/K_%d/%s' % (K, version)
                if 'gdelt' in data_name or 'icews' in data_name:
                    path_str = '*/' + path_str
                pattern = dataset_dir.joinpath(path_str)
                pattern = pattern.joinpath('avg_%s_eval.txt' % pred_type)
                results = get_averaged_results(pattern)
                if not results['MAE']:
                    continue

                vstr = version if version != 'orig-gpdpfa' else 'ogpdpfa'
                print '%s\t%f\t%f\t%f' % (vstr,
                                          np.mean(results['MRE']),
                                          np.mean(results['MAE']),
                                          np.mean(results['RMSE']))
            print


def print_latex_table():
    data_names = ['gdelt', 'icews', 'nips', 'dblp', 'stou']

    for data_name in data_names:
        dataset_dir = RESULTS_DIR.joinpath(data_name)
        if data_name == 'nips':
            dataset_dir = RESULTS_DIR.joinpath('nips-data')
            data_str = 'NIPS'
        elif data_name == 'gdelt':
            dataset_dir = RESULTS_DIR.joinpath('gdelt/directed')
            data_str = 'GDELT'
        elif data_name == 'icews':
            dataset_dir = RESULTS_DIR.joinpath('icews/undirected')
            data_str = 'ICEWS'
        elif data_name == 'dblp':
            data_str = 'DBLP'
        elif data_name == 'stou':
            data_str = 'SOTU'

        for pred_type in ['smoothing', 'forecast']:
            pred_str = '-S' if pred_type == 'smoothing' else '-F'

            line_str = '\\scriptsize{%s%s} ' % (data_str, pred_str)

            means = defaultdict(list)
            stds = defaultdict(list)
            for version in ['pgds', 'gpdpfa', 'lds']:
                K = 25 if version == 'lds' else 100

                path_str = 'masked_subset_[3|4]/K_%d/%s' % (K, version)
                if 'gdelt' in data_name or 'icews' in data_name:
                    path_str = '*/' + path_str
                pattern = dataset_dir.joinpath(path_str)
                pattern = pattern.joinpath('avg_%s_eval.txt' % pred_type)
                results = get_averaged_results(pattern)
                assert results['MAE']

                for error_metric in ['MAE', 'MRE', 'RMSE']:
                    means[error_metric].append(np.mean(results[error_metric]))
                    stds[error_metric].append(np.std(results[error_metric]))

            for error_metric in ['MAE', 'MRE', 'RMSE']:
                # print means[error_metric], np.argmin(means[error_metric])
                for m, (model_mean, model_std) in enumerate(zip(means[error_metric], stds[error_metric])):
                    if np.argmin(means[error_metric]) == m:
                        line_str += '& $\mathbf{%.2f}$ $\\mathsmaller{\\pm %.2f}$ ' % (model_mean, model_std)
                    else:
                        line_str += '& %.2f $\\mathsmaller{\\pm %.2f}$ ' % (model_mean, model_std)
            line_str += '\\\\'
            print line_str

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-p', '--pattern', type=str, required=False)
    args = p.parse_args()

    print_latex_table()
