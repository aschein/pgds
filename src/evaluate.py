import numpy as np

from glob import glob
from path import path
from argparse import ArgumentParser
from run_pgds import get_train_forecast_split, rmse, mae, mre


DATA_DIR = path('/mnt/nfs/work1/wallach/aschein/data/')


def get_data_for_results_dir(results_dir):
    data_name = results_dir.split('camera_ready')[1].split('K_')[0].strip('/')
    data_name.replace('icews', 'icews/matrices')
    data_name.replace('gdelt', 'gdelt/matrices')
    data_file = DATA_DIR.joinpath(data_name, '.npz')
    if not data_file.exists():
        raise ValueError('File not found: %s' % data_file)
    data_dict = np.load(data_file)
    data = data_dict['data']
    mask = data_dict['mask']
    data_TV, data_SV = get_train_forecast_split(data, mask)
    return data_TV, data_SV, mask


def save_avg_forecast_eval(results_dir):
    _, data_SV, _ = get_data_for_results_dir(results_dir)

    forecast_files = glob(results_dir.joinpath('*forecast_[1-9]*.npz'))  # excludes 0th forecast
    print forecast_files
    if forecast_files:
        avg_pred_SV = np.zeros_like(data_SV)

        for forecast_file in forecast_files:
            avg_pred_SV += np.load(forecast_file)['pred_SV']

        avg_pred_SV /= len(forecast_files)

        with open(results_dir.joinpath('avg_forecast_eval.txt'), 'w+') as f:
            a = mae(data_SV, avg_pred_SV)
            b = mre(data_SV, avg_pred_SV)
            c = rmse(data_SV, avg_pred_SV)
            f.write('MAE\tMRE\tRMSE\n%f\t%f\t%f\n' % (a, b, c))

        np.savez(results_dir.joinpath('avg_forecast.npz'), avg_pred_SV=avg_pred_SV)


def save_avg_smoothing_eval(results_dir):
    data_TV, _, mask = get_data_for_results_dir(results_dir)
    data_N = data_TV[mask]

    smoothed_files = glob(results_dir.joinpath('*smoothed_[1-9]*.npz'))

    if smoothed_files:
        avg_pred_N = np.zeros_like(data_N)

        for smoothed_file in smoothed_files:
            avg_pred_N += np.load(smoothed_file)['pred_N']

        avg_pred_N /= len(smoothed_files)

        with open(results_dir.joinpath('avg_smoothing_eval.txt'), 'w+') as f:
            a = mae(data_N, avg_pred_N)
            b = mre(data_N, avg_pred_N)
            c = rmse(data_N, avg_pred_N)
            f.write('MAE\tMRE\tRMSE\n%f\t%f\t%f\n' % (a, b, c))

        np.savez(results_dir.joinpath('avg_smoothed.npz'), avg_pred_N=avg_pred_N)


def main():
    p = ArgumentParser()
    p.add_argument('-r', '--results', type=path, required=True)
    args = p.parse_args()

    assert args.results.exists()
    save_avg_forecast_eval(args.results)
    save_avg_smoothing_eval(args.results)
