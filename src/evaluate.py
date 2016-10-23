import numpy as np

from glob import glob
from path import path
from argparse import ArgumentParser
from run_mcmc_model import get_train_forecast_split, rmse, mae, mre, auc


DATA_DIR = path('/mnt/nfs/work1/wallach/aschein/data/')


def get_data_for_results_dir(results_dir):
    data_name = results_dir.split('camera_ready')[1].split('K_')[0].strip('/')
    data_name = data_name.replace('icews', 'icews/matrices')
    data_name = data_name.replace('gdelt', 'gdelt/matrices')
    data_name = data_name.replace('piano_midi', 'music/piano_midi/train')
    data_file = DATA_DIR.joinpath(data_name + '.npz')
    if not data_file.exists():
        raise ValueError('File not found: %s' % data_file)
    data_dict = np.load(data_file)
    data = data_dict['data']
    mask = data_dict['mask']
    data_TV, data_SV = get_train_forecast_split(data, mask)
    T, V = data_TV.shape
    assert data_SV is not None
    mask_TV = mask[:T]
    return data_TV, data_SV, mask_TV


def save_avg_forecast_eval(results_dir):
    _, data_SV, _ = get_data_for_results_dir(results_dir)

    forecast_files = glob(results_dir.joinpath('*[1-9]*_forecast*.npz'))
    forecast_files = [x for x in forecast_files if '0_forecast' not in x]  # exclude 0th forecast
    if forecast_files:
        avg_pred_SV = np.zeros_like(data_SV, dtype=float)

        for forecast_file in forecast_files:
            avg_pred_SV += np.load(forecast_file)['pred_SV']

        avg_pred_SV /= len(forecast_files)

        with open(results_dir.joinpath('avg_forecast_eval.txt'), 'w+') as f:
            a = mae(data_SV, avg_pred_SV)
            b = mre(data_SV, avg_pred_SV)
            c = rmse(data_SV, avg_pred_SV)
            f.write('MAE\t\tMRE\t\tRMSE\n%f\t%f\t%f\n' % (a, b, c))

        np.savez(results_dir.joinpath('avg_forecast.npz'), avg_pred_SV=avg_pred_SV)


def save_avg_smoothing_eval(results_dir):
    data_TV, _, mask_TV = get_data_for_results_dir(results_dir)
    data_N = data_TV[mask_TV]

    smoothed_files = glob(results_dir.joinpath('*[1-9]*_smoothed*.npz'))
    smoothed_files = [x for x in smoothed_files if '0_smoothed' not in x]  # exclude 0th forecast
    if smoothed_files:
        avg_pred_N = np.zeros_like(data_N, dtype=float)

        for smoothed_file in smoothed_files:
            avg_pred_N += np.load(smoothed_file)['pred_N']

        avg_pred_N /= len(smoothed_files)

        with open(results_dir.joinpath('avg_smoothing_eval.txt'), 'w+') as f:
            a = mae(data_N, avg_pred_N)
            b = mre(data_N, avg_pred_N)
            c = rmse(data_N, avg_pred_N)

            if 'piano' not in results_dir:
                f.write('MAE\t\tMRE\t\tRMSE\n%f\t%f\t%f\n' % (a, b, c))
            else:
                d = auc(data_N, avg_pred_N)
                f.write('MAE\t\tMRE\t\tRMSE\t\tAUC\n%f\t%f\t%f\t%f\n' % (a, b, c, d))

        np.savez(results_dir.joinpath('avg_smoothed.npz'), avg_pred_N=avg_pred_N)


def is_results_dir(results_dir):
    return 0 < len(results_dir.files('*forecast*npz')) + len(results_dir.files('*smoothed*npz'))


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-r', '--results', type=path, required=True)
    args = p.parse_args()

    if is_results_dir(args.results):
        save_avg_forecast_eval(args.results)
        save_avg_smoothing_eval(args.results)
    else:
        for subdir in args.results.walkdirs():
            if is_results_dir(subdir):
                save_avg_forecast_eval(subdir)
                save_avg_smoothing_eval(subdir)
                print subdir
