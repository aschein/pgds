import sys

import cPickle as pickle
import numpy as np
import numpy.random as rn
import sklearn.metrics as metrics

from argparse import ArgumentParser
from path import path
from time import sleep

from pgds import PGDS
from gpdpfa import GPDPFA


def get_train_forecast_split(data, mask):
    assert mask.shape == data.shape
    if not mask[-1].all():
        return data, None
    else:
        for S in xrange(data.shape[0]):
            if not mask[-(S+1):].all():
                break
        assert S >= 1
        train_data = data[:-S]
        forecast_data = data[-S:]
        return train_data, forecast_data


def get_chain_num(out_dir):
    sleep(rn.random() * 2)

    chain_num_path = out_dir.joinpath('num_chains.txt')

    chain_num = 1
    if chain_num_path.exists():
        chain_num = int(np.loadtxt(chain_num_path)) + 1
    np.savetxt(chain_num_path, np.array([chain_num]))

    return chain_num


def rmse(truth, pred):
    return np.sqrt(((truth-pred)**2).mean())


def mae(truth, pred):
    return np.abs(truth-pred).mean()


def mre(truth, pred):
    return (np.abs(truth - pred) / (truth + 1)).mean()


def auc(truth, pred):
    assert (truth[truth > 0] == 1).all() and (truth[truth < 1] == 0).all()
    fpr, tpr, thresholds = metrics.roc_curve(truth, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def save_forecast_eval(data_SV, model, eval_path, pred_path=None):
    eval_str = ''
    if not eval_path.exists():
        eval_str = 'ITN\tMAE\t\tMRE\t\tRMSE'
        if model.get_params()['binary']:
            eval_str += '\t\tAUC'
        eval_str += '\n'

    S, V = data_SV.shape
    pred_SV = model.forecast(n_timesteps=S)

    itn = model.get_total_itns()

    eval_str += '%d\t%f\t%f\t%f' % (itn,
                                    mae(data_SV, pred_SV),
                                    mre(data_SV, pred_SV),
                                    rmse(data_SV, pred_SV))
    if model.get_params()['binary'] == 1:
        eval_str += '\t%f' % auc(data_SV.ravel(), pred_SV.ravel())
    eval_str += '\n'

    with open(eval_path, 'a+') as f:
        f.write(eval_str)

    if pred_path is not None:
        np.savez(pred_path, pred_SV=pred_SV)


def save_smoothing_eval(masked_data, model, eval_path, pred_path=None):
    eval_str = ''
    if not eval_path.exists():
        eval_str = 'ITN\tMAE\t\tMRE\t\tRMSE'
        if model.get_params()['binary'] == 1:
            eval_str += '\t\tAUC'
        eval_str += '\n'

    mask_TV = masked_data.mask
    assert mask_TV.any()

    data_N = masked_data.data[mask_TV]
    pred_N = model.reconstruct(subs=np.where(mask_TV))

    itn = model.get_total_itns()

    eval_str += '%d\t%f\t%f\t%f' % (itn,
                                    mae(data_N, pred_N),
                                    mre(data_N, pred_N),
                                    rmse(data_N, pred_N))
    if model.get_params()['binary']:
        eval_str += '\t%f' % auc(data_N, pred_N)
    eval_str += '\n'

    with open(eval_path, 'a+') as f:
        f.write(eval_str)

    if pred_path is not None:
        np.savez(pred_path, pred_N=pred_N)


def main():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-k', '--n_components', type=int, default=100)
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--gam', type=float, default=100.0)
    p.add_argument('--binary', action="store_true", default=False)
    p.add_argument('--stationary', action="store_true", default=False)
    p.add_argument('--steady', action="store_true", default=False)
    p.add_argument('-s', '--seed', type=int, default=None)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('-n', '--num_itns', type=int, default=6000)
    p.add_argument('--save_after', type=int, default=4000)
    p.add_argument('--save_every', type=int, default=100)
    p.add_argument('--eval_after', type=int, default=4000)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--version', type=str, default='gds', choices=['pgds', 'gpdpfa', 'orig-gpdpfa'])
    args = p.parse_args()

    data_dict = np.load(args.data)
    data = data_dict['data']

    mask = np.zeros_like(data).astype(bool)
    if 'mask' in data_dict.keys():
        mask = data_dict['mask']

    data_TV, data_SV = get_train_forecast_split(data, mask)

    T, V = data_TV.shape
    if data_SV is not None:
        S = data_SV.shape[0]
        mask_TV = mask[:T]
    else:
        S = 0
        mask_TV = mask
    masked_data = np.ma.array(data_TV, mask=mask_TV)

    args.out.makedirs_p()

    if args.version == 'pgds':
        model = PGDS(T=T,
                     V=V,
                     K=args.n_components,
                     eps=args.eps,
                     gam=args.gam,
                     tau=args.tau,
                     stationary=int(args.stationary),
                     steady=int(args.steady),
                     binary=int(args.binary),
                     seed=args.seed)

        burnin = {'Y_KV': 0,
                  'Y_TK': 0,
                  'L_TKK': 0,
                  'H_KK': 50,
                  'lnq_K': 50,
                  'Phi_KV': 0,
                  'delta_T': 0,
                  'Pi_KK': 40,
                  'nu_K': 50,
                  'Theta_TK': 20,
                  'beta': 60,
                  'xi_K': 70}

    elif args.version in ['gpdpfa', 'orig-gpdpfa']:
        model = GPDPFA(T=T+1,
                       V=V,
                       K=args.n_components,
                       e=args.eps,
                       f=args.eps,
                       stationary=int(args.version == 'gpdpfa'),
                       binary=int(args.binary),
                       seed=args.seed)
        burnin = {}

    num_itns = args.num_itns
    save_after = args.save_after
    save_every = args.save_every
    eval_after = args.eval_after
    eval_every = args.eval_every

    itns = np.arange(num_itns + 1)  # include iteration 0

    itns_to_eval = np.zeros(num_itns + 1, dtype=bool)  # include iteration 0
    if eval_every is not None:
        itns_to_eval = itns % eval_every == 0
        itns_to_eval[:eval_after] = False  # dont save before eval_after
        itns_to_eval[0] = True  # always evaluate the first
        itns_to_eval[-1] = True  # always evaluate the last

    itns_to_save = np.zeros(num_itns + 1, dtype=bool)  # include iteration 0
    if save_every is not None:
        itns_to_save = itns % save_every == 0
        itns_to_save[:save_after] = False  # dont save before save_after
        itns_to_save[0] = True  # except the first, always save the first
    itns_to_save[-1] = True  # always save the last

    itns_to_checkpoint = itns_to_eval + itns_to_save
    itns_to_checkpoint[0] = True  # this ensures num_itns_until_chkpt works
    checkpoint_itns = np.where(itns_to_checkpoint)[0]

    num_checkpoints = checkpoint_itns.size
    assert num_checkpoints >= 2  # checkpoint_itns will at least be [0,num_itns]
    assert checkpoint_itns[-1] == num_itns

    chain = get_chain_num(args.out)
    with open(args.out.joinpath('%d_params.p' % chain), 'wb') as params_file:
        pickle.dump(model.get_params(), params_file)

    for c, itn in enumerate(checkpoint_itns):
        if c == 0:
            initialize = True
            num_itns_until_chkpt = 0
        else:
            initialize = False
            num_itns_until_chkpt = checkpoint_itns[c] - checkpoint_itns[c-1]

        model.fit(masked_data,
                  num_itns=num_itns_until_chkpt,
                  verbose=args.verbose,
                  initialize=initialize,
                  burnin=burnin)

        if itns_to_save[itn]:
            state_name = '%d_state_%d.npz' % (chain, itn)
            np.savez(args.out.joinpath(state_name), **dict(model.get_state()))

        if itns_to_eval[itn]:
            eval_path = args.out.joinpath('%d_smoothing_eval.txt' % chain)
            pred_path = args.out.joinpath('%d_smoothed_%d.npz' % (chain, itn))
            save_smoothing_eval(masked_data, model, eval_path, pred_path)

            if S == 0:
                continue
            eval_path = args.out.joinpath('%d_forecast_eval.txt' % chain)
            pred_path = args.out.joinpath('%d_forecast_%d.npz' % (chain, itn))
            save_forecast_eval(data_SV, model, eval_path, pred_path)

if __name__ == '__main__':
    main()
