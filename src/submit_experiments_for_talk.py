import sys
import numpy as np
import numpy.random as rn

from path import path
from subprocess import Popen, PIPE


PYTHON_INSTALLATION = '~/py2/bin/python'
ICEWS_DIR = path('/mnt/nfs/work1/wallach/aschein/data/icews/matrices')
GDELT_DIR = path('/mnt/nfs/work1/wallach/aschein/data/gdelt/matrices')
STOU_DIR = path('/mnt/nfs/work1/wallach/aschein/data/stou')
NIPS_DIR = path('/mnt/nfs/work1/wallach/aschein/data/nips-data')
DBLP_DIR = path('/mnt/nfs/work1/wallach/aschein/data/dblp')
MUSIC_DIR = path('/mnt/nfs/work1/wallach/aschein/data/music/piano_midi/train')
RESULTS_DIR = path('/mnt/nfs/work1/wallach/aschein/results/NIPS16/camera_ready')
CODE_DIR = path('/home/aschein/research/pgds/src')


def qsub(cmd, job_name=None, stdout=None, stderr=None, depend=None, n_cores=None):
    print(cmd)
    if type(depend) is str:
        depend = [depend]
    args = ['qsub']
    if n_cores:
        args.extend(['-pe', 'generic', '%d' % n_cores])
    if job_name:
        args.extend(['-N', job_name])
    if stderr:
        args.extend(['-e', stderr])
    if stdout:
        args.extend(['-o', stdout])
    if depend:
        args.extend(['-hold_jid', ','.join(depend)])
    out = Popen(args, stdin=PIPE, stdout=PIPE).communicate('%s\n' % cmd)[0]
    print out.rstrip()
    job_id = out.split()[2]
    return job_id


def get_out_dir(data_file, K=50, version='pgds'):
    if 'gdelt' in data_file or 'icews' in data_file:
        dataset_str = 'gdelt' if 'gdelt' in data_file else 'icews'
        edgetype_str = 'undirected' if 'undirected' in data_file else 'directed'
        datasubset_str = str(data_file.parent.namebase)
        out_dir = RESULTS_DIR.joinpath(dataset_str, edgetype_str, datasubset_str)

    elif 'piano_midi' in data_file:
        out_dir = RESULTS_DIR.joinpath('piano_midi', str(data_file.parent.namebase))

    elif 'stou' in data_file or 'dblp' in data_file or 'nips-data' in data_file:
        out_dir = RESULTS_DIR.joinpath(str(data_file.parent.namebase))
    else:
        raise TypeError

    if data_file.namebase != 'data':
        out_dir = out_dir.joinpath(data_file.namebase)
    out_dir = out_dir.joinpath('K_%d' % K, version)
    return out_dir


def submit_train_job(data_file, K=100, version='pgds', num_itns=6000,
                     save_every=100, save_after=4000, eval_every=100, eval_after=4000):
    out_dir = get_out_dir(data_file=data_file, K=K, version=version)
    out_dir.makedirs_p()

    seed = rn.randint(1111111111)
    if version in ['pgds', 'gpdpfa', 'orig-gpdpfa']:
        cmd = '%s %s ' % (PYTHON_INSTALLATION, CODE_DIR.joinpath('run_mcmc_model.py'))
        cmd += '-d=%s -o=%s -k=%d -v --version=%s ' % (data_file, out_dir, K, version)
        cmd += ' --steady --num_itns=%d --seed=%d --gam=%f ' % (num_itns, seed, 0.75 * K)
        cmd += '--save_every=%d --save_after=%d --eval_every=%d --eval_after=%d ' % (save_every,
                                                                                     save_after,
                                                                                     eval_every,
                                                                                     eval_after)
        if version != 'orig-gpdpfa':
            cmd += '--stationary '

    elif version == 'lds':
        cmd = '%s %s ' % (PYTHON_INSTALLATION, CODE_DIR.joinpath('lds.py'))
        cmd += '-d=%s -o=%s -k=%d -v ' % (data_file, out_dir, K)
        cmd += '--stationary --num_itns=%d --seed=%d ' % (num_itns, seed)

    if 'piano_midi' in data_file:
        cmd += '--binary '

    job_name = '%s%d-%s' % (version, K, data_file.abspath().parent.namebase)
    stdout = out_dir.joinpath('output-train.out')
    stderr = out_dir.joinpath('errors-train.out')
    jid = qsub(cmd, job_name=job_name, stdout=stdout, stderr=stderr)
    return jid, out_dir


def main():
    icews_years = [2001, 2003, 2006, 2009, 2011]
    icews_datasets = ['%d-D' % year for year in icews_years]
    icews_datasets = [ICEWS_DIR.joinpath('undirected', s) for s in icews_datasets]

    gdelt_datasets = [GDELT_DIR.joinpath('undirected', s) for s in icews_datasets]

    for dataset in gdelt_datasets:
        data_file = dataset.joinpath('data.npz')

        for version in ['pgds', 'lds']:
            if version == 'lds':
                num_itns = 10
            else:
                num_itns = 6000
                save_every = 100
                save_after = 4000
                eval_every = 7000
                eval_after = 7000

            Ks = [5, 10] if version == 'lds' else [100, 200]

            for K in Ks:
                model_depend = []
                for _ in xrange(2):
                    model_jid, out_dir = submit_train_job(data_file=data_file,
                                                          K=K,
                                                          version=version,
                                                          num_itns=num_itns,
                                                          save_every=save_every,
                                                          save_after=save_after,
                                                          eval_every=eval_every,
                                                          eval_after=eval_after)
                    model_depend.append(model_jid)

if __name__ == '__main__':
    main()
