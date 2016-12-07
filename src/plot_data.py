import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import seaborn as sns
import numpy as np
import pandas as pd
from path import path
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

if __name__ == '__main__':
    sns.set_style("darkgrid")
    fig = plt.figure(dpi=1000, figsize=(40, 6))
    # fig = plt.figure()
    gs = gridspec.GridSpec(1, 2)
    gs.update(hspace=0, wspace=0.1)

    ax1 = fig.add_subplot(gs[0, 0])
    data = np.load('/Users/aaronschein/Documents/research/mlds/data/nips-data/masked_subset_1.npz')
    Y_TV = data['data']
    labels = data['words']
    dates = data['dates']

    for i in range(4):
        line = ax1.plot(Y_TV[:, i], 'o-', label=labels[i],  markersize=15, linewidth=7)
        ax1.fill_between(range(Y_TV.shape[0]), Y_TV[:, i], alpha=0.2, color=line[0].get_color())
    xticks = np.linspace(1, Y_TV.shape[0] - 2, num=6, dtype=int)
    xticks = [1, 5, 8, 11, 15]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([dates[int(x)] for x in xticks], rotation=0,
                        weight='bold', fontsize=45)
    ax1.set_yticklabels([int(x) for x in ax1.get_yticks()], fontsize=42, weight='bold')
    # plt.setp(ax1.get_yticklabels(), fontsize=42, weight='bold')

    leg = ax1.legend(prop={'size': 37, 'weight': 'bold'}, labelspacing=0.001, loc=(0.35, 0))
    for text in leg.get_texts():
        text.set_path_effects([path_effects.withStroke(foreground='w', linewidth=4, alpha=0.5)])

    ax2 = fig.add_subplot(gs[0, 1])
    data = np.load('/Users/aaronschein/Documents/research/mlds/data/icews/matrices/2007-2009-3D/masked_subset_2.npz')
    Y_TV = data['data']
    labels = data['labels']
    dates = data['dates']

    dates = [pd.Timestamp(x).strftime('%b %Y') for x in dates]

    for i in range(4):
        lstr = labels[i].replace('--', '$\leftrightarrow$')
        lstr = lstr.replace('Occupied Palestinian Territory', 'Palestine')
        lstr = lstr.replace('Russian Federation', 'Russia')
        lstr = lstr.replace('United States', 'USA')
        line = ax2.plot(Y_TV[-100:, i], 'o-', label=lstr, markersize=13, linewidth=4)
        ax2.fill_between(range(Y_TV[-100:, i].shape[0]), Y_TV[-100:, i], alpha=0.3, color=line[0].get_color())
    xticks = np.linspace(8, 100-8, num=5, dtype=int)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([dates[-100:][int(x)] for x in xticks], rotation=0,
                        weight='bold', fontsize=45)
    ax2.set_yticklabels([int(x) for x in ax2.get_yticks()], fontsize=42, weight='bold')
    # plt.setp(ax2.get_yticklabels(), fontsize=42, weight='bold')
    ax2.legend(prop={'size': 40, 'weight': 'bold'}, labelspacing=0.05, loc=(0.01, 0.5))

    # ax2.set_title('Top four most active edges in ICEWS',
    #                weight='bold', fontsize=35)

    FIGS_DIR = path('/Users/aaronschein/Documents/research/mlds/gammadynam/papers/nips16/camera_ready/figs')
    plt.savefig(FIGS_DIR.joinpath('data.pdf'), format='pdf', bbox_inches='tight')
    print FIGS_DIR.joinpath('data.pdf')

    # sns.set_style("darkgrid")
    # fig = plt.figure(dpi=1000, figsize=(40, 12))
    # # fig = plt.figure()
    # gs = gridspec.GridSpec(2, 1)
    # gs.update(hspace=0.1, wspace=0.0)

    # ax1 = fig.add_subplot(gs[0, 0])
    # data = np.load('/Users/aaronschein/Documents/research/mlds/data/nips-data/masked_subset_1.npz')
    # Y_TV = data['data']
    # labels = data['words']
    # dates = data['dates']

    # for i in range(4):
    #     line = ax1.plot(Y_TV[:, i], 'o--', label=labels[i], linewidth=3, markersize=10)
    #     ax1.fill_between(range(Y_TV.shape[0]), Y_TV[:, i], alpha=0.3, color=line[0].get_color())
    # xticks = np.linspace(1, Y_TV.shape[0] - 2, num=5, dtype=int)
    # ax1.set_xticks(xticks)
    # ax1.set_xticklabels([dates[int(x)] for x in xticks], rotation=0,
    #                     weight='bold', fontsize=25)
    # plt.setp(ax1.get_yticklabels(), fontsize=25, weight='bold')

    # leg = ax1.legend(prop={'size': 20, 'weight': 'bold'}, loc='lower center')
    # leg.set_path_effects([path_effects.withStroke(foreground='w', linewidth=5, alpha=0.9)])

    # # ax1.set_title('Top four most frequent words in the NIPS corpus',
    # #               weight='bold', fontsize=35)

    # ax2 = fig.add_subplot(gs[1, 0])
    # data = np.load('/Users/aaronschein/Documents/research/mlds/data/icews/matrices/2007-2009-3D/masked_subset_2.npz')
    # Y_TV = data['data']
    # labels = data['labels']
    # dates = data['dates']

    # dates = [pd.Timestamp(x).strftime('%b %Y') for x in dates]

    # for i in range(4):
    #     lstr = labels[i].replace('--', '$\leftrightarrow$')
    #     lstr = lstr.replace('Occupied ', '')
    #     lstr = lstr.replace('Russian Federation', 'Russia')
    #     ax2.plot(Y_TV[-100:, i], 'o--', label=lstr, linewidth=3, markersize=10)
    # xticks = np.linspace(5, 100-5, num=6, dtype=int)
    # ax2.set_xticks(xticks)
    # ax2.set_xticklabels([dates[-100:][int(x)] for x in xticks], rotation=0,
    #                     weight='bold', fontsize=25)
    # plt.setp(ax2.get_yticklabels(), fontsize=25, weight='bold')
    # ax2.legend(prop={'size': 20, 'weight': 'bold'}, loc='upper left')
    # # ax2.set_title('Top four most active edges in ICEWS',
    # #                weight='bold', fontsize=35)

    # FIGS_DIR = path('/Users/aaronschein/Documents/research/mlds/gammadynam/papers/nips16/camera_ready/figs')
    # plt.savefig(FIGS_DIR.joinpath('stacked_data.pdf'), format='pdf', bbox_inches='tight')
