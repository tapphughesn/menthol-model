import pandas as pd
import numpy as np
from datetime import datetime
import os
from simulation import Simulation
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import ticker

data_files = sorted(glob('../../outputs/numpy_arrays/np_output_menthol_ban_1_?_*'))
# data_files = sorted(glob('../../outputs/numpy_arrays/np_output_menthol_ban_?_1_*'))
data_files = ['../../outputs/numpy_arrays/np_output_2022-04-15_11-39-20-499748.npy'] + data_files
# data_files = ['../../outputs/numpy_arrays/np_output_2022-04-15_11-39-20-499748.npy',
#               '../../outputs/numpy_arrays/np_output_calibrated_2022-05-02_02-11-37-317448.npy']

arrs = [np.load(df) for df in data_files]
savedir = os.path.join("..", "..", "figs", "comparisons")

arr_y_s_s = [np.sum(arr,axis=(1,2)) for arr in arrs]

for arr in arr_y_s_s:
    arr /= 1e6

mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey',]
x = np.arange(2016, 2016 + arr_y_s_s[0].shape[0])

if False:
    fig, ax = plt.subplots(1,1,figsize=(16,9), dpi=200)
    labels = ["Never Smoker", "Former Smoker", "Menthol Smoker", "Nonmenthol Smoker", "E-cig Only", "Dead"]

    y = np.vstack([arr_y_s_s[:,i] for i in range(arr_y_s_s.shape[1])])

    ax = plt.gca()
    ax.stackplot(x,y,labels=labels, colors=mycolors, alpha=0.8)

    # for i in x:
    #     ax.axvline(x=i, c="black")

    ax.set_title('Smoking groups over time', fontsize=18)
    ax.set(ylim=[0, 2.3e2])
    ax.legend(fontsize=10, ncol=3)
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
    plt.yticks(np.arange(0,2.3e2,2e1,dtype=np.int64), fontsize=10)
    plt.xlim(x[0], x[-1])
    plt.xlabel("Year")
    plt.ylabel("Population (millions)")

    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.savefig(os.path.join(savedir, "vstack.png"))
    plt.clf()

###############################################

arr_no_deads = [arr[:,:-1] for arr in arr_y_s_s]

arr_no_dead_percents = [arr_no_dead / np.sum(arr_no_dead, axis=1).reshape(-1,1) * 100 for arr_no_dead in arr_no_deads]

if False:
    fig, ax = plt.subplots(1,1,figsize=(16,9), dpi=200)
    ax.plot(x, arr_no_dead_percents[:,0], mycolors[0],
            x, arr_no_dead_percents[:,1], mycolors[1],
            x, arr_no_dead_percents[:,2], mycolors[2],
            x, arr_no_dead_percents[:,3], mycolors[3],
            x, arr_no_dead_percents[:,4], mycolors[4])

    plt.ylim(0,100)
    plt.xlim(x[0]-1, x[-1]+1)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage of Population", fontsize=12)
    plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
    ax.legend(labels, fontsize=12, ncol=1)

    plt.title("Proportion of smoking groups in the living population", fontsize=16)
    plt.savefig(os.path.join(savedir, "group_proportions.png"))
    plt.clf()

########################################

if True:
    fig, ax = plt.subplots(1,1,figsize=(16,6), dpi=200)
    ys = [np.sum(arr[:,2:], axis=1) for arr in arr_no_dead_percents]

    for y in ys:
        ax.plot(x, y)
    
    plt.ylim(10,30)
    plt.xlim(x[0]-1, x[-1]+1)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage of Smokers in population", fontsize=12)
    plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
    # ax.legend(["complex death rate", "average death rate"], fontsize=12, ncol=1)
    ax.legend(["no menthol ban",
               "long term option 1",
               "long term option 2",
               "long term option 3",
               "long term option 4"],
                fontsize=12, ncol=1)
    # ax.legend(["no menthol ban",
    #            "short term option 1",
    #            "short term option 2",
    #            "short term option 3",
    #            "short term option 4"],
    #             fontsize=12, ncol=1)
    # ax.legend(["PATH population", "Calibrated population"])

    # for y in ys:
    #     for i,j in zip(x, y):
    #         if (i - 2016) % 5 == 0 and i > 2040:
    #         # if i == 2017:
    #             ax.annotate(str(int(j * 100) / 100) + "%", xy=(i,j+0.5))
    #             ax.scatter([i],[j],c=mycolors[0])

    # for i,j in zip(x, y2):
    #     # if (i - 2016) % 5 == 0 and i > 2040:
    #     if i == 2017:
    #         ax.annotate(str(int(j * 100) / 100) + "%", xy=(i,j+0.5))
    #         ax.scatter([i],[j],c=mycolors[1])

    # plt.title("Proportion of smokers in the living population", fontsize=16)
    plt.title("Proportion of smokers", fontsize=16)

#################################################33

if False:
    fig, ax = plt.subplots(1,1,figsize=(16,6), dpi=200)
    y = arr_no_dead_percents[:,2] / np.sum(arr_no_dead_percents[:,2:], axis=1) * 100
    # y2 = arr2_no_dead_percents[:,2] / np.sum(arr2_no_dead_percents[:,2:], axis=1) * 100

    ax.plot(x, y, mycolors[0])
    # ax.plot(x, y2, mycolors[1])

    plt.ylim(0,100)
    plt.xlim(x[0]-1, x[-1]+1)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage menthol smokers", fontsize=12)
    plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
    # ax.legend(["complex death rate", "average death rate"], fontsize=12, ncol=1)
    ax.legend(["menthol ban", "no menthol ban"], fontsize=12, ncol=1)
    for i,j in zip(x, y):
        # if (i - 2016) % 5 == 0:
        if i == 2017 or i == 2018:
            ax.annotate(str(int(j * 100) / 100) + "%", xy=(i + 0.5,j + 2))
            ax.scatter([i],[j],c=mycolors[0])
    # for i,j in zip(x, y2):
    #     # if (i - 2016) % 5 == 0:
    #     if i == 2017:
    #         ax.annotate(str(int(j * 100) / 100) + "%", xy=(i + 0.5,j + 2))
    #         ax.scatter([i],[j],c=mycolors[1])

    plt.title("Effect of menthol ban on proportion of menthol smokers in smoking population", fontsize=16)
    plt.savefig(os.path.join(savedir, "menthol_proportion.png"))
    plt.clf()

#############################################3

plt.savefig(os.path.join(savedir, "long_term_smoker_proportion_debug.png"))
# plt.savefig(os.path.join(savedir, "short_term_smoker_proportion_debug.png"))
# plt.savefig(os.path.join(savedir, "path_vs_calibrated_smoker_proportion_debug.png"))