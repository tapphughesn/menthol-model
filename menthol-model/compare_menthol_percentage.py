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

arrs = [np.load(df) for df in data_files]
savedir = os.path.join("..", "..", "figs", "comparisons")

arr_y_s_s = [np.sum(arr,axis=(1,2)) for arr in arrs]

for arr in arr_y_s_s:
    arr /= 1e6

mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey',]
x = np.arange(2016, 2016 + arr_y_s_s[0].shape[0])

arr_no_deads = [arr[:,:-1] for arr in arr_y_s_s]

arr_no_dead_percents = [arr_no_dead / np.sum(arr_no_dead, axis=1).reshape(-1,1) * 100 for arr_no_dead in arr_no_deads]

if False:
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

    plt.title("Effect of menthol ban on proportion of smokers", fontsize=16)

#################################################33

if True:
    fig, ax = plt.subplots(1,1,figsize=(16,6), dpi=200)
    ys = [arr[:,2] / np.sum(arr[:,2:], axis=1) * 100 for arr in arr_no_dead_percents]

    # x = x[:5]

    for y in ys:
        # y = y[:5]
        ax.plot(x, y)

    plt.ylim(0,50)
    # plt.ylim(5,25)
    plt.xlim(x[0]-1, x[-1]+1)
    # plt.xlim(2016, 2020)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage menthol smokers", fontsize=12)
    plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
    # plt.xticks(x[::1], fontsize=10, horizontalalignment='center')
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
    # for i,j in zip(x, y):
    #     # if (i - 2016) % 5 == 0:
    #     if i == 2017 or i == 2018:
    #         ax.annotate(str(int(j * 100) / 100) + "%", xy=(i + 0.5,j + 2))
    #         ax.scatter([i],[j],c=mycolors[0])
    # for i,j in zip(x, y2):
    #     # if (i - 2016) % 5 == 0:
    #     if i == 2017:
    #         ax.annotate(str(int(j * 100) / 100) + "%", xy=(i + 0.5,j + 2))
    #         ax.scatter([i],[j],c=mycolors[1])

    plt.title("Effect of menthol ban on proportion of menthol smokers in smoking population", fontsize=16)

#############################################3

plt.savefig(os.path.join(savedir, "long_term_menthol_proportion.png"))
# plt.savefig(os.path.join(savedir, "short_term_menthol_proportion.png"))