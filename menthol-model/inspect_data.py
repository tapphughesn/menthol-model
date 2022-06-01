import pandas as pd
import numpy as np
from datetime import datetime
import os
from simulation import Simulation
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import ticker

# data_files = sorted(glob('../../outputs/numpy_arrays/np_output_menthol_ban_?_?_*'))
# for df in data_files:
for var1 in range(1,5):
    for var2 in range(1,5):

        data_files = sorted(glob(f'../../outputs/numpy_arrays/np_output_menthol_ban_{var1}_{var2}_*'))
        # data_files = sorted(glob(f'../../outputs/numpy_arrays/np_output_calibrated*'))

        df = data_files[0]
        print(df)
        arr = np.load(df)
        savedir = os.path.join("..","..","figs","debug",
                f"short_term_option_{var1}", f"long_term_option_{var2}")
        # savedir = os.path.join("..", "..", "figs", "calibrated")

        arr_year_smoking_state = None

        arr_year_smoking_state = np.sum(arr,axis=(1,2))
        arr_year_smoking_state.shape

        arr_year_smoking_state /= 1e6

        mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey',]

        fig, ax = plt.subplots(1,1,figsize=(16,9), dpi=200)
        labels = ["Never Smoker", "Former Smoker", "Menthol Smoker", "Nonmenthol Smoker", "E-cig Only", "Dead"]

        x = np.arange(2016, 2016 + arr_year_smoking_state.shape[0])
        y = np.vstack([arr_year_smoking_state[:,i] for i in range(arr_year_smoking_state.shape[1])])

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

        arr_no_dead = arr_year_smoking_state[:,:-1]

        arr_no_dead_percents = arr_no_dead / np.sum(arr_no_dead, axis=1).reshape(-1,1) * 100

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

        fig, ax = plt.subplots(1,1,figsize=(16,6), dpi=200)
        y = np.sum(arr_no_dead_percents[:,2:], axis=1)

        ax.plot(x, y, mycolors[0])

        plt.ylim(10,30)
        plt.xlim(x[0]-1, x[-1]+1)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Percentage of Smokers in population", fontsize=12)
        plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
        # ax.legend(["complex death rate", "average death rate"], fontsize=12, ncol=1)
        ax.legend(["menthol ban", "no menthol ban"], fontsize=12, ncol=1)
        for i,j in zip(x, y):
            # if (i - 2016) % 5 == 0 and i > 2040:
            if i == 2017:
                ax.annotate(str(int(j * 100) / 100) + "%", xy=(i,j+0.5))
                ax.scatter([i],[j],c=mycolors[0])
        # for i,j in zip(x, y2):
        #     # if (i - 2016) % 5 == 0 and i > 2040:
        #     if i == 2017:
        #         ax.annotate(str(int(j * 100) / 100) + "%", xy=(i,j+0.5))
        #         ax.scatter([i],[j],c=mycolors[1])

        plt.title("Proportion of smokers in the living population", fontsize=16)
        # plt.title("Effect of menthol ban on proportion of smokers", fontsize=16)

        plt.savefig(os.path.join(savedir, "smoker_proportion.png"))
        plt.clf()

        #################################################33

        fig, ax = plt.subplots(1,1,figsize=(16,6), dpi=200)
        y = arr_no_dead_percents[:,2] / np.sum(arr_no_dead_percents[:,2:], axis=1) * 100

        ax.plot(x, y, mycolors[0])

        plt.ylim(0,100)
        plt.xlim(x[0]-1, x[-1]+1)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Percentage menthol smokers", fontsize=12)
        plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
        # ax.legend(["complex death rate", "average death rate"], fontsize=12, ncol=1)
        # ax.legend(["menthol ban", "no menthol ban"], fontsize=12, ncol=1)
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

        # plt.title("Effect of menthol ban on proportion of menthol smokers in smoking population", fontsize=16)
        plt.title("Proportion of menthol smokers in smoking population", fontsize=16)
        plt.savefig(os.path.join(savedir, "menthol_proportion.png"))
        plt.clf()

        #############################################3

