#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:03, 06/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from matplotlib.ticker import FuncFormatter
import pandas as pd
import matplotlib.pyplot as plt


def read_dataset_file(filepath=None, usecols=None, header=0, index_col=False, inplace=True):
    df = pd.read_csv(filepath, usecols=usecols, header=header, index_col=index_col)
    df.dropna(inplace=inplace)
    return df.values

def plot_all_files(filenames, col_indexs, xlabels, ylabels, titles, colours, pathsaves):
    for i in range(0, len(filenames)):
        filename = filenames[i] + ".csv"
        pathsave = pathsaves[i] + ".pdf"
        col_index = col_indexs[i]
        color = colours[i]
        xlabel = xlabels[i]
        ylabel = ylabels[i]
        title = titles[i]

        dataset = read_dataset_file(filename, usecols=col_index, header=0)
        ax = plt.subplot()
        plt.plot(dataset, color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_title(title)

        plt.savefig(pathsave, bbox_inches = "tight")
        plt.show()

import glob

xlabels = ["Timestamp (day)", "Timestamp (week)"]
ylabels = ["Value", "Value"]
filenames = pathsaves = titles = [f.split(".")[0] for f in glob.glob("*.csv")]
col_indexs = [ [1] for _ in range(len(filenames)) ]
colours = [ '#1f77b4' for _ in range(len(filenames)) ]
plot_all_files(filenames, col_indexs, xlabels, ylabels, titles, colours, pathsaves)
