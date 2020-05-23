#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 22:36, 23/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%


import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pandas import read_csv
import numpy as np

## https://matplotlib.org/api/markers_api.html

x_labels = ["Epoch", "Epoch"]
y_label = "MSE"
titles = ["Group 1 in daily data", "Group 2 in daily data"]
# titles = ["MLP, RNN, GA-MLP, PSO-MLP, WDO-MLP, EO-MLP", "LSTM, DE-MLP, FPA-MLP, WOA-MLP, MVO-MLP, NRO-MLP"]

file_names = ["daily_error_run2", "weekly_error_run4"]
path_folder = "history/results/error/"
path_save = "history/results/img/"

point_numbers = [1000, 1000]  # Depend on file names
point_starts = [0, 0]  # Depend on file names

col_names = ["Epoch", "MLP", "RNN", "LSTM", "GA-MLP", "DE-MLP", "FPA-MLP", "PSO-MLP", "WOA-MLP", "WDO-MLP", "MVO-MLP", "EO-MLP", "NRO-MLP"]

for idx, datafile in enumerate(file_names):
    dataframe = read_csv(path_folder + datafile + ".csv", header=0, names=col_names, index_col=False, engine='python')

    epoch = dataframe['Epoch'].values
    mlp = dataframe['MLP'].values
    rnn = dataframe['RNN'].values
    lstm = dataframe['LSTM'].values
    ga = dataframe['GA-MLP'].values
    de = dataframe['DE-MLP'].values
    fpa = dataframe['FPA-MLP'].values
    pso = dataframe['PSO-MLP'].values
    woa = dataframe['WOA-MLP'].values
    wdo = dataframe['WDO-MLP'].values
    mvo = dataframe['MVO-MLP'].values
    eo = dataframe['EO-MLP'].values
    nro = dataframe['NRO-MLP'].values

    point_number = point_numbers[idx]
    point_start = point_starts[idx]
    x = np.arange(point_number)

    ## Group 1: MLP, RNN, GA-MLP, PSO-MLP, WDO-MLP, EO-MLP"
    plt.plot(x, mlp[point_start:point_start + point_number], marker='.', linestyle='solid', label='MLP')
    plt.plot(x, rnn[point_start:point_start + point_number], marker='.', linestyle='solid', label='RNN')
    plt.plot(x, ga[point_start:point_start + point_number], marker='.', linestyle='solid', label='GA-MLP')
    plt.plot(x, pso[point_start:point_start + point_number], marker='.', linestyle='solid', label='PSO-MLP')
    plt.plot(x, wdo[point_start:point_start + point_number], marker='.', linestyle='solid', label='WDO-MLP')
    plt.plot(x, eo[point_start:point_start + point_number], marker='.', linestyle='solid', label='EO-MLP')

    plt.xlabel(x_labels[idx])
    plt.ylabel(y_label)
    plt.title(titles[0])
    plt.legend()
    plt.savefig(path_save + datafile + "_error_group1.pdf", bbox_inches='tight')
    plt.show()

    ## Group 2: , "LSTM, DE-MLP, FPA-MLP, WOA-MLP, MVO-MLP, NRO-MLP"
    plt.plot(x, lstm[point_start:point_start + point_number], marker='.', linestyle='solid', label='LSTM')
    plt.plot(x, de[point_start:point_start + point_number], marker='.', linestyle='solid', label='DE-MLP')
    plt.plot(x, fpa[point_start:point_start + point_number], marker='.', linestyle='solid', label='FPA-MLP')
    plt.plot(x, woa[point_start:point_start + point_number], marker='.', linestyle='solid', label='WOA-MLP')
    plt.plot(x, mvo[point_start:point_start + point_number], marker='.', linestyle='solid', label='MVO-MLP')
    plt.plot(x, nro[point_start:point_start + point_number], marker='.', linestyle='solid', label='NRO-MLP')

    plt.xlabel(x_labels[idx])
    plt.ylabel(y_label)
    plt.title(titles[1])
    plt.legend()
    plt.savefig(path_save + datafile + "_error_group2.pdf", bbox_inches='tight')
    plt.show()