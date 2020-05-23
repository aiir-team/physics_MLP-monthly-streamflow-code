#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:59, 23/05/2020                                                        %
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

x_labels = ["Timestamp (daily)", "Timestamp (weekly)"]
y_label = "Value"
titles = ["Group 1 in daily data", "Group 2 in daily data", "Group 3 in daily data"]
#titles = ["Truth, MLP, GA-MLP, PSO-MLP, MVO-MLP", "Truth, RNN, DE-MLP, WOA-MLP, EO-MLP", "Truth, LSTM, FPA-MLP, WDO-MLP, NRO-MLP"]

file_names = ["prediction_daily_run9", "prediction_weekly_run6"]
path_folder = "history/results/error/"
path_save = "history/results/img/"

point_numbers = [100, 65]           # Depend on file names
point_starts = [600, 0]             # Depend on file names

col_names = ["Truth", "MLP", "RNN", "LSTM", "GA-MLP", "DE-MLP", "FPA-MLP", "PSO-MLP", "WOA-MLP", "WDO-MLP", "MVO-MLP", "EO-MLP", "NRO-MLP"]

for idx, datafile in enumerate(file_names):

    dataframe = read_csv(path_folder + datafile + ".csv", header=0, names=col_names, index_col=False, engine='python')

    truth = dataframe['Truth'].values
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

    ## Group 1: "Truth, MLP, GA-MLP, PSO-MLP, MVO-MLP"
    plt.plot(x, truth[point_start:point_start + point_number], marker='o', linestyle=':', label='Truth')
    plt.plot(x, mlp[point_start:point_start + point_number], marker='s', linestyle=':', label='MLP')
    plt.plot(x, ga[point_start:point_start + point_number], marker='x', linestyle=':', label='GA-MLP')
    plt.plot(x, pso[point_start:point_start + point_number], marker='*', linestyle=':', label='PSO-MLP')
    plt.plot(x, mvo[point_start:point_start + point_number], marker=4, label='MVO-MLP')

    plt.xlabel(x_labels[idx])
    plt.ylabel(y_label)
    plt.title(titles[0])
    plt.legend()
    plt.savefig(path_save + datafile + "_pred_group1.pdf", bbox_inches='tight')
    plt.show()

    ## Group 2: "Truth, RNN, DE-MLP, WOA-MLP, EO-MLP"
    plt.plot(x, truth[point_start:point_start + point_number], marker='o', linestyle=':', label='Truth')
    plt.plot(x, rnn[point_start:point_start + point_number], marker='s', linestyle=':', label='RNN')
    plt.plot(x, de[point_start:point_start + point_number], marker='x', linestyle=':', label='DE-MLP')
    plt.plot(x, woa[point_start:point_start + point_number], marker='*', linestyle=':', label='WOA-MLP')
    plt.plot(x, eo[point_start:point_start + point_number], marker=4, label='EO-MLP')

    plt.xlabel(x_labels[idx])
    plt.ylabel(y_label)
    plt.title(titles[1])
    plt.legend()
    plt.savefig(path_save + datafile + "_pred_group2.pdf", bbox_inches='tight')
    plt.show()

    ## Group 3: "Truth, LSTM, FPA-MLP, WDO-MLP, NRO-MLP"
    plt.plot(x, truth[point_start:point_start + point_number], marker='o', linestyle=':', label='Truth')
    plt.plot(x, lstm[point_start:point_start + point_number], marker='s', linestyle=':', label='LSTM')
    plt.plot(x, fpa[point_start:point_start + point_number], marker='x', linestyle=':', label='FPA-MLP')
    plt.plot(x, wdo[point_start:point_start + point_number], marker='*', linestyle=':', label='WDO-MLP')
    plt.plot(x, nro[point_start:point_start + point_number], marker=4, label='NRO-MLP')

    plt.xlabel(x_labels[idx])
    plt.ylabel(y_label)
    plt.title(titles[2])
    plt.legend()
    plt.savefig(path_save + datafile + "_pred_group3.pdf", bbox_inches='tight')
    plt.show()





    #
    # plt.plot(x, truth[point_start:point_start + point_number], marker=lines.CARETDOWN, linestyle=':', label='Truth')
    # plt.plot(x, de[point_start:point_start + point_number], marker=4, label='DE-MLP')
    #
    # plt.plot(x, pso[point_start:point_start + point_number], marker='s', linestyle=':', label='PSO-MLP')
    # plt.plot(x, woa[point_start:point_start + point_number], marker='*', linestyle=':', label='WOA-MLP')
    # plt.xlabel(x_labels[0])
    # plt.ylabel(y_label)
    # plt.title(titles[0])
    # plt.legend()
    # plt.savefig(path_save + datafile + "_test.png", bbox_inches='tight')
    # plt.show()
    #
    # plt.plot(x, truth[point_start:point_start + point_number], marker='o', linestyle=':', label='Truth')
    #
    # plt.plot(x, mvo[point_start:point_start + point_number], marker='x', linestyle=':', label='MVO-MLP')
    # plt.plot(x, eo[point_start:point_start + point_number], marker=4, label='EO-MLP')
    # plt.plot(x, nro[point_start:point_start + point_number], marker=4, label='NRO-MLP')
    #
    # plt.xlabel(x_labels[0])
    # plt.ylabel(y_label)
    # plt.title(titles[0])
    # plt.legend()
    # plt.savefig(path_save + datafile + "_test.png", bbox_inches='tight')
    # plt.show()
