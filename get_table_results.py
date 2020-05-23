#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:14, 23/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import pandas as pd
import numpy as np

path_file = "history/results/"
data_files = ["daily_rainfall_22022020", "weekly_rainfall_22022020"]
model_files = ["mlp", "Rnn1HL", "Lstm1HL", "GaMlp", "DeMlp", "FpaMlp", "PsoMlp", "WoaMlp", "WdoMlp", "MvoMlp", "EoMlp", "NroMlp", "HgsoMlp"]
model_files2 = ["MLP", "RNN", "LSTM", "GA-MLP", "DE-MLP", "FPA-MLP", "PSO-MLP", "WOA-MLP", "WDO-MLP", "MVO-MLP", "EO-MLP", "NRO-MLP", "HGSO-MLP"]
cols = ["EVS", "ME", "MAE",	"MSE", "RMSE", "MSLE", "MedAE", "R2", "SMAPE", "MAAPE", "MASE"]

for datafile in data_files:
    table_mean = []
    table_std = []
    table_var = []
    for modelfile in model_files:
        filename = path_file + datafile + "/" + modelfile + ".csv"
        dataframe = pd.read_csv(filename, usecols=cols)
        # dataset = dataframe.values
        t_mean = dataframe.mean()
        t_std = dataframe.std()
        t_var = dataframe.var()
        t_mean = np.round(t_mean.values, 3)
        t_std = np.round(t_std.values, 4)
        t_var = t_var.values
        table_mean.append(t_mean)
        table_std.append(t_std)
        table_var.append(t_var)

        # print(t_var)
        # t4 = dataframe.kurt()
        # t5 = dataframe.skew()
        # t6 = dataframe.describe()

    table_mean = np.array(table_mean)
    table_std = np.array(table_std)
    table_var = np.array(table_var)

    df_mean = pd.DataFrame(data=table_mean, columns=cols, index=model_files2)
    df_std = pd.DataFrame(data=table_std, columns=cols, index=model_files2)
    df_var = pd.DataFrame(data=table_var, columns=cols, index=model_files2)

    df_mean.to_csv(path_file + "/" + datafile + "_mean.csv")
    df_std.to_csv(path_file + "/" + datafile + "_std.csv")
    df_var.to_csv(path_file + "/" + datafile + "_var.csv")