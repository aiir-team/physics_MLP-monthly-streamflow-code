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

path_file = "results3_final/"
data_files = ["full_dataset"]
model_files = ["mlp0", "Rnn1HL-0", "Lstm1HL-0", "GaMlp-0", "DeMlp-0", "FpaMlp-0", "PsoMlp-0", "WoaMlp-0", "GwoMlp-0", "SsaMlp-0",
               "WdoMlp-0", "MvoMlp-0", "EoMlp-0", "NroMlp-0", "HgsoMlp-0"]
model_files2 = ["MLP", "RNN", "LSTM", "GA-MLP", "DE-MLP", "FPA-MLP", "PSO-MLP", "WOA-MLP", "GWO-MLP", "SSA-MLP",
                "WDO-MLP", "MVO-MLP", "EO-MLP", "NRO-MLP", "HGSO-MLP"]

cols = ["EVS", "MAE", "MSE", "RMSE", "MSLE", "R2", "MRE", "MAPE", "SMAPE", "MAAPE", "MASE", "NSE", "Willmott_Index", "R", "Confidence"]

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

    df_mean.to_csv(path_file + "/csv/" + datafile + "_mean.csv")
    df_std.to_csv(path_file + "/csv/" + datafile + "_std.csv")
    df_var.to_csv(path_file + "/csv/" + datafile + "_var.csv")