#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 22:57, 24/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import pandas as pd
import numpy as np

path_file = "history/results/"
data_files = ["daily_rainfall_22022020", "weekly_rainfall_22022020"]
model_files = ["mlp", "Rnn1HL", "Lstm1HL", "GaMlp", "DeMlp", "FpaMlp", "PsoMlp", "WoaMlp", "WdoMlp", "MvoMlp", "EoMlp", "NroMlp"]
model_files2 = ["MLP", "RNN", "LSTM", "GA-MLP", "DE-MLP", "FPA-MLP", "PSO-MLP", "WOA-MLP", "WDO-MLP", "MVO-MLP", "EO-MLP", "NRO-MLP"]
cols = ["model_name", "EVS", "ME", "MAE", "MSE", "RMSE", "MSLE", "MedAE", "R2", "SMAPE", "MAAPE", "MASE"]
cols_error = ["Epoch", "MSE"]
cols_predict = ["y_true", "y_pred"]

cols_error_final = ["Epoch"] + model_files2
cols_predict_final = ["Actual"] + model_files2

for datafile in data_files:
    global csv_error
    global csv_predict
    for modelfile in model_files:
        file_final = path_file + datafile + "/" + modelfile + ".csv"
        df_final = pd.read_csv(file_final, usecols=cols)
        # dataset = dataframe.values
        best_model_name = df_final["model_name"][df_final["RMSE"].argmin(axis=0)]

        file_error = path_file + datafile + "/Error-" + best_model_name + ".csv"
        file_predict = path_file + datafile + "/" + best_model_name + ".csv"

        if modelfile == "mlp":
            df_error = pd.read_csv(file_error, usecols=["Epoch", "MSE"])
            csv_error = df_error.values

            df_predict = pd.read_csv(file_predict, usecols=["y_true", "y_pred"])
            csv_predict = df_predict.values
        else:
            df_error = pd.read_csv(file_error, usecols=["MSE"])
            csv_error = np.hstack((csv_error, df_error.values))

            df_predict = pd.read_csv(file_predict, usecols=["y_pred"])
            csv_predict = np.hstack((csv_predict, df_predict.values))


    df_error = pd.DataFrame(data=csv_error, columns=cols_error_final)
    df_predict = pd.DataFrame(data=csv_predict, columns=cols_predict_final)

    df_error.to_csv(path_file + "/csv/error_best_" + datafile + ".csv", header=cols_error_final, index=False)
    df_predict.to_csv(path_file + "/csv/predict_best_" + datafile + ".csv", header=cols_predict_final, index=False)