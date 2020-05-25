#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 13:18, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from pandas import read_csv

## 2 groups: 1 with value, 1 with log value
filename1 = ["daily_rainfall.csv", "daily_rainfall.csv"]
x_cords = ["Timestamp (daily)", "Timestamp (daily)"]
y_cords = ["MM", "MM"]
titles = ["Daily Rainfall", "Daily Log Rainfall"]
cols_name = ["value", "value_log"]

path_save = "acf/"

for it in range(len(filename1)):
    df = read_csv(filename1[it])

    # plt.figure(figsize=(6, 4)).suptitle(titles[it])
    # plt.xlabel(x_cords[it], fontsize=14)
    # plt.ylabel(y_cords[it], fontsize=14)
    df.plot()
    plt.savefig(path_save + "img_" + filename1[it] + ".pdf", bbox_inches='tight')
    plt.show()


    tsaplots.plot_acf(df[cols_name[it]], lags=72)
    plt.savefig(path_save + "acf_" + filename1[it] + ".pdf", bbox_inches='tight')
    plt.show()

