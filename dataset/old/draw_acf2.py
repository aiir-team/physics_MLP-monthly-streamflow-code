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
filename1 = ["full_dataset2.csv"]
x_cords = ["Timestamp (daily)"]
y_cords = ["MM"]
titles = ["Daily Rainfall"]
cols_name = ["value"]

path_save = "acf/"

for it in range(len(filename1)):
    df = read_csv(filename1[it])

    # plt.figure(figsize=(6, 4)).suptitle(titles[it])
    # plt.xlabel(x_cords[it], fontsize=14)
    # plt.ylabel(y_cords[it], fontsize=14)
    df.plot()
    plt.savefig(path_save + "img_" + filename1[it] + ".pdf", bbox_inches='tight')
    plt.show()


    tsaplots.plot_acf(df, lags=72)
    plt.savefig(path_save + "acf_" + filename1[it] + ".pdf", bbox_inches='tight')
    plt.show()

