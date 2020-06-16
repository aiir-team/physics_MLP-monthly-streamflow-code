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
from statsmodels.tsa.stattools import acf, pacf
from pandas import read_csv, DataFrame

## 2 groups: 1 with value, 1 with log value
filename1 = ["full_dataset"]
x_cords = ["Timestamp"]
y_cords = ["Value"]
title = "Streamflow"
cols_name = ["value"]               # Change it if your header of csv file has different name

path_save = "acf/"
final_acf = {}
for id1, fullname in enumerate(filename1):
    df = read_csv(fullname + ".csv")

    # plt.figure(figsize=(6, 4)).suptitle(titles[it])
    # plt.xlabel(x_cords[it], fontsize=14)
    # plt.ylabel(y_cords[it], fontsize=14)
    df.plot()
    plt.savefig(path_save + "img_" + fullname + ".png", bbox_inches='tight')
    plt.show()

    # print(acf(df["value"], nlags=36))
    # print(acf(df["value"].values, nlags=36))

    final_acf["acf" + fullname] = acf(df["value"], nlags=36)
    print(acf(df["value"], nlags=36))
    tsaplots.plot_acf(df["value"], lags=36)
    plt.savefig(path_save + "acf_" + fullname + ".png", bbox_inches='tight')
    plt.show()

    final_acf["pacf" + fullname] = pacf(df["value"], nlags=36)
    print(pacf(df["value"], nlags=36))
    tsaplots.plot_pacf(df["value"], lags=36)
    plt.savefig(path_save + "pacf_" + fullname + ".png", bbox_inches='tight')
    plt.show()

df = DataFrame(final_acf)
df.to_csv("acf/acf.csv", index=True)

