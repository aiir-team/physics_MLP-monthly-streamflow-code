#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:18, 25/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import pandas as pd
import matplotlib.pyplot as plt

file_name = "daily_rainfall.csv"
df = pd.read_csv(file_name, usecols=["value"], header=0)
print(df.describe())
print("Skewness: {}".format(df["value"].skew(axis=0)))
print("Kurtosis: {}".format(df["value"].kurtosis(axis=0)))


