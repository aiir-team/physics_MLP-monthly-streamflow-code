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

file_names = ["gg_cpu", "gg_ram", "it_eu_5m", "it_uk_5m", "worldcup98_5m"]
cols = [["meanCPUUsage"], ["CanonicalMemUsage"], ["traffic (in Megabyte)"], ["bytes"], ["count_thousand"]]

for idx, filename in enumerate(file_names):
    df = pd.read_csv(filename + ".csv", usecols=cols[idx], header=0)
    print(df.describe())
    print("Skewness: {}".format(df[cols[idx][0]].skew(axis=0)))
    print("Kurtosis: {}".format(df[cols[idx][0]].kurtosis(axis=0)))


