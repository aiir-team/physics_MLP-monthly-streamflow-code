#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:37, 25/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import pandas as pd
import matplotlib.pyplot as plt

file_name = "daily_rainfall_22022020.csv"
df = pd.read_csv(file_name, usecols=["value"], header=0)
df = df[df["value"] != 0]

df.to_csv("daily_rainfall.csv", header=["value"])


