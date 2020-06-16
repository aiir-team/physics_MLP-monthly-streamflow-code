#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 00:20, 29/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import pandas as pd
import numpy as np
import logging
logging.basicConfig(filename='acf/kurtosis.log', level=logging.DEBUG)

filenames = ["full_dataset"]

for idx, fname in enumerate(filenames):
    df = pd.read_csv(fname + '.csv', header=0)
    df['time'] = pd.to_datetime(df['time'])
    logging.debug("\nDataset: {}".format(fname))
    logging.info(df["value"].describe())
    logging.info(df["value"].isnull().sum())
    logging.info(df[df["value"] == 0].sum())
    logging.info(df[df["value"] < 0].sum())
    logging.warning("\tSkewness: {}".format(df.skew(axis=0)))
    logging.warning("\tKurtosis: {}".format(df.kurtosis(axis=0)))

