#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:12, 28/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import reshape, array

def _univariate_data__(dataset, history_column=None, start_index=0, end_index=None, pre_type="2D"):
    """
    :param dataset: 2-D numpy array
    :param start_index: 0- training set, N- valid or testing set
    :param end_index: N-training or valid set, None-testing set
    :param history_size: sliding window, 1 mean t-1, 2 mean t-2, 3 mean t-3,...
    :param target_size: 1 mean t, 2 mean t+1, 3 mean t+2
    :param type: 3D for RNN-based, 2D for normal neural network like MLP, FFLN,..
    :return:
    """
    data = []
    labels = []

    history_size = len(history_column)
    if end_index is None:
        end_index = len(dataset) - history_column[-1] # for time t, such as: t-1, t-4, t-7 and finally t
    else:
        end_index = end_index - history_column[-1]

    for i in range(start_index, end_index):
        indices = i - 1 + array(history_column)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + history_column[-1]])
    if pre_type == "3D":
        return array(data), array(labels)
    return reshape(array(data), (-1, history_size)), array(labels)


from utils.IOUtil import _load_dataset__

filename = "../dataset/paper/daily_rainfall_22022020"
dataset = _load_dataset__(filename, cols=[1])


dataset, labels = _univariate_data__(dataset, [1, 2, 5], 0, 50, "2D")
print(labels)
print(dataset)


