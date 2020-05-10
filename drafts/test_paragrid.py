#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:34, 10/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid

param_grid = {
    "thieu": [1, 2, 6, 4],
    "hong_anh": [3, 20, 11],
}
item_list = list(ParameterGrid(param_grid))
print(item_list)
# for item in list(ParameterGrid(param_grid)):
#     print(item)