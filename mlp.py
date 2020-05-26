#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:29, 23/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import os
import platform
from os.path import splitext, basename, realpath
from sklearn.model_selection import ParameterGrid
from models.main.traditional_mlp import Mlnn1HL
from utils.IOUtil import _load_dataset__
from utils.Settings import *
from utils.Settings import mlnn1hl_final as param_grid
import time
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)  # matrix multiplication and reductions
tf.config.threading.set_inter_op_parallelism_threads(2)  # number of threads used by independent non-blocking operations

# if platform.system() == "Linux":  # Linux: "Linux", Mac: "Darwin", Windows: "Windows"
#     os.sched_setaffinity(0, {1})

# name of the models ==> such as: rnn1hl.csv
model_name = str(splitext(basename(realpath(__file__)))[0])


def train_model(item):
    root_base_paras = {
        "data_original": outside_dataset,
        "train_split": SPF_TRAIN_SPLIT,  # should use the same in all test
        "data_window": outside_data_window,  # same
        "scaling": SPF_SCALING,  # minmax or std
        "feature_size": SPF_FEATURE_SIZE,  # same, usually : 1
        "network_type": SPF_2D_NETWORK,  # RNN-based: 3D, others: 2D
        "log_filename": outside_log_filename,
        "path_save_result": outside_path_save_results,
        "draw": SPF_DRAW,
        "log": SPF_LOG
    }
    paras_name = outside_log_filename.upper() + "-run_{}-hs_{}-ep_{}-bs_{}-lr_{}-ac_{}-op_{}-lo_{}".format(N_RUNS, item["hidden_sizes"], item["epoch"],
                                                                                                          item["batch_size"], item["learning_rate"],
                                                                                                          item["activations"], item["optimizer"], item["loss"])
    root_mlp_paras = {
        "hidden_sizes": item["hidden_sizes"], "epoch": item["epoch"], "batch_size": item["batch_size"], "learning_rate": item["learning_rate"],
        "activations": item["activations"], "optimizer": item["optimizer"], "loss": item["loss"], "paras_name": paras_name
    }
    md = Mlnn1HL(root_base_paras=root_base_paras, root_mlp_paras=root_mlp_paras)
    md._running__()


start_time = time.time()
for id_file in range(len(SPF_DATA_FILENAME)):
    outside_data_windows = SPF_DATA_WINDOWS[id_file]
    outside_dataset = _load_dataset__(SPF_LOAD_DATA_FROM + SPF_DATA_FILENAME[id_file], cols=SPF_DATA_COLS[id_file])
    for id_window in range(len(outside_data_windows)):
        outside_path_save_results = SPF_PATH_SAVE_BASE + SPF_DATA_FILENAME[id_file] + "/"
        outside_data_window = outside_data_windows[id_window]
        outside_log_filename = model_name.lower() + str(id_window)
        for N_RUNS in range(SPF_RUN_TIMES):
            # Create combination of params.
            for item in list(ParameterGrid(param_grid)):
                train_model(item)
end_time = time.time() - start_time
print("Taken: {} seconds".format(end_time))

