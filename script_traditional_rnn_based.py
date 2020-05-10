#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:30, 10/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from models.main import traditional_rnn
from utils.IOUtil import _load_dataset__
from utils import Settings
from utils.Settings import *
import multiprocessing
from time import time
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)  # matrix multiplication and reductions
tf.config.threading.set_inter_op_parallelism_threads(2)  # number of threads used by independent non-blocking operations


def setting_and_running(my_model):
    for N_RUNS in range(SPF_RUN_TIMES):
        for loop in range(len(SPF_DATA_FILENAME)):
            dataset = _load_dataset__(SPF_LOAD_DATA_FROM + SPF_DATA_FILENAME[loop], cols=SPF_DATA_COLS[loop])
            data_window = SPF_DATA_WINDOWS[loop]
            parameters_grid = list(ParameterGrid(my_model["param_grid"]))
            for paras in parameters_grid:
                root_base_paras = {
                    "data_original": dataset,
                    "train_split": SPF_TRAIN_SPLIT,  # should use the same in all test
                    "data_window": data_window,  # same
                    "scaling": SPF_SCALING,  # minmax or std
                    "feature_size": SPF_FEATURE_SIZE,  # same, usually : 1
                    "network_type": SPF_3D_NETWORK,  # RNN-based: 3D, others: 2D
                    "log_filename": my_model["name"],
                    "path_save_result": SPF_PATH_SAVE_BASE + SPF_DATA_FILENAME[loop] + "/",
                    "draw": SPF_DRAW,
                    "log": SPF_LOG
                }
                paras_name = my_model["name"] + "-run_{}-hs_{}-ep_{}-bs_{}-lr_{}-ac_{}-op_{}-lo_{}-dr_{}".format(N_RUNS, paras["hidden_sizes"],
                    paras["epoch"], paras["batch_size"], paras["learning_rate"], paras["activations"], paras["optimizer"], paras["loss"], paras["dropouts"])
                root_rnn_paras = {
                    "hidden_sizes": paras["hidden_sizes"], "epoch": paras["epoch"], "batch_size": paras["batch_size"], "learning_rate": paras["learning_rate"],
                    "activations": paras["activations"], "optimizer": paras["optimizer"], "loss": paras["loss"], "dropouts": paras["dropouts"],
                    "paras_name": paras_name
                }
                md = getattr(traditional_rnn, my_model["name"])(root_base_paras, root_rnn_paras)
                md._running__()

def multiprocessing_func(model):
    print("Start running: {}\n".format(model["name"]))
    setting_and_running(model)

models = [
    {"name": "Rnn1HL", "param_grid": getattr(Settings, "rnn1hl_final")},
    {"name": "Lstm1HL", "param_grid": getattr(Settings, "lstm1hl_final")},
    {"name": "Gru1HL", "param_grid": getattr(Settings, "gru1hl_final")}
]

if __name__ == '__main__':
    starttime = time()
    processes = []
    for idx_md, my_md in enumerate(models):
        p = multiprocessing.Process(target=multiprocessing_func, args=(my_md,))
        processes.append(p)
        p.start()
        # Pin created processes in a round-robin                                # 0%8 = 0 --> core_id: 0, pid: rnn
        os.system("taskset -p -c %d %d" % ((idx_md % os.cpu_count()), p.pid))   # 1 % 8 = 1 --> core_id: 1, pid: lstm

    for process in processes:
        process.join()
    print('That took: {} seconds'.format(time() - starttime))
