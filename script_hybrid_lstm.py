#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:28, 11/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from models.main import hybrid_lstm
from utils.IOUtil import _load_dataset__
from utils import Settings
from utils.Settings import *
import multiprocessing
from time import time
import tensorflow as tf
import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

tf.config.threading.set_intra_op_parallelism_threads(2)  # matrix multiplication and reductions
tf.config.threading.set_inter_op_parallelism_threads(2)  # number of threads used by independent non-blocking operations


def setting_and_running(my_model):
    for N_RUNS in range(SPF_RUN_TIMES):
        for loop in range(len(SPF_DATA_FILENAME)):
            dataset = _load_dataset__(SPF_LOAD_DATA_FROM + SPF_DATA_FILENAME[loop], cols=SPF_DATA_COLS[loop])
            data_window = SPF_DATA_WINDOWS[loop]

            for hidden_sizes in SPF_HIDDEN_SIZES_HYBRID_RNN:
                for my_activations in SPF_ACTIVATIONS:
                    paras_name = my_model["name"] + "-run_" + str(N_RUNS) + "-hs_" + "".join(map(str, hidden_sizes)) + "-ac_" + "".join(my_activations) + "-"
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
                        paras_name = paras_name + "-".join([k + "_" + str(v) for k, v in paras.items()])
                        root_hybrid_paras = {
                            "hidden_sizes": hidden_sizes, "activations": my_activations, "domain_range": SPF_DOMAIN_RANGE_HYBRID, "paras_name": paras_name
                        }
                        md = getattr(hybrid_lstm, my_model["name"])(root_base_paras, root_hybrid_paras, paras)
                        md._running__()


def multiprocessing_func(model):
    print("Start running: {}\n".format(model["name"]))
    setting_and_running(model)


models = [
    {"name": "GaLstm", "param_grid": getattr(Settings, "ga_final")},
    {"name": "FpaLstm", "param_grid": getattr(Settings, "fpa_final")},
    {"name": "DeLstm", "param_grid": getattr(Settings, "de_final")},
    {"name": "PsoLstm", "param_grid": getattr(Settings, "pso_final")},
    {"name": "WoaLstm", "param_grid": getattr(Settings, "woa_final")},
    {"name": "WdoLstm", "param_grid": getattr(Settings, "wdo_final")},
    {"name": "MvoLstm", "param_grid": getattr(Settings, "mvo_final")},
    {"name": "EoLstm", "param_grid": getattr(Settings, "eo_final")},
    {"name": "NroLstm", "param_grid": getattr(Settings, "nro_final")},
    {"name": "HgsoLstm", "param_grid": getattr(Settings, "hgso_final")}
]


if __name__ == '__main__':
    starttime = time()
    processes = []
    for idx_md, my_md in enumerate(models):
        p = multiprocessing.Process(target=multiprocessing_func, args=(my_md,))
        processes.append(p)
        p.start()
        # Pin created processes in a round-robin                                # 0%8 = 0 --> core_id: 0, pid: galstm
        os.system("taskset -p -c %d %d" % ((idx_md % os.cpu_count()), p.pid))   # 1 % 8 = 1 --> core_id: 1, pid: delstm....

    for process in processes:
        process.join()

    print('That took: {} seconds'.format(time() - starttime))
