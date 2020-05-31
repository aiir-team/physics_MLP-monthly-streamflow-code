#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:13, 10/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from models.main import hybrid_mlp
from utils.IOUtil import _load_dataset__
from utils import Settings
from utils.Settings import *
import multiprocessing
from time import time


def setting_and_running_multi_processing(my_model):
    print("Start running: {}\n".format(my_model["name"]))

    for f_id, f_name in enumerate(SPF_DATA_FILENAME):
        outside_data_windows = SPF_DATA_WINDOWS[f_id]
        outside_dataset = _load_dataset__(SPF_LOAD_DATA_FROM + SPF_DATA_FILENAME[f_id], cols=SPF_DATA_COLS[f_id])

        for w_id, w_name in enumerate(outside_data_windows):
            outside_path_save_results = SPF_PATH_SAVE_BASE + SPF_DATA_FILENAME[f_id] + "/"
            outside_data_window = outside_data_windows[w_id]
            outside_log_filename = my_model["name"] + "-" + str(w_id)
            for N_RUNS in range(SPF_RUN_TIMES):

                for hidden_size in SPF_HIDDEN_SIZES_HYBRID:
                    for activation in SPF_ACTIVATIONS:
                        paras_name = outside_log_filename + "-run_" + str(N_RUNS) + "-hs_" + "".join(map(str, hidden_size)) + "-ac_" + "".join(activation) + "-"
                        parameters_grid = list(ParameterGrid(my_model["param_grid"]))
                        for paras in parameters_grid:
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
                            paras_name = paras_name + "-".join([k + "_" + str(v) for k, v in paras.items()])
                            root_hybrid_paras = {
                                "hidden_size": hidden_size, "activations": activation, "domain_range": SPF_DOMAIN_RANGE_HYBRID, "paras_name": paras_name
                            }
                            md = getattr(hybrid_mlp, my_model["name"])(root_base_paras, root_hybrid_paras, paras)
                            md._running__()

models = [
    {"name": "GaMlp", "param_grid": getattr(Settings, "ga_final")},
    {"name": "FpaMlp", "param_grid": getattr(Settings, "fpa_final")},
    {"name": "DeMlp", "param_grid": getattr(Settings, "de_final")},
    {"name": "PsoMlp", "param_grid": getattr(Settings, "pso_final")},
    {"name": "WoaMlp", "param_grid": getattr(Settings, "woa_final")},
    {"name": "WdoMlp", "param_grid": getattr(Settings, "wdo_final")},
    {"name": "MvoMlp", "param_grid": getattr(Settings, "mvo_final")},
    {"name": "EoMlp", "param_grid": getattr(Settings, "eo_final")},
    {"name": "NroMlp", "param_grid": getattr(Settings, "nro_final")},
    {"name": "HgsoMlp", "param_grid": getattr(Settings, "hgso_final")}
]

if __name__ == '__main__':
    starttime = time()
    processes = []
    for my_md in models:
        p = multiprocessing.Process(target=setting_and_running_multi_processing, args=(my_md,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took: {} seconds'.format(time() - starttime))
