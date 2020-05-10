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

def setting_and_running(my_model):

    for N_RUNS in range(SPF_RUN_TIMES):
        for loop in range(len(SPF_DATA_FILENAME)):
            dataset = _load_dataset__(SPF_LOAD_DATA_FROM + SPF_DATA_FILENAME[loop], cols=SPF_DATA_COLS[loop])
            data_window = SPF_DATA_WINDOWS[loop]

            for hidden_size in SPF_HIDDEN_SIZES_HYBRID:
                for activation in SPF_ACTIVATIONS:
                    paras_name = my_model["name"] + "-run_" + str(N_RUNS) + "-hs_" + "".join(map(str, hidden_size)) + "-ac_" + "".join(activation) + "-"
                    parameters_grid = list(ParameterGrid(my_model["param_grid"]))
                    for paras in parameters_grid:
                        root_base_paras = {
                            "data_original": dataset,
                            "train_split": SPF_TRAIN_SPLIT,  # should use the same in all test
                            "data_window": data_window,  # same
                            "scaling": SPF_SCALING,  # minmax or std
                            "feature_size": SPF_FEATURE_SIZE,  # same, usually : 1
                            "network_type": SPF_2D_NETWORK,  # RNN-based: 3D, others: 2D
                            "log_filename": my_model["name"],
                            "path_save_result": SPF_PATH_SAVE_BASE + SPF_DATA_FILENAME[loop] + "/",
                            "draw": SPF_DRAW,
                            "log": SPF_LOG
                        }
                        paras_name = paras_name + "-".join([k + "_" + str(v) for k, v in paras.items()])
                        root_hybrid_paras = {
                            "hidden_size": hidden_size, "activations": activation, "domain_range": SPF_DOMAIN_RANGE_HYBRID, "paras_name": paras_name
                        }
                        md = getattr(hybrid_mlp, my_model["model"])(root_base_paras, root_hybrid_paras, paras)
                        md._running__()


def multiprocessing_func(model):
    print("Start running: {}\n".format(model["name"]))
    setting_and_running(model)

models = [
    {"name": "GaMlp", "param_grid": getattr(Settings, "ga_mlp_final")},
    {"name": "DeMlp", "param_grid": getattr(Settings, "de_mlp_final")},
    {"name": "PsoMlp", "param_grid": getattr(Settings, "pso_mlp_final")},
    {"name": "WoaMlp", "param_grid": getattr(Settings, "woa_mlp_final")},
    {"name": "WdoMlp", "param_grid": getattr(Settings, "wdo_mlp_final")},
    {"name": "MvoMlp", "param_grid": getattr(Settings, "mvo_mlp_final")},
    {"name": "EoMlp", "param_grid": getattr(Settings, "eo_mlp_final")},
    {"name": "NroMlp", "param_grid": getattr(Settings, "nro_mlp_final")},
    {"name": "HgsoMlp", "param_grid": getattr(Settings, "hgso_mlp_final")},
    {"name": "AsoMlp", "param_grid": getattr(Settings, "aso_mlp_final")},
]

if __name__ == '__main__':
    starttime = time()
    processes = []
    for my_md in models:
        p = multiprocessing.Process(target=multiprocessing_func, args=(my_md,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took: {} seconds'.format(time() - starttime))
