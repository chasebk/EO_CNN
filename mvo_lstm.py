#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:16, 26/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from os.path import splitext, basename, realpath
from sklearn.model_selection import ParameterGrid
from models.main.hybrid_lstm import MvoLstm
from utils.IOUtil import _load_dataset__
from utils.Settings import *
from utils.Settings import mvo_lstm_final as param_grid

if SPF_RUN_TIMES == 1:
    all_model_file_name = SPF_LOG_FILENAME
else:  # If runs with more than 1, like stability test --> name of the models ==> such as: rnn1hl.csv
    all_model_file_name = str(splitext(basename(realpath(__file__)))[0])


def train_model(item):
    root_base_paras = {
        "data_original": dataset,
        "train_split": SPF_TRAIN_SPLIT,  # should use the same in all test
        "data_window": data_window,  # same
        "scaling": SPF_SCALING,  # minmax or std
        "feature_size": SPF_FEATURE_SIZE,  # same, usually : 1
        "network_type": SPF_3D_NETWORK,  # RNN-based: 3D, others: 2D
        "n_runs": SPF_RUN_TIMES,  # 1 or others
        "log_filename": all_model_file_name,
        "path_save_result": SPF_PATH_SAVE_BASE + SPF_DATA_FILENAME[loop] + "/",
        "draw": SPF_DRAW,
        "log": SPF_LOG
    }
    paras_name = "hs_{}-ac_{}-do_{}--ep_{}-ps_{}-wep_minmax_{}".format(item["hidden_sizes"], item["activations"], item["dropouts"], item["epoch"],
                                                                       item["pop_size"], item["wep_minmax"])
    root_hybrid_paras = {
        "hidden_sizes": item["hidden_sizes"], "activations": item["activations"], "dropouts": item["dropouts"],
        "domain_range": item["domain_range"], "paras_name": paras_name
    }
    mvo_paras = {"epoch": item["epoch"], "pop_size": item["pop_size"], "wep_minmax": item["wep_minmax"]}

    md = MvoLstm(root_base_paras=root_base_paras, root_hybrid_paras=root_hybrid_paras, mvo_paras=mvo_paras)
    md._running__()


for _ in range(SPF_RUN_TIMES):
    for loop in range(len(SPF_DATA_FILENAME)):
        filename = SPF_LOAD_DATA_FROM + SPF_DATA_FILENAME[loop]
        dataset = _load_dataset__(filename, cols=SPF_DATA_COLS[loop])
        feature_size = len(SPF_DATA_COLS[loop])
        data_window = SPF_DATA_WINDOWS[loop]
        # Create combination of params.
        for item in list(ParameterGrid(param_grid)):
            train_model(item)

