#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:46, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from os.path import splitext, basename, realpath
from sklearn.model_selection import ParameterGrid
from models.main.traditional_sonia import Sonia
from utils.IOUtil import _load_dataset__
from utils.Settings import *
from utils.Settings import sonia_final as param_grid

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
        "network_type": SPF_2D_NETWORK,  # RNN-based: 3D, others: 2D
        "n_runs": SPF_RUN_TIMES,  # 1 or others
        "log_filename": all_model_file_name,
        "path_save_result": SPF_PATH_SAVE_BASE + SPF_DATA_FILENAME[loop] + "/",
        "draw": SPF_DRAW,
        "log": SPF_LOG
    }
    paras_name = "sl_{}-pn_{}-dl_{}-mi_{}-net_paras-{}-{}-{}-{}-{}-{}-{}".format(item["stimulation_level"], item["positive_number"], item["distance_level"],
        item["mutation_id"], item["epoch"], item["batch_size"], item["learning_rate"], item["activations"],
        item["optimizer"], item["loss"], item["clustering_type"])

    clustering_paras = {
        "stimulation_level": item["stimulation_level"], "positive_number": item["positive_number"],
        "distance_level": item["distance_level"], "max_cluster": item["max_cluster"], "mutation_id": item["mutation_id"]
    }
    root_ssnn_paras = {
        "clustering_type": item["clustering_type"], "epoch": item["epoch"], "batch_size": item["batch_size"], "learning_rate": item["learning_rate"],
        "activations": item["activations"], "optimizer": item["optimizer"], "loss": item["loss"], 'paras_name': paras_name
    }
    md = Sonia(root_base_paras=root_base_paras, clustering_paras=clustering_paras, root_ssnn_paras=root_ssnn_paras)
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

