#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:37, 31/01/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

SPF_RUN_TIMES = 10
SPF_2D_NETWORK = "2D"
SPF_3D_NETWORK = "3D"
SPF_SCALING = "minmax"
SPF_FEATURE_SIZE = 1
SPF_TRAIN_SPLIT = 0.75
SPF_LOG_FILENAME = "LOG_MODELS"
SPF_PATH_SAVE_BASE = "history/results_final/"
SPF_DRAW = True
SPF_LOG = 0  # 0: nothing, 1 : full detail, 2: short version

SPF_LOAD_DATA_FROM = "data/paper/"

SPF_DATA_FILENAME = ["f_occupancy_t4013", "f_speed_7578", "f_TravelTime_451"]
SPF_DATA_COLS = [[1], [1], [1]]
SPF_DATA_WINDOWS = [(1,2,3,4,5,6,7,8,9,10,11,12), (1,2,3,4,5), (1,2,3,4)]  # Using ACF to determine which one will used

# 0.4 -> 1-12, 0.5 -> 1-8
# 0.4 -> 1-5, 0.5 -> 1-3
# 0.4 -> 1-4, 0.5 -> 1-3
# > upper: 1-33, 1-10 - 29-40, 1-14     (very bad results)
# 0.57 -> 1-2 all  -> not good as 0.4


###### Setting for paper running on server ==============================
epochs = [1000]
activations = [("elu", "elu")]

hidden_sizes1 = [(7, True), ]  # (num_node, checker), default checker is True
learning_rates = [0.01]
optimizers = ['sgd']
losses = ["mse"]
batch_sizes = [128]
dropouts = [(0.2,)]

hidden_sizes2 = [([7, ], True), ]
pop_sizes = [50]
domain_ranges = [(-1, 1)]

## For ELM network
elm_activation = ['elu']

###================= Settings models for drafts ==============================#####


#### : Immune-SONIA
sonia_final = {
	"clustering_type": ['immune'],  # immune_full: cluster + mutation, else: cluster
	"stimulation_level": [0.15, 0.25, 0.5],
	"positive_number": [0.25],
	"distance_level": [0.15],

	"max_cluster": [500],  # default
	"mutation_id": [0],  # default

	"epoch": epochs,
	"batch_size": batch_sizes,
	"learning_rate": learning_rates,
	"activations": activations,
	"optimizer": optimizers,
	"loss": losses
}



#### : ELM
elm_final = {
	"hidden_size": hidden_sizes1,
	"activation": elm_activation
}


####: GRU-1HL
gru1hl_final = {
	"hidden_sizes": hidden_sizes1,
	"activations": activations,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
	"loss": losses,
	"dropouts": dropouts
}

####: CNN-1
cnn1_final = {
	"hidden_sizes": hidden_sizes1,
	"activations": activations,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
	"loss": losses,
	"dropouts": dropouts,

	"filters_size": [64, ],
	"kernel_size": [2, ],
	"pool_size": [2, ]
}

#### : GA-RNN
ga_rnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}

#### : GA-GRU
ga_gru_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}

#### : GA-CNN
ga_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": [64, ],
	"kernel_size": [2, ],
	"pool_size": [2],

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}

###================= Settings models for paper ============================####


####: MLNN-1HL
mlnn1hl_final = {
	"hidden_sizes": hidden_sizes1,
	"activations": activations,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
	## Optimizer like keras: sgd = SGD, adam = Adam,  adagrad = Adagrad, adadelta = Adadelta, rmsprop = RMSprop, adamax = Adamax, nadam = Nadam
	"loss": losses
}

####: RNN-1HL
rnn1hl_final = {
	"hidden_sizes": hidden_sizes1,
	"activations": activations,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
	## Optimizer like keras: sgd = SGD, adam = Adam,  adagrad = Adagrad, adadelta = Adadelta, rmsprop = RMSprop, adamax = Adamax, nadam = Nadam
	"loss": ['mse'],
	"dropouts": dropouts
}

####: LSTM-1HL
lstm1hl_final = {
	"hidden_sizes": hidden_sizes1,
	"activations": activations,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
	## Optimizer like keras: sgd = SGD, adam = Adam,  adagrad = Adagrad, adadelta = Adadelta, rmsprop = RMSprop, adamax = Adamax, nadam = Nadam
	"loss": losses,
	"dropouts": dropouts
}

#### ============== Hybrid LSTM ==============================######

#### : GA-LSTM
ga_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}

#### : DE-LSTM
de_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"wf": [0.8],
	"cr": [0.9],
	"domain_range": domain_ranges
}

#### : PSO-LSTM
pso_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"c1": [2.0],
	"c2": [2.0],
	"w_min": [0.4],
	"w_max": [0.9],
	"domain_range": domain_ranges
}

#### : WOA-LSTM
woa_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : WDO-LSTM
wdo_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"RT": [3],
	"g": [0.2],
	"alp": [0.4],
	"c": [0.4],
	"max_v": [0.3],
	"domain_range": domain_ranges
}

#### : MVO-LSTM
mvo_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"wep_minmax": [(1.0, 0.2), ],
	"domain_range": domain_ranges
}

#### : EO-LSTM
eo_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### ============== Hybrid MLP ==============================######

#### : GA-MLP
ga_mlp_final = {
	"hidden_size": hidden_sizes1,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}

#### : DE-MLP
de_mlp_final = {
	"hidden_size": hidden_sizes1,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"wf": [0.8],
	"cr": [0.9],
	"domain_range": domain_ranges
}

#### : PSO-MLP
pso_mlp_final = {
	"hidden_size": hidden_sizes1,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"c1": [1.2],
	"c2": [1.2],
	"w_min": [0.4],
	"w_max": [0.9],
	"domain_range": domain_ranges
}

#### : WOA-MLP
woa_mlp_final = {
	"hidden_size": hidden_sizes1,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : WDO-MLP
wdo_mlp_final = {
	"hidden_size": hidden_sizes1,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"RT": [3],
	"g": [0.2],
	"alp": [0.4],
	"c": [0.4],
	"max_v": [0.3],
	"domain_range": domain_ranges
}

#### : MVO-MLP
mvo_mlp_final = {
	"hidden_size": hidden_sizes1,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"wep_minmax": [(1.0, 0.2), ],
	"domain_range": domain_ranges
}

#### : EO-MLP
eo_mlp_final = {
	"hidden_size": hidden_sizes1,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### ============== Hybrid ELM ==============================######

#### : GA-ELM
ga_elm_final = {
	"hidden_size": hidden_sizes1,
	"activation": elm_activation,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}


# ================ Hybrid SONIA : Cluster + Mutation + Algorithm ============================

#### : GA-SONIA
ga_sonia_final = {
	"stimulation_level": [0.25],
	"positive_number": [0.10],
	"distance_level": [0.5],
	"activations": activations,

	"max_cluster": [500],  # default
	"mutation_id": [0],  # default
	"clustering_type": ['immune_full', 'immune'],  # immune_full: cluster + mutation, immune: cluster

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}

