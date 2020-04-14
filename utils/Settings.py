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
SPF_PATH_SAVE_BASE = "history/results_final_2/"
SPF_DRAW = True
SPF_LOG = 0  # 0: nothing, 1 : full detail, 2: short version

SPF_LOAD_DATA_FROM = "data/paper/"

SPF_DATA_FILENAME = ["f_occupancy_t4013", "f_speed_7578", "f_TravelTime_451"]
SPF_DATA_COLS = [[1], [1], [1]]
SPF_DATA_WINDOWS = [(1,2,3,4,5,6,7,8,9,10,11,12), (1,2,3,4,5), (1,2,3,4)]  # Using ACF to determine which one will used


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

## For CNN networks
filters_sizes = [8, ]
kernel_sizes = [2, ]
pool_sizes = [2, ]


###================= Settings models for drafts ==============================#####


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

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes
}

###================= Settings models for paper hybrid-CNN ============================####

#### : GA-CNN
ga_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}

#### : WOA-CNN
woa_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : MVO-CNN
mvo_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"wep_minmax": [(1.0, 0.2), ],
	"domain_range": domain_ranges
}

#### : SBO-CNN
sbo_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"alpha": [0.94],
	"pm": [0.05],
	"z": [0.02],
	"domain_range": domain_ranges
}

#### : SSDO-CNN
ssdo_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : LCBO-CNN
lcbo_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : EO-CNN
eo_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : AEO-CNN
aeo_cnn_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"filters_size": filters_sizes,
	"kernel_size": kernel_sizes,
	"pool_size": pool_sizes,

	"epoch": epochs,
	"pop_size": pop_sizes,
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
