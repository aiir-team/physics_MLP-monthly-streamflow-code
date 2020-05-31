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
SPF_SCALING = "minmax"      # minmax, std, loge
SPF_FEATURE_SIZE = 1
SPF_TRAIN_SPLIT = 0.75
SPF_PATH_SAVE_BASE = "history/results/"
SPF_DRAW = True
SPF_LOG = 0  # 0: nothing, 1 : full detail, 2: short version

SPF_LOAD_DATA_FROM = "dataset/test/"

# SPF_DATA_FILENAME = ["daily_rainfall_22022020", "weekly_rainfall_22022020"]
# SPF_DATA_COLS = [[1], [1]]
# SPF_DATA_WINDOWS = [(1, 2, 3), (1, 49, 50, 54)]  # Using ACF to determine which one will used

# SPF_DATA_FILENAME = ["full_dataset2"]
# SPF_DATA_COLS = [[0]]
# SPF_DATA_WINDOWS = [[1, 11, 12, 13, 23, 24, 25]]  # Using ACF to determine which one will used

SPF_DATA_FILENAME = ["full_dataset"]
SPF_DATA_COLS = [[1]]
SPF_DATA_WINDOWS = [
	[   ## full_dataset
		[1, 4, 5, 6, 7, 8, 11, 12]
	]
]
# SPF_DATA_WINDOWS = [
# 	[
# 		[1,5,6,7,11,12,13],
# 		[1,2,10,11,12],
# 		[1,2,3,4,5,6,7,8,9,10,11,12],
# 		[1,4,5,6,7,8,11,12]
# 	],
#
# ]  # Using ACF to determine which one will used

## Default settings
SPF_HIDDEN_SIZES_HYBRID = [(7, False), ]             # (num_node, checker), default checker is True
SPF_DOMAIN_RANGE_HYBRID = (-1, 1)                   # For all hybrid models
SPF_ACTIVATIONS = [("elu", "elu")]

SPF_HIDDEN_SIZES_HYBRID_RNN = [([7, ], False), ]     # For hybrid LSTM

###### Setting for paper running on server ==============================
epochs = [1000]
hidden_sizes_traditional = [(20, False), ]  # (num_node, checker), default checker is True
learning_rates = [0.1]
optimizers = ['SGD']  ## SGD, Adam, Adagrad, Adadelta, RMSprop, Adamax, Nadam
losses = ["mse"]
batch_sizes = [32]
dropouts = [(0.2,)]
pop_sizes = [50]

###================= Settings models for paper ============================####

####: MLNN-1HL
mlnn1hl_final = {
	"hidden_sizes": [(20, False)],      # True: 20, False: 2*n+1
	"activations": SPF_ACTIVATIONS,
	"learning_rate": [0.01],
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'],
	"loss": losses
}

####: MLP
mlp_final = {
	"hidden_sizes": hidden_sizes_traditional,
	"activations": SPF_ACTIVATIONS,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": ['sgd'],
	"loss": losses
}

####: RNN-1HL
rnn1hl_final = {
	"hidden_sizes": hidden_sizes_traditional,
	"activations": SPF_ACTIVATIONS,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
	"loss": ['mse'],
	"dropouts": dropouts
}

####: LSTM-1HL
lstm1hl_final = {
	"hidden_sizes": hidden_sizes_traditional,
	"activations": SPF_ACTIVATIONS,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
	"loss": losses,
	"dropouts": dropouts
}

####: GRU-1HL
gru1hl_final = {
	"hidden_sizes": hidden_sizes_traditional,
	"activations": SPF_ACTIVATIONS,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
	"loss": losses,
	"dropouts": dropouts
}

#### ============== Hybrid MLP/RNN/LSTM/GRU/CNN ==============================######

#### : FPA-MLP/RNN/LSTM/GRU/CNN
fpa_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"p": [0.8]
}

#### : GA-MLP/RNN/LSTM/GRU/CNN
ga_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.9],  # 0.85 -> 0.97
	"pm": [0.1]  # 0.005 -> 0.10
}

#### : DE-MLP/RNN/LSTM/GRU/CNN
de_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"wf": [0.8],
	"cr": [0.9]
}

#### : PSO-MLP/RNN/LSTM/GRU/CNN
pso_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"c1": [2.0],
	"c2": [2.0],
	"w_min": [0.4],
	"w_max": [0.9]
}

#### : WOA-MLP/RNN/LSTM/GRU/CNN
woa_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}

#### : WDO-MLP/RNN/LSTM/GRU/CNN
wdo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"RT": [3],
	"g": [0.2],
	"alp": [0.4],
	"c": [0.4],
	"max_v": [0.3]
}

#### : MVO-MLP/RNN/LSTM/GRU/CNN
mvo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"wep_minmax": [(1.0, 0.2), ]
}

#### : EO-MLP/RNN/LSTM/GRU/CNN
eo_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}

#### : NRO-MLP/RNN/LSTM/GRU/CNN
nro_final = {
	"epoch": epochs,
	"pop_size": pop_sizes
}

#### : HGSO-MLP/RNN/LSTM/GRU/CNN
hgso_final = {
	"epoch": epochs,
	"pop_size": pop_sizes,
	"n_clusters": [2, ]
}
