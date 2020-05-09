#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:37, 31/01/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

SPF_RUN_TIMES = 1
SPF_2D_NETWORK = "2D"
SPF_3D_NETWORK = "3D"
SPF_SCALING = "minmax"
SPF_FEATURE_SIZE = 1
SPF_TRAIN_SPLIT = 0.75
SPF_PATH_SAVE_BASE = "history/results1/"
SPF_DRAW = True
SPF_LOG = 0  # 0: nothing, 1 : full detail, 2: short version

SPF_LOAD_DATA_FROM = "dataset/paper/"

SPF_DATA_FILENAME = ["daily_rainfall_22022020", "weekly_rainfall_22022020"]
SPF_DATA_COLS = [[1], [1]]
SPF_DATA_WINDOWS = [(1, 2, 3), (1, 49, 50)]  # Using ACF to determine which one will used


###### Setting for paper running on server ==============================
epochs = [100]
activations = [("elu", "elu")]

hidden_sizes1 = [(20, True), ]  # (num_node, checker), default checker is True
learning_rates = [0.2]
optimizers = ['sgd']  ## sgd = SGD, adam = Adam,  adagrad = Adagrad, adadelta = Adadelta, rmsprop = RMSprop, adamax = Adamax, nadam = Nadam
losses = ["mse"]
batch_sizes = [64]
dropouts = [(0.2,)]

hidden_sizes2 = [([7, ], True), ]
pop_sizes = [50]
hidden_sizes11 = [(7, True), ]  # (num_node, checker), default checker is True
domain_ranges = [(-1, 1)]


###================= Settings models for paper ============================####


####: MLNN-1HL
mlnn1hl_final = {
	"hidden_sizes": hidden_sizes1,
	"activations": activations,
	"learning_rate": learning_rates,
	"epoch": epochs,
	"batch_size": batch_sizes,
	"optimizer": optimizers,
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
	"loss": losses,
	"dropouts": dropouts
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

#### : NRO-LSTM
nro_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : HGSO-LSTM
hgso_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"n_clusters": [2, ],
	"domain_range": domain_ranges
}

#### : ASO-LSTM
aso_lstm_final = {
	"hidden_sizes": hidden_sizes2,
	"activations": activations,
	"dropouts": dropouts,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"alpha": [50],
	"beta": [0.2],
	"domain_range": domain_ranges
}

#### ============== Hybrid MLP ==============================######

#### : GA-MLP
ga_mlp_final = {
	"hidden_size": hidden_sizes11,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"pc": [0.95],  # 0.85 -> 0.97
	"pm": [0.025],  # 0.005 -> 0.10
	"domain_range": domain_ranges
}

#### : DE-MLP
de_mlp_final = {
	"hidden_size": hidden_sizes11,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"wf": [0.8],
	"cr": [0.9],
	"domain_range": domain_ranges
}

#### : PSO-MLP
pso_mlp_final = {
	"hidden_size": hidden_sizes11,
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
	"hidden_size": hidden_sizes11,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : WDO-MLP
wdo_mlp_final = {
	"hidden_size": hidden_sizes11,
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
	"hidden_size": hidden_sizes11,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"wep_minmax": [(1.0, 0.2), ],
	"domain_range": domain_ranges
}

#### : EO-MLP
eo_mlp_final = {
	"hidden_size": hidden_sizes11,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : NRO-MLP
nro_mlp_final = {
	"hidden_size": hidden_sizes11,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"domain_range": domain_ranges
}

#### : HGSO-MLP
hgso_mlp_final = {
	"hidden_size": hidden_sizes11,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"n_clusters": [2, ],
	"domain_range": domain_ranges
}

#### : ASO-MLP
aso_mlp_final = {
	"hidden_size": hidden_sizes11,
	"activations": activations,

	"epoch": epochs,
	"pop_size": pop_sizes,
	"alpha": [50],
	"beta": [0.2],
	"domain_range": domain_ranges
}