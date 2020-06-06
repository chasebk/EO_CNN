#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:59, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.layers import Dense, LSTM, Dropout, GRU, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential

from time import time
from numpy import reshape
from numpy.random import rand
from models.root.root_base import RootBase
import utils.MathUtil as my_math


class RootHybridDeepNets(RootBase):
	"""
        This is root of all hybrid recurrent neural network (meta-heuristics + RNNs)
    """

	def __init__(self, root_base_paras=None, root_hybrid_paras=None):
		RootBase.__init__(self, root_base_paras)
		self.domain_range = root_hybrid_paras["domain_range"]
		self.activations = root_hybrid_paras["activations"]
		self.dropouts = root_hybrid_paras["dropouts"]
		if root_hybrid_paras["hidden_sizes"][-1]:
			self.hidden_sizes = root_hybrid_paras["hidden_sizes"][0]
		else:
			self.hidden_sizes = 2 * root_base_paras["sliding"] * root_base_paras["feature_size"] + 1
		## New discovery
		self._activation1__ = getattr(my_math, self.activations[0])
		self._activation2__ = getattr(my_math, self.activations[1])

		self.model_rnn = None
		self.problem_size = None
		self.epoch = None

	def _building_architecture__(self):
		pass

	def _setting_paras__(self):
		weights = [rand(*w.shape) for w in self.model_rnn.get_weights()]
		problem_size = 0
		for wei in weights:
			# print(wei.shape)
			# print(len(wei.reshape(-1)))
			problem_size += len(wei.reshape(-1))
		self.problem_size = problem_size

	# self.y_train = self.y_train.reshape(-1, 1)
	# self.input_size, self.output_size = self.X_train.shape[1], self.y_train.shape[1]

	def _forecasting__(self):
		self.model_rnn.set_weights(self.model)
		y_pred = self.model_rnn.predict(self.X_test)
		y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
		y_true_unscaled = self.time_series._inverse_scaling__(self.y_test, scale_type=self.scaling)
		return y_true_unscaled, y_pred_unscaled, self.y_test, y_pred

	def _running__(self):
		self.time_system = time()
		self._processing__()
		self._building_architecture__()
		self._setting_paras__()
		self.time_total_train = time()
		self._training__()
		self._get_model__(self.solution)
		self.time_total_train = round(time() - self.time_total_train, 4)
		self.time_epoch = round(self.time_total_train / self.epoch, 4)
		self.time_predict = time()
		y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled = self._forecasting__()
		self.time_predict = round(time() - self.time_predict, 8)
		self.time_system = round(time() - self.time_system, 4)
		self._save_results__(y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled, self.loss_train, self.n_runs)

	## Helper functions
	def _get_model__(self, individual=None):
		weights = [rand(*w.shape) for w in self.model_rnn.get_weights()]
		ws = []
		cur_point = 0
		for wei in weights:
			ws.append(reshape(individual[cur_point:cur_point + len(wei.reshape(-1))], wei.shape))
			cur_point += len(wei.reshape(-1))
		self.model = ws

	def _get_average_error__(self, individual=None, X_data=None, y_data=None):
		t1 = time()
		weights = [rand(*w.shape) for w in self.model_rnn.get_weights()]
		ws = []
		cur_point = 0
		for wei in weights:
			ws.append(reshape(individual[cur_point:cur_point + len(wei.reshape(-1))], wei.shape))
			cur_point += len(wei.reshape(-1))

		self.model_rnn.set_weights(ws)
		y_pred = self.model_rnn.predict(X_data)
		# print("GAE time: {}".format(time() - t1))

		# return [mean_squared_error(y_pred, y_data), mean_absolute_error(y_pred, y_data)]
		return mean_squared_error(y_pred, y_data)

	def _objective_function__(self, solution=None):
		weights = [rand(*w.shape) for w in self.model_rnn.get_weights()]
		ws = []
		cur_point = 0
		for wei in weights:
			ws.append(reshape(solution[cur_point:cur_point + len(wei.reshape(-1))], wei.shape))
			cur_point += len(wei.reshape(-1))

		self.model_rnn.set_weights(ws)
		y_pred = self.model_rnn.predict(self.X_train)
		return mean_squared_error(y_pred, self.y_train)


class RootHybridCnn(RootHybridDeepNets):

	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None):
		RootHybridDeepNets.__init__(self, root_base_paras, root_hybrid_paras)
		self.filters_size = cnn_paras["filters_size"]
		self.kernel_size = cnn_paras["kernel_size"]
		self.pool_size = cnn_paras["pool_size"]

	def _building_architecture__(self):
		#  The CNN 1-HL architecture
		self.model_rnn = Sequential()
		self.model_rnn.add(Conv1D(filters=self.filters_size, kernel_size=self.kernel_size, activation=self.activations[0],
		                          input_shape=(self.X_train.shape[1], 1)))
		self.model_rnn.add(MaxPooling1D(pool_size=self.pool_size))
		self.model_rnn.add(Flatten())
		self.model_rnn.add(Dense(self.hidden_sizes[0], activation=self.activations[1]))
		self.model_rnn.add(Dense(1))



