#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:08, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.root_base import RootBase
from time import time


class RootRnn(RootBase):
	"""
		This is the root of all RNN-based models like RNNs, LSTMs, GRUs,...
	"""

	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		RootBase.__init__(self, root_base_paras)
		self.epoch = root_rnn_paras["epoch"]
		self.batch_size = root_rnn_paras["batch_size"]
		self.learning_rate = root_rnn_paras["learning_rate"]
		self.activations = root_rnn_paras["activations"]
		self.optimizer = root_rnn_paras["optimizer"]
		self.loss = root_rnn_paras["loss"]
		self.dropouts = root_rnn_paras["dropouts"]

		## if you want to set the number of hidden sizes by yourself --> [N1, N2, ..., True]
		## if not, ---> [None, None, ..., False]
		##     Our settings would be: assumpt 4 hidden layers
		##          input:      n ( = sliding * feature_size )
		##          1st layer: 4*n + 1
		##          2nd layer: 3*n + 1
		##          3rd layer: 2*n + 1
		##          4th layer: n + 1
		##          output:     1
		if root_rnn_paras["hidden_sizes"][-1]:
			self.hidden_sizes = root_rnn_paras["hidden_sizes"][:-1]
		else:
			num_hid = len(root_rnn_paras["hidden_sizes"]) - 1
			self.hidden_sizes = [(num_hid - i) * root_base_paras["sliding"] * root_base_paras["feature_size"] + 1 for i in range(num_hid)]

	def _forecasting__(self):
		y_pred = self.model.predict(self.X_test)
		y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
		y_true_unscaled = self.time_series._inverse_scaling__(self.y_test, scale_type=self.scaling)
		return y_true_unscaled, y_pred_unscaled, self.y_test, y_pred

	def _running__(self):
		self.time_system = time()
		self._processing__()
		self.time_total_train = time()
		self._training__()
		self.time_total_train = round(time() - self.time_total_train, 4)
		self.time_epoch = round(self.time_total_train / self.epoch, 4)
		self.time_predict = time()
		y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled = self._forecasting__()
		self.time_predict = round(time() - self.time_predict, 8)
		self.time_system = round(time() - self.time_system, 4)
		self._save_results__(y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled, self.loss_train, self.n_runs)

