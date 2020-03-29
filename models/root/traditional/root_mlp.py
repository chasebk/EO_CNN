#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:44, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.root_base import RootBase
from time import time


class RootMlp(RootBase):
	"""
		This is the root of all Multi-layer Perceptron-based models like FFNN, ANN, MLNN,...
	"""

	def __init__(self, root_base_paras=None, root_mlp_paras=None):
		RootBase.__init__(self, root_base_paras)
		self.epoch = root_mlp_paras["epoch"]
		self.batch_size = root_mlp_paras["batch_size"]
		self.learning_rate = root_mlp_paras["learning_rate"]
		self.activations = root_mlp_paras["activations"]
		self.optimizer = root_mlp_paras["optimizer"]
		self.loss = root_mlp_paras["loss"]
		if root_mlp_paras["hidden_sizes"][-1]:
			self.hidden_sizes = root_mlp_paras["hidden_sizes"][:-1]
		else:
			num_hid = len(root_mlp_paras["hidden_sizes"]) - 1
			self.hidden_sizes = [(num_hid - i) * root_base_paras["sliding"] * root_base_paras["feature_size"] + 1 for i in range(num_hid)]

	def _forecasting__(self):
		# Evaluate models on the test set
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
