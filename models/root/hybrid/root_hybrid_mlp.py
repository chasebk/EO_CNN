#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:39, 26/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import reshape, add, matmul
from time import time
from models.root.root_base import RootBase
import utils.MathUtil as my_math


class RootHybridMlp(RootBase):
    """
        This is root of all hybrid multi-layer perceptron (meta-heuristics + MLP)
    """

    def __init__(self, root_base_paras=None, root_hybrid_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.domain_range = root_hybrid_paras["domain_range"]
        self.activations = root_hybrid_paras["activations"]
        if root_hybrid_paras["hidden_size"][1]:
            self.hidden_size = root_hybrid_paras["hidden_size"][0]
        else:
            self.hidden_size = 2 * root_base_paras["sliding"] * root_base_paras["feature_size"] + 1
        self.problem_size, self.epoch = None, None

    def _setting__(self):
        ## New discovery
        self._activation1__ = getattr(my_math, self.activations[0])
        self._activation2__ = getattr(my_math, self.activations[1])

        self.input_size, self.output_size = self.X_train.shape[1], self.y_train.shape[1]
        self.w1_size = self.input_size * self.hidden_size
        self.b1_size = self.hidden_size
        self.w2_size = self.hidden_size * self.output_size
        self.b2_size = self.output_size
        self.problem_size = self.w1_size + self.b1_size + self.w2_size + self.b2_size

    def _forecasting__(self):
        hidden = self._activation1__(add(matmul(self.X_test, self.model["w1"]), self.model["b1"]))
        y_pred = self._activation2__(add(matmul(hidden, self.model["w2"]), self.model["b2"]))
        y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
        y_true_unscaled = self.time_series._inverse_scaling__(self.y_test, scale_type=self.scaling)
        return y_true_unscaled, y_pred_unscaled, self.y_test, y_pred

    def _running__(self):
        self.time_system = time()
        self._processing__()
        self._setting__()
        self.time_total_train = time()
        self._training__()
        self.model = self._get_model__(self.solution)
        self.time_total_train = round(time() - self.time_total_train, 4)
        self.time_epoch = round(self.time_total_train / self.epoch, 4)
        self.time_predict = time()
        y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled = self._forecasting__()
        self.time_predict = round(time() - self.time_predict, 8)
        self.time_system = round(time() - self.time_system, 4)
        self._save_results__(y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled, self.loss_train, self.n_runs)

    ## Helper functions
    def _get_model__(self, individual=None):
        w1 = reshape(individual[:self.w1_size], (self.input_size, self.hidden_size))
        b1 = reshape(individual[self.w1_size:self.w1_size + self.b1_size], (-1, self.hidden_size))
        w2 = reshape(individual[self.w1_size + self.b1_size: self.w1_size + self.b1_size + self.w2_size], (self.hidden_size, self.output_size))
        b2 = reshape(individual[self.w1_size + self.b1_size + self.w2_size:], (-1, self.output_size))
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _objective_function__(self, solution=None):
        md = self._get_model__(solution)
        hidd = self._activation1__(add(matmul(self.X_train, md["w1"]), md["b1"]))
        y_pred = self._activation2__(add(matmul(hidd, md["w2"]), md["b2"]))
        return mean_squared_error(y_pred, self.y_train)