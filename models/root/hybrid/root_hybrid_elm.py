#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:06, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.metrics import mean_squared_error
from numpy import matmul, reshape, add, dot
from numpy.linalg import pinv
from time import time
from models.root.root_base import RootBase
import utils.MathUtil as my_math


class RootHybridElm(RootBase):
    """
        This is root of all hybrid models which include Extreme Learning Machine and Optimization Algorithms.
    """

    def __init__(self, root_base_paras=None, root_hybrid_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.activation = root_hybrid_paras["activation"]
        self.domain_range = root_hybrid_paras["domain_range"]
        if root_hybrid_paras["hidden_size"][1]:
            self.hidden_size = root_hybrid_paras["hidden_size"][0]
        else:
            self.hidden_size = 2 * root_base_paras["sliding"] * root_base_paras["feature_size"] + 1
        self._activation__ = getattr(my_math, self.activation)

    def _setting__(self):
        self.input_size, self.output_size = self.X_train.shape[1], self.y_train.shape[1]
        self.w1_size = self.input_size * self.hidden_size
        self.b_size = self.hidden_size
        self.w2_size = self.hidden_size * self.output_size
        self.problem_size = self.w1_size + self.b_size

    ## Helper functions
    def _get_model__(self, solution=None):
        w1 = reshape(solution[:self.w1_size], (self.input_size, self.hidden_size))
        b = reshape(solution[self.w1_size:self.w1_size + self.b_size], (-1, self.hidden_size))
        H = self._activation__(add(matmul(self.X_train, w1), b))
        w2 = dot(pinv(H), self.y_train)  # calculate weights between hidden and output layer
        return {"w1": w1, "b": b, "w2": w2}

    def _objective_function__(self, solution=None):
        w1 = reshape(solution[:self.w1_size], (self.input_size, self.hidden_size))
        b = reshape(solution[self.w1_size:self.w1_size + self.b_size], (-1, self.hidden_size))
        H = self._activation__(add(matmul(self.X_train, w1), b))
        H_pinv = pinv(H)  # compute a pseudo-inverse of H
        w2 = dot(H_pinv, self.y_train)  # calculate weights between hidden and output layer
        y_pred = matmul(H, w2)
        return mean_squared_error(y_pred, self.y_train)

    def _forecasting__(self):
        hidd = self._activation__(add(matmul(self.X_test, self.model["w1"]), self.model["b"]))
        y_pred = matmul(hidd, self.model["w2"])
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
        self.time_epoch = None
        self.time_predict = time()
        y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled = self._forecasting__()
        self.time_predict = round(time() - self.time_predict, 8)
        self.time_system = round(time() - self.time_system, 4)
        self._save_results__(y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled, self.loss_train, self.n_runs)
