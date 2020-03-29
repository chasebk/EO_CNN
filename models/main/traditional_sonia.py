#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:42, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.traditional.root_ssnn import RootSsnn
from keras.models import Sequential
from keras.layers import Dense


class Sonia(RootSsnn):
    """
        Traditional Self-Organized Network Inspired by Immune Algorithm (SONIA)
            (Self-Organizing Neural Network)
    """

    def __init__(self, root_base_paras=None, clustering_paras=None, root_ssnn_paras=None):
        RootSsnn.__init__(self, root_base_paras, clustering_paras, root_ssnn_paras)
        self.filename = "SONIA-" + root_ssnn_paras["paras_name"]

    def _training__(self):
        self.S_train = self.clustering._transforming__(self._activation1__, self.X_train)
        self.S_test = self.clustering._transforming__(self._activation1__, self.X_test)
        self.input_size, self.output_size = self.S_train.shape[1], self.y_train.shape[1]
        self.model = Sequential()
        self.model.add(Dense(units=self.output_size, input_dim=self.input_size, activation=self.activations[1]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.S_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.log)
        self.loss_train = ml.history["loss"]

    def _forecasting__(self):
        y_pred = self.model.predict(self.S_test)
        y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
        y_true_unscaled = self.time_series._inverse_scaling__(self.y_test, scale_type=self.scaling)
        return y_true_unscaled, y_pred_unscaled, self.y_test, y_pred


