#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:52, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import reshape, add, matmul
from time import time
from models.root.root_base import RootBase
from utils.MeasureUtil import MeasureTimeSeries
from utils.ClusterUtil.algorithm.immune import ImmuneInspiration
from utils.IOUtil import _save_results_to_csv__, _save_prediction_to_csv__, _save_loss_train_to_csv__
from utils.GraphUtil import _draw_predict_with_error__
import utils.MathUtil as my_math


class RootHybridSsnnBase(RootBase):
    """
        This is root of all hybrid models which include Self-Structure Neural Network and Optimization Algorithms.
    """

    def __init__(self, root_base_paras=None, root_hybrid_ssnn_base_paras=None):
        """
        :param root_base_paras:
        :param root_hybrid_ssnn_base_paras:
        """
        RootBase.__init__(self, root_base_paras)
        self.activations = root_hybrid_ssnn_base_paras["activations"]
        self.domain_range = root_hybrid_ssnn_base_paras["domain_range"]
        self.n_clusters, self.clustering, self.cluster_score, self.time_cluster, self.epoch = None, None, None, None, None

        self._activation1__ = getattr(my_math, self.activations[0])
        self._activation2__ = getattr(my_math, self.activations[1])

    def _clustering__(self):
        pass

    def _setting__(self):
        self.S_train = self.clustering._transforming__(self._activation1__, self.X_train)
        self.S_test = self.clustering._transforming__(self._activation1__, self.X_test)
        self.input_size, self.output_size = self.S_train.shape[1], self.y_train.shape[1]
        self.w_size = self.input_size * self.output_size
        self.b_size = self.output_size
        self.problem_size = self.w_size + self.b_size

    def _forecasting__(self):
        y_pred = self._activation2__(add(matmul(self.S_test, self.model["w"]), self.model["b"]))
        y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)
        y_true_unscaled = self.time_series._inverse_scaling__(self.y_test, scale_type=self.scaling)
        return y_true_unscaled, y_pred_unscaled, self.y_test, y_pred

    def _save_results__(self, y_true=None, y_pred=None, y_true_scaled=None, y_pred_scaled=None, loss_train=None, n_runs=1):
        measure_scaled = MeasureTimeSeries(y_true_scaled, y_pred_scaled, "raw_values", number_rounding=4)
        measure_scaled._fit__()
        measure_unscaled = MeasureTimeSeries(y_true, y_pred, "raw_values", number_rounding=4)
        measure_unscaled._fit__()

        item = {'model_name': self.filename, 'n_clusters': self.n_clusters, 'time_cluster': self.time_cluster, 'time_epoch': self.time_epoch,
                'total_time_train': self.time_total_train, 'time_predict': self.time_predict, 'time_system': self.time_system,
                'silhouette': self.cluster_score[0], 'calinski': self.cluster_score[1], 'davies': self.cluster_score[2],
                'scaled_EV': measure_scaled.score_ev, 'scaled_MSLE': measure_scaled.score_msle, 'scaled_R2': measure_scaled.score_r2,
                'scaled_MAE': measure_scaled.score_mae, 'scaled_MSE': measure_scaled.score_mse, 'scaled_RMSE': measure_scaled.score_rmse,
                'scaled_MAPE': measure_scaled.score_mape, 'scaled_SMAPE': measure_scaled.score_smape,
                'unscaled_EV': measure_unscaled.score_ev, 'unscaled_MSLE': measure_unscaled.score_msle, 'unscaled_R2': measure_unscaled.score_r2,
                'unscaled_MAE': measure_unscaled.score_mae, 'unscaled_MSE': measure_unscaled.score_mse, 'unscaled_RMSE': measure_unscaled.score_rmse,
                'unscaled_MAPE': measure_unscaled.score_mape, 'unscaled_SMAPE': measure_unscaled.score_smape}
        if n_runs == 1:
            _save_prediction_to_csv__(y_true, y_pred, y_true_scaled, y_pred_scaled, self.filename, self.path_save_result)
            _save_loss_train_to_csv__(loss_train, self.filename, self.path_save_result + "Error-")
            if self.draw:
                _draw_predict_with_error__([y_true, y_pred], [measure_unscaled.score_rmse, measure_unscaled.score_mae], self.filename, self.path_save_result)
            if self.log:
                print('Predict DONE - RMSE: %f, MAE: %f' % (measure_unscaled.score_rmse, measure_unscaled.score_mae))
        _save_results_to_csv__(item, self.log_filename, self.path_save_result)

    def _running__(self):
        self.time_system = time()
        self._processing__()
        self.time_total_train = time()
        self.time_cluster = time()
        self._clustering__()
        self.time_cluster = round(time() - self.time_cluster, 4)
        self._setting__()
        self.time_epoch = time()
        self._training__()
        self.model = self._get_model__(self.solution)
        self.time_epoch = round((time() - self.time_epoch) / self.epoch, 4)
        self.time_total_train = round(time() - self.time_total_train, 4)
        self.time_predict = time()
        y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled = self._forecasting__()
        self.time_predict = round(time() - self.time_predict, 8)
        self.time_system = round(time() - self.time_system, 4)
        self._save_results__(y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled, self.loss_train, self.n_runs)

    ## Helper functions
    def _get_model__(self, solution=None):
        w = reshape(solution[:self.w_size], (self.input_size, self.output_size))
        b = reshape(solution[self.w_size:], (-1, self.output_size))
        return {"w": w, "b": b}

    def _objective_function__(self, solution=None):
        w = reshape(solution[:self.w_size], (self.input_size, self.output_size))
        b = reshape(solution[self.w_size:], (-1, self.output_size))
        y_pred = self._activation2__(add(matmul(self.S_train, w), b))
        return mean_squared_error(y_pred, self.y_train)


class RootHybridSonia(RootHybridSsnnBase):
    def __init__(self, root_base_paras=None, root_hybrid_ssnn_base_paras=None, sonia_paras=None):
        RootHybridSsnnBase.__init__(self, root_base_paras, root_hybrid_ssnn_base_paras)
        self.clustering_type = sonia_paras["clustering_type"]
        self.stimulation_level = sonia_paras["stimulation_level"]
        self.positive_number = sonia_paras["positive_number"]
        self.distance_level = sonia_paras["distance_level"]
        self.max_cluster = sonia_paras["max_cluster"]
        self.mutation_id = sonia_paras["mutation_id"]

    def _clustering__(self):
        """
            0. immune + mutation
            1. immune
        """
        if self.clustering_type == "immune_full":
            self.clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                                distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = \
                self.clustering._immune_mutation__(self.X_train, self.list_clusters, self.distance_level, self.mutation_id)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        elif self.clustering_type == "immune":
            self.clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                                distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        else:
            print("==================There is no clustering method=====================")
            exit(0)


