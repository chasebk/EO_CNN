#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:28, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from time import time
from models.root.root_base import RootBase
import utils.MathUtil as my_math
from utils.MeasureUtil import MeasureTimeSeries
from utils.ClusterUtil.algorithm.immune import ImmuneInspiration
from utils.ClusterUtil.algorithm.kmeans import BaseKmeans, KMeansPlusPlus
from utils.ClusterUtil.algorithm.mean_shift import MeanShiftSklearn
from utils.ClusterUtil.algorithm.expectation_maximization import GaussianMixtureSklearn
from utils.ClusterUtil.algorithm.dbscan import DbscanSklearn
from utils.IOUtil import _save_results_to_csv__, _save_prediction_to_csv__, _save_loss_train_to_csv__
from utils.GraphUtil import _draw_predict_with_error__




class RootSsnn(RootBase):
    def __init__(self, root_base_paras=None, clustering_paras=None, root_ssnn_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.clustering_paras = clustering_paras
        self.clustering_type = root_ssnn_paras["clustering_type"]
        self.epoch = root_ssnn_paras["epoch"]
        self.batch_size = root_ssnn_paras["batch_size"]
        self.learning_rate = root_ssnn_paras["learning_rate"]
        self.activations = root_ssnn_paras["activations"]
        self._activation1__ = getattr(my_math, self.activations[0])
        self.optimizer = root_ssnn_paras["optimizer"]
        self.loss = root_ssnn_paras["loss"]
        self.n_clusters, self.clustering, self.cluster_score, self.time_cluster = None, None, None, None

    def _clustering__(self):
        """
            0. immune + mutation
            1. immune
            2. mean shift
            3. dbscan
            4. kmeans
            5. kmeans++
            6. exp_max (Expectation–Maximization)

            7. immune + kmeans++
            8. immune + exp_max
        """
        if self.clustering_type == "immune_full":
            self.stimulation_level = self.clustering_paras["stimulation_level"]
            self.positive_number = self.clustering_paras["positive_number"]
            self.distance_level = self.clustering_paras["distance_level"]
            self.max_cluster = self.clustering_paras["max_cluster"]
            self.mutation_id = self.clustering_paras["mutation_id"]
            self.clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                                distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = \
                self.clustering._immune_mutation__(self.X_train, self.list_clusters, self.distance_level, self.mutation_id)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        elif self.clustering_type == "immune":
            self.stimulation_level = self.clustering_paras["stimulation_level"]
            self.positive_number = self.clustering_paras["positive_number"]
            self.distance_level = self.clustering_paras["distance_level"]
            self.max_cluster = self.clustering_paras["max_cluster"]
            self.mutation_id = self.clustering_paras["mutation_id"]
            self.clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                                distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        elif self.clustering_type == "meanshift":
            self.bandwidth = self.clustering_paras["bandwidth"]
            self.clustering = MeanShiftSklearn(self.bandwidth)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        elif self.clustering_type == "dbscan":
            self.eps = self.clustering_paras["eps"]
            self.min_samples = self.clustering_paras["min_samples"]
            self.clustering = DbscanSklearn(eps=self.eps, min_samples=self.min_samples)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        elif self.clustering_type == "kmeans":
            self.n_clusters = self.clustering_paras["n_clusters"]
            self.clustering = BaseKmeans(self.n_clusters)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        elif self.clustering_type == "kmeans++":
            self.n_clusters = self.clustering_paras["n_clusters"]
            self.clustering = KMeansPlusPlus(self.n_clusters)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        elif self.clustering_type == "expmax":  # Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
            self.n_clusters = self.clustering_paras["n_clusters"]
            self.clustering = GaussianMixtureSklearn(self.n_clusters)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)


        elif self.clustering_type == "immune_kmeans++":  # Immune + Kmeans++ (No need Kmeans)
            self.stimulation_level = self.clustering_paras["stimulation_level"]
            self.positive_number = self.clustering_paras["positive_number"]
            self.distance_level = self.clustering_paras["distance_level"]
            self.max_cluster = self.clustering_paras["max_cluster"]
            self.mutation_id = self.clustering_paras["mutation_id"]
            clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                           distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
            t0, t1, t2, t3, t4 = clustering._cluster__(X_data=self.X_train)
            self.clustering = KMeansPlusPlus(t0)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        elif self.clustering_type == "immune_expmax":  # Immune + Expectation–Maximization
            self.stimulation_level = self.clustering_paras["stimulation_level"]
            self.positive_number = self.clustering_paras["positive_number"]
            self.distance_level = self.clustering_paras["distance_level"]
            self.max_cluster = self.clustering_paras["max_cluster"]
            self.mutation_id = self.clustering_paras["mutation_id"]
            clustering = ImmuneInspiration(stimulation_level=self.stimulation_level, positive_number=self.positive_number,
                                           distance_level=self.distance_level, mutation_id=self.mutation_id, max_cluster=self.max_cluster)
            t0, t1, t2, t3, t4 = clustering._cluster__(X_data=self.X_train)
            self.clustering = GaussianMixtureSklearn(t0)
            self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label = self.clustering._cluster__(X_data=self.X_train)
            self.cluster_score = self.clustering._evaluation__(self.X_train, self.clustering.labels, 0)

        else:
            print("==================There is no clustering method=====================")
            exit(0)

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
        self.time_epoch = time()
        self._training__()
        self.time_epoch = round((time() - self.time_epoch) / self.epoch, 4)
        self.time_total_train = round(time() - self.time_total_train, 4)
        self.time_predict = time()
        y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled = self._forecasting__()
        self.time_predict = round(time() - self.time_predict, 8)
        self.time_system = round(time() - self.time_system, 4)
        self._save_results__(y_true_unscaled, y_pred_unscaled, y_true_scaled, y_pred_scaled, self.loss_train, self.n_runs)
