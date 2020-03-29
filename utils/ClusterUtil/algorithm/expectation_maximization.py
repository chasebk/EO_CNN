#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:36, 30/01/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from utils.ClusterUtil.root_cluster import RootCluster

class GaussianMixtureSklearn(RootCluster):
    """
    Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
        http://scikit-learn.org/stable/modules/mixture.html#mixture
    """
    def __init__(self, n_clusters=10, covariance_type='full'):
        """
        :param n_clusters:
        :param covariance_type: spherical, diag, tied or full
        """
        RootCluster.__init__(self)
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type

    def _cluster__(self, X_data=None):
        gauss_model = GaussianMixture(n_components=self.n_clusters, covariance_type=self.covariance_type).fit(X_data)
        labels = gauss_model.predict(X_data)
        centers = gauss_model.means_
        t1, t2 = np.unique(labels, return_counts=True)
        self.list_clusters = [[cou, centers[uni]] for uni, cou in zip(t1, t2)]
        self.centers = deepcopy(centers)
        self.labels = deepcopy(np.reshape(labels, (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(labels, (-1, 1))), axis=1)

        return [self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label]

