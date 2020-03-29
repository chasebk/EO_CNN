#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 9:06, 30/01/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from sklearn.cluster import DBSCAN
from utils.ClusterUtil.root_cluster import RootCluster

class DbscanSklearn(RootCluster):
    """
    DBSCAN
        https://en.wikipedia.org/wiki/DBSCAN
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    """
    def __init__(self, eps=0.5, min_samples=5):
        """
        :param eps: density-based consideration
        :param min_samples: the min number to construct a cluster
        """
        RootCluster.__init__(self)
        self.eps = eps
        self.min_samples = min_samples

    def _cluster__(self, X_data=None):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X_data)
        labels = dbscan.labels_

        t1_uni_label, t2_sum_label = np.unique(labels, return_counts=True)
        centers = []
        for label in t1_uni_label:
            temp = X_data[ np.where(labels == label) ]
            center = np.mean(temp, axis=0)
            centers.append(center)
        if t1_uni_label[0] == -1:
            centers = np.array(centers)[1:]
            t2_sum_label = t2_sum_label[1:]
        self.list_clusters = [[t2_sum_label[i], centers[i]] for i in range(len(centers))]
        self.n_clusters = len(centers)
        self.centers = deepcopy(centers)
        self.labels = deepcopy(np.reshape(labels, (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(labels, (-1, 1))), axis=1)

        return [self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label]



