#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 9:36, 30/01/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from sklearn.cluster import MeanShift
from utils.ClusterUtil.root_cluster import RootCluster

class MeanShiftSklearn(RootCluster):
    """
    This is the standard mean_shift but using scikit-learn library:
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    """
    def __init__(self, bandwidth = 0.25):
        RootCluster.__init__(self)
        self.bandwidth = bandwidth

    def _cluster__(self, X_data=None, type=None):
        """
        :param X_data:
        :param type: True (clustering all data points), False (noise with label = -1)
        :return:
        """
        mean_shift = MeanShift(bandwidth=self.bandwidth, cluster_all=type).fit(X_data)
        labels = mean_shift.predict(X_data)
        centers = mean_shift.cluster_centers_
        t1, t2 = np.unique(labels, return_counts=True)
        self.n_clusters = len(centers)
        self.list_clusters = [[cou, centers[uni]] for uni, cou in zip(t1, t2)]
        self.centers = deepcopy(centers)
        self.labels = deepcopy(np.reshape(labels, (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(labels, (-1, 1))), axis=1)

        return [self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label]
