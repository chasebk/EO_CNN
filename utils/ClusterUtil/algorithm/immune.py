#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:06, 30/01/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
from utils.ClusterUtil.root_cluster import RootCluster

class ImmuneInspiration(RootCluster):
    """
    This cluster algorithm implement based on from paper:
        Improving recognition and generalization capability of
        back-propagation NN using a self-organized network inspired by immune algorithm (SONIA)
    """
    def __init__(self, stimulation_level=0.25, positive_number=0.15, distance_level=0.5, mutation_id=0, max_cluster=50):
        """
        :param stimulation_level:
        :param positive_number:
        :param distance_level:
        :param mutation_id: 0 - Mean(parents), 1 - Uniform(parents)
        :param max_cluster:
        """
        RootCluster.__init__(self)
        self.stimulation_level = stimulation_level
        self.positive_number = positive_number
        self.distance_level = distance_level
        self.max_cluster = max_cluster
        self.mutation_id = mutation_id
        self.threshold_number = None

    def _cluster__(self, X_data=None):
        """
        :param X_data:
        :return:
        1st: the number of clusters
        2nd: the numpy array includes the clusters center
        3rd: the python array includes list of clusters center and the number of point belong to that cluster
        4th: the numpy array indicates which cluster that point belong to (based on index of array)
        5th: the concatenate of dataset and 1 column includes 4th
        """

        ### Phrase 1: Cluster data - First step training of two-step training
        # 2. Init hidden unit 1st
        y_pred = np.zeros(len(X_data))
        hu1 = [0, deepcopy(X_data[np.random.randint(0, len(X_data))])]  # hidden unit 1 (t1, wH)
        list_clusters = [deepcopy(hu1)]                         # list hidden units
        centers = deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])  # 2-d array
        m = 0
        while m < len(X_data):
            D = cdist(np.reshape(X_data[m], (1, -1)), centers)  # calculate pairwise distances btw mth point and centers
            c = np.argmin(D, axis=1)[0]        # return index of the closest center
            distmc = np.min(D, axis=1)      # distmc: minimum distance

            if distmc < self.stimulation_level:
                y_pred[m] = c               # Which cluster c, does example m belong to?
                list_clusters[c][0] += 1  # update hidden unit cth

                centers[c] = centers[c] + self.positive_number * distmc * (X_data[m] - list_clusters[c][1])
                list_clusters[c][1] = list_clusters[c][1] + self.positive_number * distmc * (X_data[m] - list_clusters[c][1])
                # Next example
                m += 1
                # if m % 1000 == 0:
                #     print "distmc = {0}".format(distmc)
                #     print "m = {0}".format(m)
            else:
                # print "Failed !!!. distmc = {0}".format(distmc)
                list_clusters.append([0, deepcopy(X_data[m])])
                # print "Hidden unit: {0} created.".format(len(list_clusters))
                centers = np.append(centers, [deepcopy(X_data[m])], axis=0)
                for hu in list_clusters:
                    hu[0] = 0
                # then go to step 1
                m = 0
                if len(list_clusters) > self.max_cluster:
                    print("==== Over the number of clusters allowable =====")
                    break
        self.n_clusters = len(list_clusters)
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)
        self.labels = deepcopy(np.reshape(y_pred, (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(y_pred, (-1, 1))), axis=1)

        return [self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label]


class SomInspiration(ImmuneInspiration):
    """
    This version based on the Self-organizing Map network with more parameters:
        https://en.wikipedia.org/wiki/Self-organizing_map
    """
    def __init__(self, stimulation_level=0.25, positive_number=0.15, distance_level=0.5, mutation_id=0, max_cluster=50,
                 neighbourhood_density=0.2, gauss_width=1.0):
        """
        :param neighbourhood_density:
        :param gauss_width:
        """
        ImmuneInspiration.__init__(self, stimulation_level, positive_number, distance_level, mutation_id, max_cluster)
        self.neighbourhood_density = neighbourhood_density
        self.gauss_width = gauss_width

    def _cluster__(self, X_data=None):
        """
        :return:
        1st: the number of clusters
        2nd: the numpy array includes the clusters center
        3rd: the python array includes list of clusters center and the number of point belong to that cluster
        4th: the numpy array indicates which cluster that point belong to (based on index of array)
        5th: the concatenate of dataset and 1 column includes 4th
        """

        ### Phrase 1: Clustering data
        # 2. Init hidden unit 1st
        y_pred = np.zeros(len(X_data))
        hu1 = [0, deepcopy(X_data[np.random.randint(0, len(X_data))])]  # hidden unit 1 (t1, wH)
        list_clusters = [deepcopy(hu1)]                         # list hidden units
        centers = deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])  # 2-d array
        m = 0
        while m < len(X_data):
            D = cdist(np.reshape(X_data[m],(1, -1)), centers)   # calculate pairwise distances btw mth point and centers
            c = np.argmin(D, axis=1)[0]                         # return index of the closest center
            distmc = np.min(D, axis=1)                          # distmc: Minimum distance

            if distmc < self.stimulation_level:
                y_pred[m] = c                                   # Which cluster c, does example m belong to?
                list_clusters[c][0] += 1                        # update hidden unit cth

                # Find Neighbourhood
                nei = cdist(np.reshape(centers[c], (1, -1)), centers)
                nei_id_sorted = np.argsort(nei)
                nei_dist_sorted = np.sort(nei)

                # Update BMU (Best matching unit and it's neighbourhood)
                if len(list_clusters) < 5:
                    for i in range(0, len(list_clusters)):
                        if i == 0:
                            delta = distmc * (X_data[m] - list_clusters[c][1])
                            c = c
                        else:
                            c_temp, distjc = nei_id_sorted[0][i], nei_dist_sorted[0][i]
                            hic = np.exp(-distjc * distjc / self.gauss_width)
                            delta = (self.positive_number * hic) * (X_data[m] - list_clusters[c_temp][1])
                            c = c_temp
                        list_clusters[c][1] += delta
                        centers[c] += delta
                else:
                    neighbourhood_node = int(1 + np.ceil(self.neighbourhood_density * (len(list_clusters) - 1)))
                    for i in range(0, neighbourhood_node):
                        if i == 0:
                            delta = distmc * (X_data[m] - list_clusters[c][1])
                            c = c
                        else:
                            c_temp, distjc = nei_id_sorted[0][i], nei_dist_sorted[0][i]
                            hic = np.exp(-distjc * distjc / self.gauss_width)
                            delta = (self.positive_number * hic) * (X_data[m] - list_clusters[c_temp][1])
                            c = c_temp
                        list_clusters[c][1] += delta
                        centers[c] += delta
                # Next example
                m += 1
                # if m % 1000 == 0:
                #     print "distmc = {0}".format(distmc)
                #     print "m = {0}".format(m)
            else:
                # print "Failed !!!. distmc = {0}".format(distmc)
                list_clusters.append([0, deepcopy(X_data[m])])
                # print "Hidden unit: {0} created.".format(len(list_clusters))
                centers = np.append(centers, [deepcopy(X_data[m])], axis=0)
                for hu in list_clusters:
                    hu[0] = 0
                # then go to step 1
                m = 0
                if len(list_clusters) > self.max_cluster:
                    print("==== Over the number of clusters allowable =====")
                    break

        self.n_clusters = len(list_clusters)
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)
        self.labels = deepcopy(np.reshape(y_pred, (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(y_pred, (-1, 1))), axis=1)

        return [self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label]