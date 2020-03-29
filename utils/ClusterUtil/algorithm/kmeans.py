#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:06, 30/01/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from scipy.spatial.distance import cdist
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans
from utils.ClusterUtil.root_cluster import RootCluster

class BaseKmeans(RootCluster):
    """
    This is standard version of Kmeans: https://en.wikipedia.org/wiki/K-means_clustering
    """
    def __init__(self, n_clusters=10):
        RootCluster.__init__(self)
        self.n_clusters = n_clusters

    def _init_centers__(self, X_data=None):
        # randomly pick k rows of X as initial centers
        return X_data[np.random.choice(X_data.shape[0], self.n_clusters, replace=False)]

    def __kmeans_assign_labels__(self, X_data=None, centers=None):
        D = cdist(X_data, centers)      # calculate pairwise distances btw data and centers
        return np.argmin(D, axis=1)     # return index of the closest center

    def __kmeans_update_centers__(self, X_data=None, labels=None):
        centers = np.zeros((self.n_clusters, X_data.shape[1]))
        for k in range(self.n_clusters):
            # collect all points assigned to the k-th cluster
            Xk = X_data[labels == k, :]
            # take average
            centers[k, :] = np.mean(Xk, axis=0)
        return centers

    def __has_converged__(self, centers=None, new_centers=None):
        # return True if two sets of centers are the same
        return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

    def _cluster__(self, X_data=None):
        """
        :param X_data:
        :return:
        1st: the number of clusters
        2nd: the numpy array includes the clusters center
        3rd: number of generations made k-means convergence
        4th: the numpy array indicates which cluster that point belong to (based on index of array)
        5th: the concatenate of dataset and 1 column includes 4th
        """
        centers = [self._init_centers__(X_data)]
        labels = []
        it = 0
        while True:
            labels.append(self.__kmeans_assign_labels__(X_data, centers[-1]))
            new_centers = self.__kmeans_update_centers__(X_data, labels[-1])
            if self.__has_converged__(centers[-1], new_centers):
                break
            centers.append(new_centers)
            it += 1

        t1, t2 = np.unique(labels[-1], return_counts=True)
        self.list_clusters = [[cou, centers[-1][uni]] for uni, cou in zip( t1, t2)]
        self.centers = deepcopy(centers[-1])
        self.labels = deepcopy(np.reshape(labels[-1], (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(labels[-1], (-1, 1))), axis=1)
        return [self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label]


class KMeansPlusPlus(BaseKmeans):
    """
    This is the improved version of Kmeans called Kmeans++ :  https://en.wikipedia.org/wiki/K-means%2B%2B
    """
    def __init__(self, n_clusters=10):
        BaseKmeans.__init__(self, n_clusters)

    def _init_centers__(self, X_data=None):
        # randomly pick k rows of X as initial centers
        centers = np.reshape( X_data[np.random.choice(X_data.shape[0], 1, replace=False)], (1, -1))

        for i in range(1, self.n_clusters):
            D = cdist(X_data, centers)          # calculate pairwise distances btw data and centers
            Dmin = np.min(D, axis=1)            # return index of the closest center
            sum = np.sum(Dmin) * np.random.rand()

            for j, di in enumerate(Dmin):
                sum -= di
                if sum <= 0:
                    centers = np.concatenate((centers, deepcopy(np.reshape(X_data[j], (1, -1)))), axis=0)
                    break
        return centers


class KMeansSklearn(RootCluster):
    """
    This is the standard k-means but using scikit-learn library:
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    def __init__(self, n_clusters = 10):
        RootCluster.__init__(self)
        self.n_clusters = n_clusters

    def _cluster__(self, X_data=None, type=None):
        """
        :param type: 'random', 'k-means++'
        :return:
        """
        kmeans = KMeans(n_clusters=self.n_clusters, init=type, random_state=11).fit(X_data)
        labels = kmeans.predict(X_data)
        centers = kmeans.cluster_centers_
        t1, t2 = np.unique(labels, return_counts=True)
        self.list_clusters = [[cou, centers[uni]] for uni, cou in zip(t1, t2)]
        self.centers = deepcopy(centers)
        self.labels = deepcopy(np.reshape(labels, (-1, 1)))
        self.feature_label = np.concatenate((X_data, np.reshape(labels, (-1, 1))), axis=1)

        return [self.n_clusters, self.centers, self.list_clusters, self.labels, self.feature_label]





