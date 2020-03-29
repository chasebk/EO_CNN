from utils.ClusterUtil.algorithm.kmeans import BaseKmeans, KMeansPlusPlus, KMeansSklearn
from utils.PreprocessingUtil import TimeSeries
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
colors = ['y', 'k', 'w', 'b', 'g', 'r', 'c', 'm']
markers = ["+", "x", "D", "|", "1", "4", "s", "p", "*"]
def cluster_display(X, label):
    K = np.amax(label) + 1

    for i in range(K):
        X0 = X[label == i, :]
        plt.plot(X0[:, 0], X0[:, 1], colors[i] + ".", markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()

# parameters
data_index = [5]
data_idx = (0.8, 0, 0.2)
sliding = 3
method_statistic = 0            # 0: sliding window, 1: mean, 2: min-mean-max, 3: min-median-max
scaler = preprocessing.MinMaxScaler()
output_index = None
df = pd.read_csv('../../../data/formatted/google_5m.csv', usecols=[1], header=0, index_col=False)

timeseries = TimeSeries(df.values, data_idx, sliding, output_index, method_statistic, scaler)
X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = timeseries._preprocessing_2d__()
print("Processing data done!!!")

#clustering = BaseKmeans(8)
clustering = KMeansPlusPlus(8)
#clustering = KMeansSklearn(8)

n_clusters, centers, list_clusters, labels, feature_label = clustering._cluster__(X_data=X_train)
print('Centers found by our algorithm:')
print(n_clusters)
print(centers)
s1, s2, s3 = clustering._evaluation__(X_train, labels ,0)
print("s1 = {}, s2 = {}, s3 = {}".format(s1, s2, s3))

cluster_display(X_train, np.reshape(labels, (len(labels))))



