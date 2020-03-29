from utils.ClusterUtil.algorithm.immune import ImmuneInspiration, SomInspiration
from utils.PreprocessingUtil import TimeSeries
from sklearn import preprocessing
import pandas as pd

# parameters
data_index = [5]
data_idx = (0.8, 0, 0.2)
sliding = 2
method_statistic = 0            # 0: sliding window, 1: mean, 2: min-mean-max, 3: min-median-max
scaler = preprocessing.MinMaxScaler()
output_index = None
df = pd.read_csv('../../../data/formatted/google_5m.csv', usecols=[1], header=0, index_col=False)
timeseries = TimeSeries(df.values, data_idx, sliding, output_index, method_statistic, scaler)
X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = timeseries._preprocessing_2d__()
print("Processing data done!!!")

# clustering = ImmuneInspiration(stimulation_level=0.25, positive_number=0.15, distance_level=0.5, mutation_id=0, max_cluster=100)

clustering = SomInspiration(stimulation_level=0.25, positive_number=0.15, distance_level=0.5, mutation_id=0,
                             max_cluster=50, neighbourhood_density=0.05, gauss_width=5.0)

n_clusters, centers, list_clusters, labels, feature_label = clustering._cluster__(X_data=X_train)
s1, s2, s3 = clustering._evaluation__(X_train, labels ,0)
print("s1 = {}, s2 = {}, s3 = {}".format(s1, s2, s3))
