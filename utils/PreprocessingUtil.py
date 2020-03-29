#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:46, 09/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import reshape, array


class CheckDataset:
    def __init__(self):
        pass

    def _checking_consecutive__(self, df, time_name="timestamp", time_different=300):
        """
        :param df: Type of this must be dataframe
        :param time_name: the column name of date time
        :param time_different by seconds: 300 = 5 minutes
            https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Timedelta.html
        :return:
        """
        consecutive = True
        for i in range(df.shape[0] - 1):
            diff = (df[time_name].iloc[i + 1] - df[time_name].iloc[i]).seconds
            if time_different != diff:
                print("===========Not consecutive at: {}, different: {} ====================".format(i + 3, diff))
                consecutive = False
        return consecutive


class TimeSeries:
    def __init__(self, data=None, train_split=0.8):
        self.data_original = data
        if train_split < 1.0:
            self.train_split = int(train_split * self.data_original.shape[0])
        else:
            self.train_split = train_split
        self.data_new = None
        self.train_mean, self.train_std, self.train_min, self.train_max = None, None, None, None
        self.data_mean, self.data_std, self.data_min, self.data_max = None, None, None, None

    def _scaling__(self, scale_type="std"):
        """
        :param dataset: 2D numpy array
        :param scale_type: std / minmax
        :return:
        """
        self.train_mean, self.train_std = self.data_original[:self.train_split].mean(axis=0), self.data_original[:self.train_split].std(axis=0)
        self.train_min, self.train_max = self.data_original[:self.train_split].min(axis=0), self.data_original[:self.train_split].max(axis=0)

        self.data_mean, self.data_std = self.data_original.mean(axis=0), self.data_original.std(axis=0)
        self.data_min, self.data_max = self.data_original.min(axis=0), self.data_original.max(axis=0)

        if scale_type == "std":
            self.data_new = (self.data_original - self.train_mean) / self.train_std
        elif scale_type == "minmax":
            self.data_new = (self.data_original - self.train_min) / (self.train_max - self.train_min)
        return self.data_new

    def _inverse_scaling__(self, data=None, scale_type="std"):
        """
        :param data:
        :type data:
        :param scale_type:
        :type scale_type:
        :return:
        :rtype:
        """
        if scale_type == "std":
            return self.train_std * data - self.train_mean
        elif scale_type == "minmax":
            return data * (self.train_max - self.train_min) + self.train_min

    def _univariate_data__(self, dataset, history_column=None, start_index=0, end_index=None, pre_type="2D"):
        """
        :param dataset: 2-D numpy array
        :param history_column: python list time in the past you want to use. (1, 2, 5) means (t-1, t-2, t-5) predict time t
        :param start_index: 0- training set, N- valid or testing set
        :param end_index: N-training or valid set, None-testing set
        :param pre_type: 3D for RNN-based, 2D for normal neural network like MLP, FFLN,..
        :return:
        """
        data = []
        labels = []

        history_size = len(history_column)
        if end_index is None:
            end_index = len(dataset) - history_column[-1] - 1  # for time t, such as: t-1, t-4, t-7 and finally t
        else:
            end_index = end_index - history_column[-1] - 1

        for i in range(start_index, end_index):
            indices = i - 1 + array(history_column)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + history_column[-1]])
        if pre_type == "3D":
            return array(data), array(labels)
        return reshape(array(data), (-1, history_size)), array(labels)

    #### 2 methods below we don't use it anymore
    def _univariate_data_old__(self, dataset, start_index=0, end_index=None, history_size=3, target_size=0, pre_type="2D"):
        """
        :param dataset: 2-D numpy array
        :param start_index: 0- training set, N- valid or testing set
        :param end_index: N-training or valid set, None-testing set
        :param history_size: sliding window, 1 mean t-1, 2 mean t-2, 3 mean t-3,...
        :param target_size: 1 mean t, 2 mean t+1, 3 mean t+2
        :param type: 3D for RNN-based, 2D for normal neural network like MLP, FFLN,..
        :return:
        """
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + target_size])
        if pre_type == "3D":
            return array(data), array(labels)
        return reshape(array(data), (-1, history_size)), array(labels)

    def _multivariate_data__(self, dataset, target_label, start_index, end_index, history_size, target_size, step,
                             single_step=False, pre_type="2D"):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target_label[i + target_size])
            else:
                labels.append(target_label[i:i + target_size])

        if pre_type == "3D":
            return array(data), array(labels)
        return reshape(array(data), (-1, history_size)), array(labels)