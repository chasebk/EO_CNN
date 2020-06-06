#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 15:20, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%


from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from models.root.traditional.root_rnn import RootRnn


class Rnn1HL(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		RootRnn.__init__(self, root_base_paras, root_rnn_paras)
		self.filename = "RNN-1HL-" + root_rnn_paras["paras_name"]

	def _training__(self):
		#  The RNN 1-HL architecture
		self.model = Sequential()
		self.model.add(LSTM(units=self.hidden_sizes[0], activation=self.activations[0], input_shape=(self.X_train.shape[1], 1)))
		self.model.add(Dropout(self.dropouts[0]))
		self.model.add(Dense(units=1, activation=self.activations[1]))
		self.model.compile(loss=self.loss, optimizer=self.optimizer)
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.log)
		self.loss_train = ml.history["loss"]


class Cnn1(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None, cnn_paras=None):
		RootRnn.__init__(self, root_base_paras, root_rnn_paras)
		self.filename = "CNN1-" + root_rnn_paras["paras_name"]
		self.filters_size = cnn_paras["filters_size"]
		self.kernel_size = cnn_paras["kernel_size"]
		self.pool_size = cnn_paras["pool_size"]

	def _training__(self):
		#  The CNN 1-HL architecture
		self.model = Sequential()
		self.model.add(Conv1D(filters=self.filters_size, kernel_size=self.kernel_size, activation=self.activations[0], input_shape=(self.X_train.shape[1], 1)))
		self.model.add(MaxPooling1D(pool_size=self.pool_size))
		self.model.add(Flatten())
		self.model.add(Dense(self.hidden_sizes[0], activation=self.activations[1]))
		self.model.add(Dense(1))
		self.model.compile(loss=self.loss, optimizer=self.optimizer)
		# fit mode
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.log)
		self.loss_train = ml.history["loss"]
