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


class Rnn2HL(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		RootRnn.__init__(self, root_base_paras, root_rnn_paras)
		self.filename = "RNN-2HL-" + root_rnn_paras["paras_name"]

	def _training__(self):
		#  The RNN 2-HL architecture
		self.model = Sequential()
		self.model.add(LSTM(units=self.hidden_sizes[0], return_sequences=True, input_shape=(self.X_train.shape[1], 1), activation=self.activations[0]))
		self.model.add(Dropout(self.dropouts[0]))
		self.model.add(LSTM(units=self.hidden_sizes[1], activation=self.activations[1]))
		self.model.add(Dropout(self.dropouts[1]))
		self.model.add(Dense(units=1, activation=self.activations[2]))
		self.model.compile(loss=self.loss, optimizer=self.optimizer)
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.log)
		self.loss_train = ml.history["loss"]


class Lstm1HL(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		RootRnn.__init__(self, root_base_paras, root_rnn_paras)
		self.filename = "LSTM-1HL-" + root_rnn_paras["paras_name"]

	def _training__(self):
		#  The LSTM 1-HL architecture
		self.model = Sequential()
		self.model.add(LSTM(units=self.hidden_sizes[0], input_shape=(None, 1), activation=self.activations[0]))
		self.model.add(Dense(units=1, activation=self.activations[1]))
		self.model.compile(loss=self.loss, optimizer=self.optimizer)
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.log)
		self.loss_train = ml.history["loss"]


class Lstm2HL(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		RootRnn.__init__(self, root_base_paras, root_rnn_paras)
		self.filename = "LSTM-2HL-" + root_rnn_paras["paras_name"]

	def _training__(self):
		#  The LSTM 2-HL architecture
		self.model = Sequential()
		self.model.add(LSTM(units=self.hidden_sizes[0], return_sequences=True, input_shape=(None, 1), activation=self.activations[0]))
		self.model.add(LSTM(units=self.hidden_sizes[1], activation=self.activations[1]))
		self.model.add(Dense(units=1, activation=self.activations[2]))
		self.model.compile(loss=self.loss, optimizer=self.optimizer)
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.log)
		self.loss_train = ml.history["loss"]


class Gru1HL(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		RootRnn.__init__(self, root_base_paras, root_rnn_paras)
		self.filename = "GRU-1HL-" + root_rnn_paras["paras_name"]

	def _training__(self):
		#  The GRU 1-HL architecture
		self.model = Sequential()
		self.model.add(GRU(units=self.hidden_sizes[0], input_shape=(self.X_train.shape[1], 1), activation=self.activations[0]))
		self.model.add(Dropout(self.dropouts[0]))
		self.model.add(Dense(units=1, activation=self.activations[1]))
		self.model.compile(loss=self.loss, optimizer=self.optimizer)
		ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.log)
		self.loss_train = ml.history["loss"]


class Gru2HL(RootRnn):
	def __init__(self, root_base_paras=None, root_rnn_paras=None):
		RootRnn.__init__(self, root_base_paras, root_rnn_paras)
		self.filename = "GRU-2HL-" + root_rnn_paras["paras_name"]

	def _training__(self):
		#  The GRU 2-HL architecture
		self.model = Sequential()
		self.model.add(GRU(units=self.hidden_sizes[0], return_sequences=True, input_shape=(self.X_train.shape[1], 1), activation=self.activations[0]))
		self.model.add(Dropout(self.dropouts[0]))
		self.model.add(GRU(units=self.hidden_sizes[1], activation=self.activations[1]))
		self.model.add(Dropout(self.dropouts[1]))
		self.model.add(Dense(units=1, activation=self.activations[2]))
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
