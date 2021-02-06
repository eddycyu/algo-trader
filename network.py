"""
Artificial Neural Networks Library

@author: eyu
"""

import logging

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D, TimeDistributed, RepeatVector
from keras.layers import Dropout, Flatten
from keras import backend

# create logger
logger = logging.getLogger("algo-trader")


# Simple feed forward baseline
def feed_forward():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    return model


# 1D Convolutional baseline (to experiment with)
def conv1D_net():
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape=(1, 8)))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    return model


# Simple GRU with dropout
def gru_simple():
    model = Sequential()
    model.add(GRU(32, input_shape=(1, 7), dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae', metrics=['mean_absolute_error'])
    return model


# Advanced model combining 1D convs for downsampling then GRUs for time-dependent information encoding
def conv1D_gru_net():
    model = Sequential()
    model.add(Conv1D(32, 1, activation='relu', input_shape=(1, 7)))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(32, 1, activation='relu'))
    model.add(GRU(32, dropout=0.1, recurrent_dropout=0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# Custom LSTM 1
# LSTM is an artificial recurrent neural network (RNN) with feedback connections.
def lstm1(hidden_nodes, steps_in=5, steps_out=1, features=1):
    """
    A custom LSTM model.

    LSTM with tanh
    Dense with None

    :param hidden_nodes: number of hidden nodes
    :param steps_in: number of (look back) time steps for each sample input
    :param steps_out: number of (look front) time steps for each sample output
    :param features: number of features for each sample input (e.g. 1 for univariate or 2+ for multivariate time series)
    :return: simple LSTM model
    """
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(steps_in, features)))  # default activation: tanh
    model.add(Dense(steps_out))  # default activation: None
    model.compile(optimizer='adam', loss='mse')
    return model


# Custom LSTM 2
def lstm2(hidden_nodes, steps_in=5, steps_out=1, features=1):
    """
    A custom LSTM model.

    :param hidden_nodes: number of hidden nodes
    :param steps_in: number of (look back) time steps for each sample input
    :param steps_out: number of (look front) time steps for each sample output
    :param features: number of features for each sample input (e.g. 1 for univariate or 2+ for multivariate time series)
    :return: simple LSTM model
    """
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(steps_in, features), return_sequences=True))  # default activation: tanh
    model.add(LSTM(hidden_nodes))  # default activation: tanh
    model.add(Dense(steps_out))  # default activation: None
    model.compile(optimizer='adam', loss='mse')
    return model


# Custom LSTM 3
def lstm3(hidden_nodes, steps_in=20, steps_out=5, features=1):
    """
    A custom LSTM model.

    :param hidden_nodes: number of hidden nodes
    :param steps_in: number of (look back) time steps for each sample input
    :param steps_out: number of (look front) time steps for each sample output
    :param features: number of features for each sample input (e.g. 1 for univariate or 2+ for multivariate time series)
    :return: simple LSTM model
    """
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(steps_in, features)))  # default activation: tanh
    model.add(RepeatVector(steps_out))
    model.add(LSTM(hidden_nodes, return_sequences=True))  # default activation: tanh
    model.add(TimeDistributed(Dense(1)))  # default activation: None
    model.compile(optimizer='adam', loss='mse')
    return model


# Simple LSTM
def lstm_simple(steps=14, features=1):
    """
    A simple LSTM model.

    :param steps: number of (look back) time steps for each sample input
    :param features: number of features for each sample input (e.g. 1 for univariate or 2+ for multivariate time series)
    :return: simple LSTM model
    """
    model = Sequential()
    model.add(LSTM(32, activation="tanh", input_shape=(steps, features)))
    # model.add(LSTM(32, activation="relu", input_shape=(steps, features)))
    model.add(Dense(1, activation='linear'))
    # model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    # model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer='adam', loss='mse',  metrics=[rmse])
    return model


# Stacked LSTM
def lstm_stacked(steps=14, features=1):
    model = Sequential()
    # model.add(LSTM(50, activation="tanh", input_shape=(steps, features), return_sequences=True, dropout=0.1, recurrent_dropout=0.2))
    # model.add(LSTM(50, activation="tanh", input_shape=(steps, features), return_sequences=True))
    model.add(LSTM(50, activation="relu", input_shape=(steps, features), return_sequences=True))
    # model.add(LSTM(50, activation="tanh", dropout=0.1, recurrent_dropout=0.2))
    # model.add(LSTM(50, activation="tanh"))
    model.add(LSTM(50, activation="relu"))
    # model.add(Dense(1, activation='linear'))
    # model.add(Dense(1, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    return model


# Stateful LSTM
def lstm_stateful():
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, 1, 7), stateful=True, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model
