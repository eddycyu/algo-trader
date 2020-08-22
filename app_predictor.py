"""
Price Predictor

@author: eyu
"""

import os
import logging
import click

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import date

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten

from data_reader_stooq import StooqDataReader
from data_reader_yahoo import YahooDataReader
import talib as talib
import palib as palib
from paplot import PAPlot
import constants as c

# set logging level
logging.basicConfig(level=logging.INFO)

# format display width
pd.set_option('display.width', 1000)

# check if the log directory exists; if not, make it
if not os.path.exists(c.LOG_DIR):
    os.makedirs(c.LOG_DIR)

# create logger
logger = logging.getLogger("algo-trader")

# create handlers
file_handler = logging.FileHandler(os.path.join(c.LOG_DIR, "algo-trader.log"))
console_handler = logging.StreamHandler()

# create log formatter
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_formatter)
console_handler.setFormatter(log_formatter)

# add handler to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Simple feed forward baseline
def feed_forward():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model


# 1D Convolutional baseline (to experiment with)
def conv1D_net():
    model = Sequential()
    model.add(Conv1D(32, 2, activation='relu', input_shape=(1, 8)))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model


# Simple GRU with dropout
def gru_simple():
    model = Sequential()
    model.add(GRU(32, input_shape=(1, 7), dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam', metrics=['mean_absolute_error'])
    return model


# Advanced model combining 1D convs for downsampling then GRUs for time-dependent information encoding (to experimnent with)
def conv1D_gru_net():
    model = Sequential()
    model.add(Conv1D(32, 1, activation='relu', input_shape=(1, 7)))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(32, 1, activation='relu'))
    model.add(GRU(32, dropout=0.1, recurrent_dropout=0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model


# Simple LSTM
def lstm_simple():
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 7)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model


# Stacked LSTM
def lstm_stacked():
    model = Sequential()
    model.add(LSTM(16, input_shape=(1, 7), dropout=0.1, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(16, dropout=0.1, recurrent_dropout=0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model


# Stateful LSTM
def lstm_stateful():
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(1, 1, 7), stateful=True, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam')
    return model


def predict_equities(start_date, end_date):
    # symbols = ["AMZN", "GOOG", "MSFT", "FB", "AAPL", "NFLX", "TSLA"]
    symbols = ["AAPL"]
    data_reader = YahooDataReader()
    plotter = PAPlot(c.CHART_PA_DIR)
    for symbol in symbols:
        symbol_name = symbol

        # load data
        df = data_reader.load(symbol, start_date, end_date, symbol_name)

        # use 'Adj Close' as our close price
        df = talib.copy_column(df, "Adj Close", "u_close")

        # split data to create training and testing data sets
        # train_df, test_df = palib.split_train_test_df_by_ratio(data, 0.2)
        train_df, test_df = palib.split_train_test_df_by_fixed(df, test_size=708)

        # scaler to normalize feature range
        scaler = MinMaxScaler(feature_range=(0, 1))

        # normalize (windowed approach) training data
        #train_df = palib.normalize_window(train_df, "u_close", "u_close_norm", scaler, 250)
        train_df = palib.normalize(train_df, "u_close", "u_close_norm", scaler)

        # normalize (non-windowed approach) testing data
        test_df = palib.normalize(test_df, "u_close", "u_close_norm", scaler)

        # use exponential smoothing to remove noise from the normalized training data
        train_df = palib.exponential_smooth(train_df, "u_close_norm", "u_close_norm_smooth", gamma=0.1)

        plotter.plot_multiple(train_df.tail(252), "u_close", "u_close_norm", "u_close_norm_smooth",
                              symbol_name=symbol_name, ylabel="Price $")

        # create datasets
        look_back = 7
        foresight = 3
        x_train, y_train = palib.create_train_test_dataset(train_df, "u_close_norm_smooth", look_back, foresight)
        x_test, y_test = palib.create_train_test_dataset(test_df, "u_close_norm", look_back, foresight)

        # reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        # iterate through each model to train and test
        epochs = 50
        model_list = [feed_forward, gru_simple, conv1D_gru_net, lstm_simple, lstm_stacked]
        for model_function in model_list:
            # train network
            network, model, model_name = palib.train_network(
                model_function, x_train, y_train, x_test, y_test,
                epochs, symbol_name, c.MODEL_DIR)

            # plot network loss
            plotter.plot_losses(network, model_name, symbol_name, look_back, foresight)

            # use model to predict
            y_pred = model.predict(x_test)

            # plot prediction (y_pred) vs actual (y_test)
            plotter.plot_predictions(y_pred, y_test, scaler, model_name, symbol_name, look_back, foresight)


def predict_equities_save(start_date, end_date):
    # symbols = ["AMZN", "GOOG", "MSFT", "FB", "AAPL", "NFLX", "TSLA"]
    symbols = ["AAPL"]
    data_reader = YahooDataReader()
    plotter = PAPlot(c.CHART_PA_DIR)
    for symbol in symbols:
        symbol_name = symbol
        # load data
        df = data_reader.load(symbol, start_date, end_date, symbol_name)

        # use 'Adj Close' as our close price
        df = talib.copy_column(df, "Adj Close", "u_close")

        # split data to create training and testing data sets
        # train_df, test_df = palib.split_train_test_df_by_ratio(data, 0.2)
        train_df, test_df = palib.split_train_test_df_by_fixed(df, test_size=708)

        # normalize (windowed approach) training data
        train_df = palib.normalize_window(train_df, "u_close", "u_close_norm", scaler, 250)

        # scaler to normalize feature range
        scaler = MinMaxScaler(feature_range=(0, 1))

        # normalize (non-windowed approach) testing data
        test_df = palib.normalize(test_df, "u_close", "u_close_norm", scaler)

        # use exponential smoothing to remove noise from the normalized training data
        train_df = palib.exponential_smooth(train_df, "u_close_norm", "u_close_norm_smooth", gamma=0.1)

        plotter.plot_multiple(train_df.tail(252), "u_close", "u_close_norm", "u_close_norm_smooth",
                              symbol_name=symbol_name, ylabel="Price $")

        # create datasets
        look_back = 7
        foresight = 3
        x_train, y_train = palib.create_train_test_dataset(train_df, "u_close_norm_smooth", look_back, foresight)
        x_test, y_test = palib.create_train_test_dataset(test_df, "u_close_norm", look_back, foresight)

        # reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        # iterate through each model to train and test
        epochs = 50
        model_list = [feed_forward, gru_simple, conv1D_gru_net, lstm_simple, lstm_stacked]
        for model_function in model_list:
            # train network
            network, model, model_name = palib.train_network(
                model_function, x_train, y_train, x_test, y_test,
                epochs, symbol_name, c.MODEL_DIR)

            # plot network loss
            plotter.plot_losses(network, model_name, symbol_name, look_back, foresight)

            # use model to predict
            y_pred = model.predict(x_test)

            # plot prediction (y_pred) vs actual (y_test)
            plotter.plot_predictions(y_pred, y_test, scaler, model_name, symbol_name, look_back, foresight)


@click.command(help="Run app_predictor.")
@click.option('--symbol', multiple=False, help="Use data for symbol (e.g. GOOG, SCHB, etc.)")
@click.option('--source', multiple=False, help="Use data from source (e.g. yahoo)")
@click.option('--start', multiple=False, help="Use data from start date (YYY-MM-DD). Period will be (start,end].")
@click.option('--end', multiple=False, help="Use data from end date (YYY-MM-DD). Period will be (start,end].")
def main(symbol, source, start, end):
    start_date = "1980-01-01"
    end_date = date.today()
    # predict_indices(start_date, end_date)
    predict_equities(start_date, end_date)


if __name__ == "__main__":
    main()
