"""
Price Predictor

https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046

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
from keras import backend

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
    model.compile(optimizer='adam', loss='mae')
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


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# Custom LSTM 1
def lstm_custom1(hidden_nodes, steps=5, features=1):
    """
    A custom LSTM model.

    LSTM with tanh
    Dense with None

    :param hidden_nodes: number of hidden nodes
    :param steps: number of (look back) time steps for each sample input
    :param features: number of features for each sample input (e.g. 1 for univariate or 2+ for multivariate time series)
    :return: simple LSTM model
    """
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(steps, features)))  # default activation: tanh
    model.add(Dense(1))  # default activation: None
    model.compile(optimizer='adam', loss='mae')
    return model


# Custom LSTM 2
def lstm_custom2(hidden_nodes, steps=5, features=1):
    """
    A custom LSTM model.

    :param hidden_nodes: number of hidden nodes
    :param steps: number of (look back) time steps for each sample input
    :param features: number of features for each sample input (e.g. 1 for univariate or 2+ for multivariate time series)
    :return: simple LSTM model
    """
    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(steps, features)))  # default activation: tanh
    model.add(Dense(1))  # default activation: None
    model.compile(optimizer='adam', loss='mae')
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


def process_indices(symbols, start_date, end_date):
    """
    :param symbols: list of index symbols to fetch, predict and plot
    :param start_date: earliest date to fetch
    :param end_date:: latest date to fetch
    :return:
    """
    data_reader = StooqDataReader()
    plotter = PAPlot(c.CHART_PA_DIR)
    for symbol in symbols:
        symbol_name = symbol[1:]  # strip off the ^ character
        # load data
        df = data_reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Close' (no adjusted close for indices) as our close price
        df = talib.copy_column(df, "Close", c.CLOSE)
        # predict and plot all
        predict_all(symbol_name, df, plotter)


def process_equities(symbols, start_date, end_date):
    """
    :param symbols: list of equity symbols to fetch, predict and plot
    :param start_date: earliest date to fetch
    :param end_date:: latest date to fetch
    :return:
    """
    data_reader = YahooDataReader()
    plotter = PAPlot(c.CHART_PA_DIR)
    for symbol in symbols:
        symbol_name = symbol
        # load data
        df = data_reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Adj Close' as our close price
        df = talib.copy_column(df, "Adj Close", c.CLOSE)
        # predict and plot all
        predict_all(symbol_name, df, plotter)


def predict_all(symbol_name, df, plotter):
    # compute technical indicators
    df = compute_ta(df)

    # remove rows with missing data
    df = df.dropna()

    # split data to create training (70%), validation (15%) and testing (15%) data sets
    train_df, val_test_df = palib.split_df_by_ratio(df, test_ratio=0.3)
    val_df, test_df = palib.split_df_by_ratio(val_test_df, test_ratio=0.5)

    # normalize (windowed approach) training data
    train_window_size = int(round(len(train_df) / 4))
    logger.info("Train Window Size: {:d}".format(train_window_size))
    train_scaler_feature1 = MinMaxScaler(feature_range=(0, 1))
    train_df = palib.normalize_window(train_df, c.CLOSE, "u_feature1", train_scaler_feature1, train_window_size)
    train_scaler_feature2 = MinMaxScaler(feature_range=(0, 1))
    train_df = palib.normalize_window(train_df, c.VOLUME, "u_feature2", train_scaler_feature2, train_window_size)
    train_scaler_feature3 = MinMaxScaler(feature_range=(0, 1))
    train_df = palib.normalize_window(train_df, c.PREV_CLOSE, "u_feature3", train_scaler_feature3, train_window_size)
    train_scaler_feature4 = MinMaxScaler(feature_range=(0, 1))
    train_df = palib.normalize_window(train_df, c.OPEN, "u_feature4", train_scaler_feature4, train_window_size)


    # normalize (non-windowed approach) validation and testing data
    val_window_size = int(round(len(val_df) / 4))
    logger.info("Validation Window Size: {:d}".format(val_window_size))
    val_scaler_feature1 = MinMaxScaler(feature_range=(0, 1))
    #val_df = palib.normalize(val_df, c.CLOSE, "u_feature1", val_scaler_feature1)
    val_df = palib.normalize_window(val_df, c.CLOSE, "u_feature1", val_scaler_feature1, val_window_size)
    val_scaler_feature2 = MinMaxScaler(feature_range=(0, 1))
    #val_df = palib.normalize(val_df, c.VOLUME, "u_feature2", val_scaler_feature2)
    val_df = palib.normalize_window(val_df, c.VOLUME, "u_feature2", val_scaler_feature2, val_window_size)
    val_scaler_feature3 = MinMaxScaler(feature_range=(0, 1))
    #val_df = palib.normalize(val_df, c.PREV_CLOSE, "u_feature3", val_scaler_feature3)
    val_df = palib.normalize_window(val_df, c.PREV_CLOSE, "u_feature3", val_scaler_feature3, val_window_size)
    val_scaler_feature4 = MinMaxScaler(feature_range=(0, 1))
    #val_df = palib.normalize(val_df, c.OPEN, "u_feature4", val_scaler_feature4)
    val_df = palib.normalize_window(val_df, c.OPEN, "u_feature4", val_scaler_feature4, val_window_size)

    test_window_size = int(round(len(test_df) / 4))
    logger.info("Test Window Size: {:d}".format(test_window_size))
    test_scaler_feature1 = MinMaxScaler(feature_range=(0, 1))
    #test_df = palib.normalize(test_df, c.CLOSE, "u_feature1", test_scaler_feature1)
    test_df = palib.normalize_window(test_df, c.CLOSE, "u_feature1", test_scaler_feature1, test_window_size)
    test_scaler_feature2 = MinMaxScaler(feature_range=(0, 1))
    #test_df = palib.normalize(test_df, c.VOLUME, "u_feature2", test_scaler_feature2)
    test_df = palib.normalize_window(test_df, c.VOLUME, "u_feature2", test_scaler_feature2, test_window_size)
    test_scaler_feature3 = MinMaxScaler(feature_range=(0, 1))
    #test_df = palib.normalize(test_df, c.PREV_CLOSE, "u_feature3", test_scaler_feature3)
    test_df = palib.normalize_window(test_df, c.PREV_CLOSE, "u_feature3", test_scaler_feature3, test_window_size)
    test_scaler_feature4 = MinMaxScaler(feature_range=(0, 1))
    #test_df = palib.normalize(test_df, c.OPEN, "u_feature4", test_scaler_feature4)
    test_df = palib.normalize_window(test_df, c.OPEN, "u_feature4", test_scaler_feature4, test_window_size)

    # use exponential smoothing to remove noise from the normalized data (helps to reduce training loss)
    train_df = palib.exponential_smooth(train_df, "u_feature1", "u_feature1_exp", gamma=0.1)
    train_df = palib.exponential_smooth(train_df, "u_feature2", "u_feature2_exp", gamma=0.1)
    train_df = palib.exponential_smooth(train_df, "u_feature3", "u_feature3_exp", gamma=0.1)
    train_df = palib.exponential_smooth(train_df, "u_feature4", "u_feature4_exp", gamma=0.1)
    val_df = palib.exponential_smooth(val_df, "u_feature1", "u_feature1_exp", gamma=0.1)
    val_df = palib.exponential_smooth(val_df, "u_feature2", "u_feature2_exp", gamma=0.1)
    val_df = palib.exponential_smooth(val_df, "u_feature3", "u_feature3_exp", gamma=0.1)
    val_df = palib.exponential_smooth(val_df, "u_feature4", "u_feature4_exp", gamma=0.1)
    test_df = palib.exponential_smooth(test_df, "u_feature1", "u_feature1_exp", gamma=0.1)
    test_df = palib.exponential_smooth(test_df, "u_feature2", "u_feature2_exp", gamma=0.1)
    test_df = palib.exponential_smooth(test_df, "u_feature3", "u_feature3_exp", gamma=0.1)
    test_df = palib.exponential_smooth(test_df, "u_feature4", "u_feature4_exp", gamma=0.1)

    # create sequences for train/val/test
    train_in1 = train_df["u_feature1_exp"].values
    train_in2 = train_df["u_feature2_exp"].values
    train_in3 = train_df["u_feature3_exp"].values
    train_in4 = train_df["u_feature4_exp"].values
    train_out = train_df["u_feature1"].values
    val_in1 = val_df["u_feature1_exp"].values
    val_in2 = val_df["u_feature2_exp"].values
    val_in3 = val_df["u_feature3_exp"].values
    val_in4 = val_df["u_feature4_exp"].values
    val_out = val_df["u_feature1"].values
    test_in1 = test_df["u_feature1_exp"].values
    test_in2 = test_df["u_feature2_exp"].values
    test_in3 = test_df["u_feature3_exp"].values
    test_in4 = test_df["u_feature4_exp"].values
    test_out = test_df["u_feature1"].values

    # convert to [rows, columns] structure
    train_in1 = train_in1.reshape((len(train_in1), 1))
    train_in2 = train_in2.reshape((len(train_in2), 1))
    train_in3 = train_in3.reshape((len(train_in3), 1))
    train_in4 = train_in4.reshape((len(train_in4), 1))
    train_out = train_out.reshape((len(train_out), 1))
    val_in1 = val_in1.reshape((len(val_in1), 1))
    val_in2 = val_in2.reshape((len(val_in2), 1))
    val_in3 = val_in3.reshape((len(val_in3), 1))
    val_in4 = val_in4.reshape((len(val_in4), 1))
    val_out = val_out.reshape((len(val_out), 1))
    test_in1 = test_in1.reshape((len(test_in1), 1))
    test_in2 = test_in2.reshape((len(test_in2), 1))
    test_in3 = test_in3.reshape((len(test_in3), 1))
    test_in4 = test_in4.reshape((len(test_in4), 1))
    test_out = test_out.reshape((len(test_out), 1))

    # horizontally stack columns
    #train_sequences = np.hstack((train_in1, train_out))
    #val_sequences = np.hstack((val_in1, val_out))
    #test_sequences = np.hstack((test_in1, test_out))
    #train_sequences = np.hstack((train_in1, train_in2, train_out))
    #val_sequences = np.hstack((val_in1, val_in2, val_out))
    #test_sequences = np.hstack((test_in1, test_in2, test_out))
    #train_sequences = np.hstack((train_in1, train_in2, train_in3, train_out))
    #val_sequences = np.hstack((val_in1, val_in2, val_in3, val_out))
    #test_sequences = np.hstack((test_in1, test_in2, test_in3, test_out))
    train_sequences = np.hstack((train_in1, train_in4, train_out))
    val_sequences = np.hstack((val_in1, val_in4, val_out))
    test_sequences = np.hstack((test_in1, test_in4, test_out))

    # create datasets
    look_back = 20
    foresight = 5
    x_train, y_train = palib.create_train_test_dataset(train_sequences, look_back)
    x_val, y_val = palib.create_train_test_dataset(val_sequences, look_back)
    x_test, y_test = palib.create_train_test_dataset(test_sequences, look_back)

    # number of features in training data
    features = x_train.shape[2]

    # reshape the data [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], features))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], features))

    # iterate through each model to train and test
    outputs = 1
    epochs = 50
    # model_list = [feed_forward, gru_simple, conv1D_gru_net, lstm_simple, lstm_stacked]
    model_list = [lstm_custom2]
    for model_function in model_list:
        # train network
        network, model, model_name = palib.train_network(
            model_function, x_train, y_train, x_val, y_val,
            outputs, epochs, symbol_name, c.MODEL_DIR)

        # plot network loss
        plotter.plot_loss(network, model_name, symbol_name, look_back, foresight)

        # use model to predict
        y_pred = model.predict(x_test)

        # plot prediction (y_pred) vs actual (y_test)
        plotter.plot_prediction(y_pred[-60:], y_test[-60:], test_scaler_feature1, model_name, symbol_name, look_back, foresight)


def compute_ta(df):
    # add previous close
    df = talib.copy_column_shift(df, c.CLOSE, c.PREV_CLOSE, 1)
    # compute change and percentage change of closing price
    df = talib.compute_change(df, c.CLOSE, c.CHANGE, c.CHANGE_PC, (1, 30, 90, 365))
    # compute daily change between open price and previous closing price
    df = talib.compute_daily_change_between_current_and_previous(
        df, c.OPEN, c.CLOSE, c.OPEN_PREV_CLOSE, c.OPEN_PREV_CLOSE_PC)
    # compute 52 week range (low~high)
    df = talib.compute_52_week_range(df, c.LOW, c.HIGH, c.R52_WK_LOW, c.R52_WK_HIGH)
    # compute change and percentage change of close price above the 52 week low price
    df = talib.compute_change_pc_above(df, c.CLOSE, c.R52_WK_LOW, c.CLOSE_ABOVE_52_WK_LOW, c.CLOSE_ABOVE_52_WK_LOW_PC)
    # compute change and percentage change of close price below the 52 week high price
    df = talib.compute_change_pc_below(df, c.CLOSE, c.R52_WK_HIGH, c.CLOSE_BELOW_52_WK_HIGH, c.CLOSE_BELOW_52_WK_HIGH_PC)
    # compute SMA of close price
    df = talib.compute_sma(df, c.CLOSE, c.SMA, (50, 100, 200))
    # compute EMA of close price
    df = talib.compute_ema(df, c.CLOSE, c.EMA, (12, 26, 50, 200))
    # compute golden/death crosses for SMA
    df = talib.compute_ma_cross(df, c.SMA, c.SMA_GOLDEN_CROSS, c.SMA_DEATH_CROSS, 50, 200)
    # compute golden/death crosses for EMA
    df = talib.compute_ma_cross(df, c.EMA, c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, 12, 26)
    df = talib.compute_ma_cross(df, c.EMA, c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, 50, 200)
    # compute average daily trading volume
    df = talib.compute_sma(df, c.VOLUME, c.ADTV, (30, 90, 180, 365))
    # compute BB of close price with SMA period of 20 and standard deviation of 2
    df = talib.compute_bb(df, c.CLOSE, c.BB, 20, 2)
    # compute MACD of close price
    df = talib.compute_macd(df, c.CLOSE, c.EMA, c.MACD, c.MACD_SIGNAL, c.MACD_HISTOGRAM, 12, 26, 9)
    # compute RSI of close price
    df = talib.compute_rsi(df, c.CLOSE, c.RSI_AVG_GAIN, c.RSI_AVG_LOSS, c.RSI, (7, 14, 21))

    return df


def predict_equities_delete(start_date, end_date):
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
        # train_df = palib.normalize_window(train_df, "u_close", "u_close_norm", scaler, 250)
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
        #train_df, test_df = palib.split_df_by_ratio(df, test_ratio=0.3)
        train_df, test_df = palib.split_df_by_fixed(df, test_size=708)

        # scaler to normalize feature range
        scaler = MinMaxScaler(feature_range=(0, 1))

        # normalize (windowed approach) training data
        train_df = palib.normalize_window(train_df, "u_close", "u_close_norm", scaler, 250)

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
            plotter.plot_loss(network, model_name, symbol_name, look_back, foresight)

            # use model to predict
            y_pred = model.predict(x_test)

            # plot prediction (y_pred) vs actual (y_test)
            plotter.plot_prediction(y_pred, y_test, scaler, model_name, symbol_name, look_back, foresight)


@click.command(help="Run app_predictor.")
@click.option('--indices', '-i', multiple=True, help="Indices to analyze (e.g. ^SPX, ^DJI, ^TWSE, ^KOSPI, etc.)")
@click.option('--stocks', '-s', multiple=True, help="Stocks to analyze (e.g. AAPL, AMZN, GOOGL, TSLA, etc.)")
@click.option('--etfs', '-e', multiple=True, help="ETFs to analyze (e.g. SCHB, SCHX, VTI, VOO, etc.)")
@click.option('--start', multiple=False, help="Use data from start date (YYY-MM-DD). Period will be (start,end].")
@click.option('--end', multiple=False, help="Use data from end date (YYY-MM-DD). Period will be (start,end].")
def main(indices, stocks, etfs, start, end):
    # defaults
    symbols_indices = ("^SPX", "^NDQ", "^NDX", "^DJI", "^TWSE", "^KOSPI", "^NKX", "^HSI", "^STI", "^SHC", "^SHBS")
    symbols_stocks = ("AAPL", "AMZN", "FB", "GOOGL", "GOOG", "MSFT", "NFLX", "TSLA", "CRM", "WDAY",
                      "BABA", "TCTZF", "NIO", "NOW", "SNOW", "PLTR")
    symbols_etfs = ("SCHB", "SCHX", "SCHG", "SCHA", "SCHF", "SCHE", "SCHD", "SCHH", "SCHZ",
                    "VTI", "VOO", "VTV", "VEA", "VWO", "QQQ", "SPY", "GLD", "IWF", "AGG")
    start_date = "1980-01-01"
    end_date = date.today()

    # initialize symbols (indices, stocks and ETFs) from command line
    if indices:
        symbols_indices = indices
    if stocks:
        symbols_stocks = stocks
    if etfs:
        symbols_etfs = etfs

    # initialize start (inclusive) and end (inclusive) date range from command line
    if start:
        start_date = start
    if end:
        end_date = end

    # process
    # process_indices(symbols_indices, start_date, end_date)
    process_equities(symbols_stocks, start_date, end_date)
    #process_equities(symbols_etfs, start_date, end_date)


if __name__ == "__main__":
    main()
