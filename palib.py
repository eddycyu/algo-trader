"""
Predictive Analysis Library

@author: eyu
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

import talib as talib
import constants as c

# create logger
logger = logging.getLogger("algo-trader")


def compute_hidden_nodes(samples, inputs, outputs, scaling_factor=2):
    """
    Compute the approximate number of hidden nodes for a layer.

    :param samples: number of samples in the data
    :param inputs: number of input neurons
    :param outputs: number of output neurons
    :param scaling_factor: scaling factor (usually between 2 [most hidden nodes] and 10 [least hidden nodes])
    :return:
    """
    return int(round(samples / (scaling_factor * (inputs + outputs))))


def split_df_by_fixed(df, test_size):
    total_size = len(df)
    train_size = total_size - test_size
    train_df = df[:-test_size]
    test_df = df[train_size:]
    logger.info("[Train: %d][Test: %d][Total: %d/%d]"
                % (len(train_df), len(test_df), len(train_df) + len(test_df), len(df)))

    return train_df, test_df


def split_df_by_ratio(df, test_ratio=0.3):
    train_df, test_df = train_test_split(df, shuffle=False, test_size=test_ratio)
    logger.info("[Train: %d][Test: %d][Total: %d/%d]"
                % (len(train_df), len(test_df), len(train_df) + len(test_df), len(df)))

    return train_df, test_df


def normalize_fit_transform(df, column_source, column_target, scaler):
    """
    Normalize (fit and transform) the data in the specified source column using the provided scaler (non-windowed
    approach), and add the normalized data back to the dataframe under the specified target column.

    :param df:
    :param column_source:
    :param column_target:
    :param scaler:
    :return:
    """
    # normalize (transform) data
    series = pd.Series.copy(df[column_source])
    series = series.values.reshape(-1, 1)  # reshape for fit()
    series = scaler.fit_transform(series)
    series = scaler.transform(series)

    # add normalized results back to dataframe
    series = series.reshape(-1)
    df = pd.concat([df, pd.Series(series, index=df.index).rename(column_target)], axis=1)

    return df


def normalize_transform(df, column_source, column_target, scaler):
    """
    Normalize (transform only) the data in the specified source column using the provided scaler (non-windowed
    approach), and add the normalized data back to the dataframe under the specified target column.

    :param df:
    :param column_source:
    :param column_target:
    :param scaler:
    :return:
    """
    # normalize (transform) data
    series = pd.Series.copy(df[column_source])
    series = series.values.reshape(-1, 1)  # reshape for fit()
    series = scaler.transform(series)

    # add normalized results back to dataframe
    series = series.reshape(-1)
    df = pd.concat([df, pd.Series(series, index=df.index).rename(column_target)], axis=1)

    return df


def normalize_window(df, column_source, column_target, scaler, window_size=2500):
    """
    Normalize (fit and transform) the data in the specified source column using the provided scaler (windowed approach),
    and add the normalized data back to the dataframe under the specified target column.

    https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/

    :param df:
    :param column_source:
    :param column_target:
    :param scaler:
    :param window_size:
    :return:
    """
    # NOTE: experiments show that smaller windows sizes (e.g. 50 vs 1000) result in significantly better predictions

    # use windowed normalization approach;
    # loop over training data in windows of size (window_size)
    series = pd.Series.copy(df[column_source])
    series = series.values.reshape(-1, 1)  # reshape for fit()
    for i in range(0, len(series), window_size):
        # fit the scaler object on the data in the current window
        scaler.fit(series[i:i + window_size, :])
        # transform the data in the current window into values between the chosen feature range (0 and 1)
        series[i:i + window_size, :] = scaler.transform(series[i:i + window_size, :])

    # add normalized results back to dataframe
    series = series.reshape(-1)
    df = pd.concat([df, pd.Series(series, index=df.index).rename(column_target)], axis=1)

    return df


def exponential_smooth(df, column_source, column_target, gamma=0.1):
    # use exponential smoothing to remove noise from the data
    series = pd.Series.copy(df[column_source])
    smooth_value = 0.0
    for i in range(len(series)):
        smooth_value = gamma * series[i] + (1 - gamma) * smooth_value
        series[i] = smooth_value

    # add smoothed results back to dataframe
    df = pd.concat([df, pd.Series(series, index=df.index).rename(column_target)], axis=1)

    return df


def create_dataset_x(sequences, steps_in=5):
    """
    Generate dataset for x (inputs) with steps_in, where:

    sequences[0]...sequences[n-1] are inputs
    sequences[n] is output

    Example:
        sequences: [[1,a,A],
                    [2,b,B],
                    [3,c,C],
                    [4,d,D],
                    [5,e,E],
                    [6,f,F]]
        steps_in=3

        output x = ([[1,a],[2,b],[3,c]],
                    [[2,b],[3,c],[4,d]],
                    [[3,c],[4,d],[5,e]],
                    [[4,d],[5,e],[6,f]])

    :param sequences:
    :param steps_in:
    :return:
    """
    x = list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_in = i + steps_in
        # check if we are beyond the dataset
        if end_in > len(sequences):
            break
        # gather the sequence of {steps_in} inputs for each observation
        seq_x = sequences[i:end_in, :-1]
        x.append(seq_x)
    return np.array(x)


def create_dataset_xy(sequences, steps_in=5, steps_out=1):
    """
    Generate dataset for x (inputs) with steps_in and y (outputs) with steps_out, where:

    sequences[0]...sequences[n-1] are inputs
    sequences[n] is output

    Example:
        sequences: [[1,a,A],
                    [2,b,B],
                    [3,c,C],
                    [4,d,D],
                    [5,e,E],
                    [6,f,F]]
        steps_in=3
        steps_out=2

        output x = ([[1,a],[2,b],[3,c]],
                    [[2,b],[3,c],[4,d]],
                    [[3,c],[4,d],[5,e]])
        output y = ([[C],[D]],
                    [[D],[E]],
                    [[E],[F]])

    :param sequences:
    :param steps_in:
    :param steps_out:
    :return:
    """
    x, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_in = i + steps_in
        end_out = end_in + steps_out - 1
        # check if we are beyond the dataset
        if end_out > len(sequences):
            break
        # gather the sequence of {steps_in} inputs and {steps_out} output for each observation
        seq_x, seq_y = sequences[i:end_in, :-1], sequences[end_in-1:end_out, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def create_dataset_xy_shift(sequences, steps_in=5, steps_out=1):
    """
    Generate dataset for x (inputs) with steps_in and y (outputs) with steps_out, where:

    sequences[0]...sequences[n-1] are inputs
    sequences[n] is output

    Example:
        sequences: [[1,a,A],
                    [2,b,B],
                    [3,c,C],
                    [4,d,D],
                    [5,e,E],
                    [6,f,F]]
        steps_in=3
        steps_out=2

        output x = ([[1,a],[2,b],[3,c]],
                    [[2,b],[3,c],[4,d]])
        output y = ([[D],[E]],
                    [[E],[F]])

    :param sequences:
    :param steps_in:
    :param steps_out:
    :return:
    """
    x, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_in = i + steps_in
        end_out = end_in + steps_out
        # check if we are beyond the dataset
        if end_out > len(sequences):
            break
        # gather the sequence of {steps_in} inputs and {steps_out} output for each observation
        seq_x, seq_y = sequences[i:end_in, :-1], sequences[end_in:end_out, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def create_dataset_parallel(sequences, steps_in=5, steps_out=1):
    """
    Generate dataset for x (inputs) with steps_in and y (outputs) with steps_out, where:

    sequences[0]...sequences[n] are both inputs and outputs

    Example:
        sequences: [[1,a,A],
                    [2,b,B],
                    [3,c,C],
                    [4,d,D],
                    [5,e,E],
                    [6,f,F]]
        steps_in=3
        steps_out=2

        output x = ([[1,a,A],[2,b,B],[3,c,C]],
                    [[2,b,B][3,c,C],[4,d,D])
        output y = ([[4,d,D],[5,e,E]],
                    [[5,e,E],[6,f,F]])

    :param sequences:
    :param steps_in:
    :param steps_out:
    :return:
    """
    x, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_in = i + steps_in
        end_out = end_in + steps_out
        # check if we are beyond the dataset
        if end_out > len(sequences):
            break
        # gather the sequence of {steps_in} inputs and {steps_out} output for each observation
        seq_x, seq_y = sequences[i:end_in, :], sequences[end_in, :]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def train_network(network_function, x_train, y_train, x_val, y_val, steps_out, epochs, symbol_name, feature, model_dir):
    # check if the model directory exists; if not, make it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    samples = x_train.shape[0]      # number of samples in training data
    steps_in = x_train.shape[1]     # number of input steps in training data
    features = x_train.shape[2]     # number of features in training data
    hidden_nodes = compute_hidden_nodes(samples, steps_in, steps_out)
    model = network_function(hidden_nodes, steps_in, steps_out, features)
    network_name = str(network_function).split(' ')[1]
    model_file = os.path.join(
        model_dir,
        "model_{symbol_name}_{feature}_{network_name}_{hidden_nodes}_{steps_in}_{steps_out}_{features}_.hdf5".format(
            symbol_name=symbol_name, feature=feature, network_name=network_name, hidden_nodes=hidden_nodes,
            steps_in=steps_in, steps_out=steps_out, features=features).lower())
    logger.info("Training Network [" + network_name + "][" + symbol_name + "][" + feature + "]")
    checkpoint = ModelCheckpoint(model_file, monitor="loss", verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor="loss", patience=50)
    #callbacks_list = [checkpoint, early_stopping]
    callbacks_list = [checkpoint]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=64,
        callbacks=callbacks_list)
    model.summary()
    return history, network_name, model, model_file


def train_and_test_networks_save(
        x_train, y_train, x_val, y_val, x_test, y_test, scaler_test,
        steps_out, epochs, symbol_name, feature, network_list, model_dir, plotter):

    steps_in = x_train.shape[1]     # number of input steps in training data
    features = x_train.shape[2]     # number of features in [train/val/test] data

    # reshape the data [samples, steps_in, features]
    x_train = np.reshape(x_train, (x_train.shape[0], steps_in, features))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], features))

    # iterate through each network to train and test
    for network_function in network_list:
        # train network
        history, network_name, model, model_file = train_network(
            network_function, x_train, y_train, x_val, y_val, steps_out, epochs, symbol_name, feature, model_dir)

        # plot network loss
        plotter.plot_loss(history, network_name, symbol_name, feature, steps_in, steps_out)

        # the model weights (that are considered the best) are loaded into the model
        model.load_weights(model_file)

        # use model to predict
        y_pred = model.predict(x_test)

        # plot prediction (y_pred) vs actual (y_test)
        if steps_out == 1:
            plotter.plot_prediction(
                y_pred[-90:], y_test[-90:], scaler_test, network_name, symbol_name, feature, steps_in, steps_out)
        else:
            plotter.plot_prediction(
                y_pred[-1], y_test[-90:, 0], scaler_test, network_name, symbol_name, feature, steps_in, steps_out)


def train_and_test_networks(
        x_train, y_train, x_val, y_val, x_test, y_test, scaler_test,
        steps_out, epochs, symbol_name, feature, network_list, model_dir, plotter):

    steps_in = x_train.shape[1]     # number of input steps in training data
    features = x_train.shape[2]     # number of features in [train/val/test] data

    # reshape the data [samples, steps_in, features]
    x_train = np.reshape(x_train, (x_train.shape[0], steps_in, features))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], features))

    # iterate through each network to train and test
    for network_function in network_list:
        # train network
        history, network_name, model, model_file = train_network(
            network_function, x_train, y_train, x_val, y_val, steps_out, epochs, symbol_name, feature, model_dir)

        # plot network loss
        plotter.plot_loss(history, network_name, symbol_name, feature, steps_in, steps_out)

        # the model weights (that are considered the best) are loaded into the model
        model.load_weights(model_file)

        # use model to predict
        y_pred = model.predict(x_test)

        # plot prediction (y_pred) vs actual (y_test)
        if steps_out == 1:
            plotter.plot_prediction(
                y_pred[-90:], y_test[-90:], scaler_test, network_name, symbol_name, feature, steps_in, steps_out)
        else:
            plotter.plot_prediction(
                y_pred[-1], y_test[-90:, 0], scaler_test, network_name, symbol_name, feature, steps_in, steps_out)


def get_normalized_dataset_x_for_close(sequences, steps_in):
    # create dataset
    x = create_dataset_x(sequences, steps_in)

    return x


def get_normalized_train_dataset_xy_for_close(df, steps_in, steps_out):
    # normalize (windowed approach) input data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = normalize_window(df, c.CLOSE, "p_in_close", scaler)
    #df = normalize_fit_transform(df, c.CLOSE, "p_in_close", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_close", "p_out_close")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_close", "p_in_close_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_close_exp"].values
    out_values = df["p_out_close"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y, scaler


def get_normalized_test_dataset_xy_for_close(df, steps_in, steps_out, scaler):
    # normalize (non-windowed approach) input data using specified scaler
    df = normalize_transform(df, c.CLOSE, "p_in_close", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_close", "p_out_close")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_close", "p_in_close_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_close_exp"].values
    out_values = df["p_out_close"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y


def get_normalized_dataset_x_for_low(sequences, steps_in):
    # create dataset
    x = create_dataset_x(sequences, steps_in)

    return x


def get_normalized_train_dataset_xy_for_low(df, steps_in, steps_out):
    # normalize (windowed approach) data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = normalize_window(df, c.LOW, "p_in_low", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_low", "p_out_low")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_low", "p_in_low_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_low_exp"].values
    out_values = df["p_out_low"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y, scaler


def get_normalized_test_dataset_xy_for_low(df, steps_in, steps_out, scaler):
    # normalize (non-windowed approach) data using specified scaler
    df = normalize_transform(df, c.LOW, "p_in_low", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_low", "p_out_low")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_low", "p_in_low_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_low_exp"].values
    out_values = df["p_out_low"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y


def get_normalized_dataset_x_for_high(sequences, steps_in):
    # create dataset
    x = create_dataset_x(sequences, steps_in)

    return x


def get_normalized_train_dataset_xy_for_high(df, steps_in, steps_out):
    # normalize (windowed approach) data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = normalize_window(df, c.HIGH, "p_in_high", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_high", "p_out_high")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_high", "p_in_high_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_high_exp"].values
    out_values = df["p_out_high"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y, scaler


def get_normalized_test_dataset_xy_for_high(df, steps_in, steps_out, scaler):
    # normalize (non-windowed approach) data using specified scaler
    df = normalize_transform(df, c.HIGH, "p_in_high", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_high", "p_out_high")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_high", "p_in_high_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_high_exp"].values
    out_values = df["p_out_high"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y


def get_normalized_dataset_x_for_macd(sequences, steps_in):
    # create dataset
    x = create_dataset_x(sequences, steps_in)

    return x


def get_normalized_train_dataset_xy_for_macd(df, steps_in, steps_out):
    # normalize (windowed approach) data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = normalize_window(df, "u_macd-12-26-9", "p_in_macd", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_macd", "p_out_macd")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_macd", "p_in_macd_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_macd_exp"].values
    out_values = df["p_out_macd"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y, scaler


def get_normalized_test_dataset_xy_for_macd(df, steps_in, steps_out, scaler):
    # normalize (non-windowed approach) data using specified scaler
    df = normalize_transform(df, "u_macd-12-26-9", "p_in_macd", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_macd", "p_out_macd")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_macd", "p_in_macd_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_macd_exp"].values
    out_values = df["p_out_macd"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y


def get_normalized_dataset_x_for_macd_signal(sequences, steps_in):
    # create dataset
    x = create_dataset_x(sequences, steps_in)

    return x


def get_normalized_train_dataset_xy_for_macd_signal(df, steps_in, steps_out):
    # normalize (windowed approach) data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = normalize_window(df, "u_macd_signal-12-26-9", "p_in_macd_signal", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_macd_signal", "p_out_macd_signal")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_macd_signal", "p_in_macd_signal_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_macd_signal_exp"].values
    out_values = df["p_out_macd_signal"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y, scaler


def get_normalized_test_dataset_xy_for_macd_signal(df, steps_in, steps_out, scaler):
    # normalize (non-windowed approach) data using specified scaler
    df = normalize_transform(df, "u_macd_signal-12-26-9", "p_in_macd_signal", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_macd_signal", "p_out_macd_signal")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_macd_signal", "p_in_macd_signal_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_macd_signal_exp"].values
    out_values = df["p_out_macd_signal"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y


def get_normalized_dataset_x_for_rsi(sequences, steps_in):
    # create dataset
    x = create_dataset_x(sequences, steps_in)

    return x


def get_normalized_train_dataset_xy_for_rsi(df, steps_in, steps_out):
    # normalize (windowed approach) data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = normalize_window(df, "u_rsi-14", "p_in_rsi", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_rsi", "p_out_rsi")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_rsi", "p_in_rsi_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_rsi_exp"].values
    out_values = df["p_out_rsi"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y, scaler


def get_normalized_test_dataset_xy_for_rsi(df, steps_in, steps_out, scaler):
    # normalize (non-windowed approach) data using specified scaler
    df = normalize_transform(df, "u_rsi-14", "p_in_rsi", scaler)

    # add normalized input data to dataframe as normalized output data
    df = talib.copy_column(df, "p_in_rsi", "p_out_rsi")

    # use exponential smoothing to remove noise from the normalized input data (helps to reduce training loss)
    # NOTE: helps to make predictions more precise (needs further testing)...
    df = exponential_smooth(df, "p_in_rsi", "p_in_rsi_exp", gamma=0.1)

    # convert to [rows, columns] structure
    in_values = df["p_in_rsi_exp"].values
    out_values = df["p_out_rsi"].values
    in_values = in_values.reshape((len(in_values), 1))
    out_values = out_values.reshape((len(out_values), 1))

    # horizontally stack columns
    sequences = np.hstack((in_values, out_values))

    # create dataset
    x, y = create_dataset_xy_shift(sequences, steps_in, steps_out)

    return df, sequences, x, y


def get_normalized_dataset_x(sequences, feature, steps_in):
    switcher = {
        "close": get_normalized_dataset_x_for_close,
        "low": get_normalized_dataset_x_for_low,
        "high": get_normalized_dataset_x_for_high,
        "macd": get_normalized_dataset_x_for_macd,
        "macd-signal": get_normalized_dataset_x_for_macd_signal,
        "rsi": get_normalized_dataset_x_for_rsi
    }
    func = switcher.get(feature)
    return func(sequences, steps_in)


def get_normalized_train_dataset_xy(df, feature, steps_in, steps_out):
    switcher = {
        "close": get_normalized_train_dataset_xy_for_close,
        "low": get_normalized_train_dataset_xy_for_low,
        "high": get_normalized_train_dataset_xy_for_high,
        "macd": get_normalized_train_dataset_xy_for_macd,
        "macd-signal": get_normalized_train_dataset_xy_for_macd_signal,
        "rsi": get_normalized_train_dataset_xy_for_rsi
    }
    func = switcher.get(feature)
    return func(df, steps_in, steps_out)


def get_normalized_test_dataset_xy(df, feature, steps_in, steps_out, scaler):
    switcher = {
        "close": get_normalized_test_dataset_xy_for_close,
        "low": get_normalized_test_dataset_xy_for_low,
        "high": get_normalized_test_dataset_xy_for_high,
        "macd": get_normalized_test_dataset_xy_for_macd,
        "macd-signal": get_normalized_test_dataset_xy_for_macd_signal,
        "rsi": get_normalized_test_dataset_xy_for_rsi
    }
    func = switcher.get(feature)
    return func(df, steps_in, steps_out, scaler)
