"""
Predictive Analysis Library

@author: eyu
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

# create logger
logger = logging.getLogger("algo-trader")


def compute_hidden_nodes(inputs, outputs, samples, scaling_factor=2):
    """
    Compute the approximate number of hidden nodes for a layer.

    :param inputs: number of input neurons
    :param outputs: number of output neurons
    :param samples: number of samples in the training data
    :param scaling_factor: scaling factor (usually between 2 and 10)
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


def normalize(df, column_source, column_target, scaler):
    series = pd.Series.copy(df[column_source])
    series = series.values.reshape(-1, 1)  # reshape for fit()
    series = scaler.fit_transform(series)

    # add normalized results back to dataframe
    series = series.reshape(-1)
    df = pd.concat([df, pd.Series(series, index=df.index).rename(column_target)], axis=1)

    return df


def normalize_window(df, column_source, column_target, scaler, normalization_window=250):
    # window size to normalize data
    normalization_window = 250

    # use windowed normalization approach;
    # loop over training data in windows of size (normalization_window)
    series = pd.Series.copy(df[column_source])
    series = series.values.reshape(-1, 1)  # reshape for fit()
    for i in range(0, len(series), normalization_window):
        # fit the scaler object on the data in the current window
        scaler.fit(series[i:i + normalization_window, :])
        # transform the data in the current window into values between the chosen feature range (0 and 1)
        series[i:i + normalization_window, :] = scaler.transform(series[i:i + normalization_window, :])

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


def create_train_test_dataset_uni_delete(df, column_source, steps=7):
    x, y = [], []
    series = df[column_source]
    series = series.values.reshape(-1, 1)  # reshape to a MultiIndex
    for i in range(len(series) - steps):
        # append sequence of {steps} values as input sample
        x.append(series[i:(i + steps), 0])
        # append value immediately after sequence of {steps} values as output sample
        y.append(series[i + steps, 0])
    return np.array(x), np.array(y)


def create_train_test_dataset_multi_delete(
        df, column_source_in1, column_source_in2, column_source_out, steps=7):
    source_in1 = df[column_source_in1].values
    source_in2 = df[column_source_in2].values
    source_out = df[column_source_out].values
    # convert to [rows, columns] structure
    in_seq1 = source_in1.reshape((len(source_in1), 1))
    in_seq2 = source_in2.reshape((len(source_in2), 1))
    out_seq = source_out.reshape((len(source_out), 1))
    # horizontally stack columns
    sequences = np.hstack((in_seq1, in_seq2, out_seq))

    x, y = list(), list()
    for i in range(len(sequences) - steps):
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:(i + steps), :-1], sequences[i + steps, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def create_train_test_dataset(sequences, steps=7):
    x, y = list(), list()
    for i in range(len(sequences) - steps):
        # gather the sequence of {steps} inputs and one output for each observation
        seq_x, seq_y = sequences[i:(i + steps), :-1], sequences[i + steps, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def train_network(model_function, x_train, y_train, x_val, y_val, outputs, epochs, symbol_name, model_dir):
    # check if the model directory exists; if not, make it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_name = str(model_function).split(' ')[1]
    # model_file = os.path.join(model_dir, "model-[" + model_name + "]-[" + symbol_name + "][epoch-{epoch:02d}][loss-{loss:.4f}].hdf5")
    model_file = os.path.join(model_dir, "model-[" + model_name + "]-[" + symbol_name + "].hdf5")
    logger.info("Training Network [" + symbol_name + "][Model: " + model_name + "]")
    checkpoint = ModelCheckpoint(model_file, monitor="loss", verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    samples = x_train.shape[0]
    steps = x_train.shape[1]
    features = x_train.shape[2]
    scaling_factor = 3  # smaller value = more hidden nodes; larger value = less hidden nodes
    hidden_nodes = compute_hidden_nodes(steps, outputs, samples, scaling_factor)
    logger.info("Hidden Nodes: {:d}".format(hidden_nodes))
    model = model_function(hidden_nodes, steps, features)
    network = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        batch_size=64,
                        callbacks=callbacks_list)
    model.summary()
    return network, model, model_name
