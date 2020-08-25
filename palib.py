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


def split_train_test_df_by_fixed(df, test_size):
    data_size = len(df)
    train_size = data_size - test_size
    train_df = df[:-test_size]
    test_df = df[data_size - test_size:]

    logger.info("Total Data Size: [Expected: %d][Actual: %d]" % (data_size, len(train_df) + len(test_df)))
    logger.info("Train Data Size: [Expected: %d][Actual: %d]" % (train_size, len(train_df)))
    logger.info("Test Data Size : [Expected: %d][Actual: %d]" % (test_size, len(test_df)))

    return train_df, test_df


def split_train_test_df_by_ratio(df, test_ratio=0.2):
    data_size = len(df)
    test_size = int(data_size * test_ratio)
    train_size = data_size - test_size
    train_df, test_df = train_test_split(df, shuffle=False, test_size=test_ratio)

    logger.info("Total Data Size: [Expected: %d][Actual: %d]" % (data_size, len(train_df) + len(test_df)))
    logger.info("Train Data Size: [Expected: %d][Actual: %d]" % (train_size, len(train_df)))
    logger.info("Test Data Size : [Expected: %d][Actual: %d]" % (test_size, len(test_df)))

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


def create_train_test_dataset(df, column_source, look_back=7, foresight=3):
    x, y = [], []
    series = df[column_source]
    series = series.values.reshape(-1, 1)  # reshape to a MultiIndex
    for i in range(len(series) - look_back - foresight):
        # append sequence of {look_back} values as features forming an observation
        x.append(series[i:(i + look_back), 0])
        # append value occurring {foresight} time-steps into future
        y.append(series[i + (look_back + foresight), 0])
    return np.array(x), np.array(y)


def create_train_test_dataset_multiple(df, column_source1, column_source2, look_back=7, foresight=3):
    x, y = [], []
    source1_series = df[column_source1]
    source2_series = df[column_source2]
    # reshape to a MultiIndex
    source1_series = source1_series.values.reshape(-1, 1)
    source2_series = source2_series.values.reshape(-1, 1)

    for i in range(len(df) - look_back - foresight):
        # append sequence of {look_back} values as features forming an observation
        x.append(np.append(source1_series[i:(i + look_back), 0], source2_series[i:(i + look_back), 0]))
        # append value occurring {foresight} time-steps into future
        y.append(source1_series[i + (look_back + foresight), 0])
    return np.array(x), np.array(y)


def train_network(model_function, x_train, y_train, x_test, y_test, epochs, symbol_name, model_dir):
    # check if the model directory exists; if not, make it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_name = str(model_function).split(' ')[1]
    #model_file = os.path.join(model_dir, "model-[" + model_name + "]-[" + symbol_name + "][epoch-{epoch:02d}][loss-{loss:.4f}].hdf5")
    model_file = os.path.join(model_dir, "model-[" + model_name + "]-[" + symbol_name + "].hdf5")
    logger.info("Training Network [" + symbol_name + "][Model: " + model_name + "]")
    checkpoint = ModelCheckpoint(model_file, monitor="loss", verbose=2, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model = model_function()
    network = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=64,
                        callbacks=callbacks_list)
    model.summary()
    return network, model, model_name
