"""
Technical Analysis Library

Library of functions to compute various technical indicators.

@author: eyu
"""

import logging
import numpy as np
import pandas as pd
import math as math
import statistics as stats
import datetime

import constants as c

# create logger
logger = logging.getLogger("algo-trader")


def copy_column(df, column_source, column_target):
    """
    Copy an existing column to a new column in dataframe.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to copy
    :param column_target: name of target column in dataframe for copied values
    :return: modified dataframe
    """
    df[column_target] = df[column_source]
    return df


def copy_column_shift(df, column_source, column_target, shift_amount):
    """
    Copy an existing column (shifted by shift_amount) to a new column in dataframe.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to copy
    :param column_target: name of target column in dataframe for copied values
    :param shift_amount: amount of rows to shift
    :return: modified dataframe
    """
    df[column_target] = df[column_source].shift(shift_amount)
    return df


def compute_sma_custom(df, column_source, column_target_sma, time_period):
    """
    Compute Simple Moving Average (SMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to compute SMA (e.g. close price)
    :param column_target_sma: prefix of target column in dataframe for SMA results
    :param time_period: time period (number of days for SMA)
    :return: modified dataframe
    """
    # compute SMA
    history_values = []
    sma_values = []
    for value in df[column_source]:
        history_values.append(value)
        if len(history_values) > time_period:
            del (history_values[0])
        sma_values.append(stats.mean(history_values))

    # add computed SMA results back to dataframe
    key_sma = column_target_sma + "-{:d}".format(time_period)
    df[key_sma] = sma_values

    return df


def compute_sma(df, column_source, column_target_sma, time_periods):
    """
    Compute Simple Moving Average (SMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to compute SMA (e.g. close price)
    :param column_target_sma: prefix of target column in dataframe for SMA results
    :param time_periods: list of time periods (number of days for SMA)
    :return: modified dataframe
    """
    # compute SMA for each time period and add results back to dataframe
    for time_period in time_periods:
        key_sma = column_target_sma + "-{:d}".format(time_period)
        df[key_sma] = df[column_source].rolling(window=time_period, min_periods=1).mean()

    return df


def compute_ema_custom(
        df, column_source, column_target_ema, column_target_golden_cross, column_target_death_cross,
        time_period_fast, time_period_slow):
    """
    Compute Exponential Moving Average (EMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to compute EMA (e.g. close price)
    :param column_target_ema: prefix of target column in dataframe for EMA results
    :param column_target_golden_cross: name of target column in dataframe for golden cross results
    :param column_target_death_cross: name of target column in dataframe for death cross results
    :param time_period_fast: number of days over which to average for fast EMA
    :param time_period_slow: number of days over which to average for slow EMA
    :return: modified dataframe
    """
    # compute EMA
    k_fast = 2 / (time_period_fast + 1)  # fast EMA smoothing factor
    ema_fast = 0
    k_slow = 2 / (time_period_slow + 1)  # slow EMA smoothing factor
    ema_slow = 0
    ema_fast_values = []
    ema_slow_values = []
    for value in df[column_source]:
        if ema_fast == 0:  # first observation
            ema_fast = value
            ema_slow = value
        else:
            ema_fast = (value - ema_fast) * k_fast + ema_fast
            ema_slow = (value - ema_slow) * k_slow + ema_slow
        ema_fast_values.append(ema_fast)
        ema_slow_values.append(ema_slow)

    # add computed EMA results back to dataframe
    key_ema_fast = column_target_ema + "-{:d}".format(time_period_fast)
    key_ema_slow = column_target_ema + "-{:d}".format(time_period_slow)
    df[key_ema_fast] = ema_fast_values
    df[key_ema_slow] = ema_slow_values

    # compute golden cross / death cross
    previous_fast_series = df[key_ema_fast].shift(1)
    previous_slow_series = df[key_ema_slow].shift(1)
    golden_cross_values = []
    death_cross_values = []
    for i in (range(0, len(df.index))):
        golden_cross_values.append(
            (ema_fast_values[i] >= ema_slow_values[i]) & (previous_fast_series[i] <= previous_slow_series[i]))
        death_cross_values.append(
            (ema_fast_values[i] <= ema_slow_values[i]) & (previous_fast_series[i] >= previous_slow_series[i]))

    # add computed crossing results back to dataframe
    key_golden_cross = column_target_golden_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
    key_death_cross = column_target_death_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
    df[key_golden_cross] = golden_cross_values
    df[key_death_cross] = death_cross_values

    return df


def compute_ema(df, column_source, column_target_ema, time_periods):
    """
    Compute Exponential Moving Average (EMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to compute EMA (e.g. close price)
    :param column_target_ema: prefix of target column in dataframe for EMA results
    :param time_periods: list of time periods (number of days for EMA)
    :return: modified dataframe
    """
    # compute EMA for each time period and add results back to dataframe
    for time_period in time_periods:
        key_ema = column_target_ema + "-{:d}".format(time_period)
        ema_series = df[column_source].ewm(span=time_period, adjust=False).mean()
        df[key_ema] = ema_series

    return df


def compute_ma_cross(
        df, column_source, column_target_golden_cross, column_target_death_cross,
        time_period_fast, time_period_slow):
    """
    Compute Moving Average (Golden/Death) Crosses.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: prefix of source column in dataframe with moving average values
    :param column_target_golden_cross: name of target column in dataframe for golden cross results
    :param column_target_death_cross: name of target column in dataframe for death cross results
    :param time_period_fast: number of days over which to average for fast MA
    :param time_period_slow: number of days over which to average for slow MA
    :return: modified dataframe
    """
    # get moving average values
    key_ma_fast = column_source + "-{:d}".format(time_period_fast)
    key_ma_slow = column_source + "-{:d}".format(time_period_slow)
    fast_series = df[key_ma_fast]
    slow_series = df[key_ma_slow]

    # compute golden cross / death cross and add results back to dataframe
    previous_fast_series = df[key_ma_fast].shift(1)
    previous_slow_series = df[key_ma_slow].shift(1)
    key_golden_cross = column_target_golden_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
    key_death_cross = column_target_death_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
    df[key_golden_cross] = (fast_series >= slow_series) & (previous_fast_series <= previous_slow_series)
    df[key_death_cross] = (fast_series <= slow_series) & (previous_fast_series >= previous_slow_series)

    return df


def compute_bb_custom(df, column_source, column_target_bb, time_period, stdev_factor=2):
    """
    Compute Bollinger Bands (BB) With Simple Moving Average (SMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to compute SMA (e.g. close price)
    :param column_target_bb: prefix of target column in dataframe for BB results
    :param time_period: number of days over which to average
    :param stdev_factor: standard deviation scaling factor for upper and lower bands
    :return: modified dataframe
    """
    # compute BB
    history_values = []
    sma_values = []
    upper_band_values = []
    lower_band_values = []
    for value in df[column_source]:
        history_values.append(value)
        if len(history_values) > time_period:
            del (history_values[0])
        sma = stats.mean(history_values)
        sma_values.append(sma)

        variance = 0  # variance is the square of standard deviation
        for history_value in history_values:
            variance = variance + ((history_value - sma) ** 2)

        stdev = math.sqrt(variance / len(history_values))  # use sqrt to get standard deviation
        upper_band_values.append(sma + (stdev_factor * stdev))
        lower_band_values.append(sma - (stdev_factor * stdev))

    # add computed BB results back to dataframe
    key_sma = column_target_bb + "-sma-{:d}-{:d}".format(time_period, stdev_factor)
    key_upper_band = column_target_bb + "-upper-{:d}-{:d}".format(time_period, stdev_factor)
    key_lower_band = column_target_bb + "-lower-{:d}-{:d}".format(time_period, stdev_factor)
    df[key_sma] = sma_values
    df[key_upper_band] = upper_band_values
    df[key_lower_band] = lower_band_values

    return df


def compute_bb(df, column_source, column_target_bb, time_period, stdev_factor=2):
    """
    Compute Bollinger Bands (BB) With Simple Moving Average (SMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe  with values to compute SMA (e.g. close price)
    :param column_target_bb: prefix of target column in dataframe for BB results
    :param time_period: number of days over which to average
    :param stdev_factor: standard deviation scaling factor for upper and lower bands
    :return: modified dataframe
    """
    # compute BB and add results back to dataframe
    key_sma = column_target_bb + "-sma-{:d}-{:d}".format(time_period, stdev_factor)
    key_upper_band = column_target_bb + "-upper-{:d}-{:d}".format(time_period, stdev_factor)
    key_lower_band = column_target_bb + "-lower-{:d}-{:d}".format(time_period, stdev_factor)
    df[key_sma] = df[column_source].rolling(window=time_period, min_periods=1).mean()
    sma_stdev = df[column_source].rolling(window=time_period, min_periods=1).std(ddof=0)
    df[key_upper_band] = df[key_sma] + (sma_stdev * stdev_factor)
    df[key_lower_band] = df[key_sma] - (sma_stdev * stdev_factor)

    return df


def compute_macd_custom(
        df, column_source, column_target_ema,
        column_target_macd, column_target_macd_signal, column_target_macd_histogram,
        time_period_fast, time_period_slow, time_period_macd):
    """
    Compute Moving Average Convergence Divergence (MACD).

    When fast ema crosses above slow ema, it indicates a reversal from downtrend to uptrend.
    When fast ema crosses below slow ema, it indicates a reversal from uptrend to downtrend.

    When macd crosses above ema_macd (signal), it indicates a reversal from downtrend to uptrend.
    When macd crosses below ema_macd (signal), it indicates a reversal from uptrend to downtrend.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe  with values to compute MACD (e.g. close price)
    :param column_target_ema: prefix of target column in dataframe for EMA results
    :param column_target_macd: name of target column in dataframe for MACD results
    :param column_target_macd_signal: name of target column in dataframe for MACD signal results
    :param column_target_macd_histogram: name of target column in dataframe for MACD histogram results
    :param time_period_fast: number of days over which to average for fast EMA
    :param time_period_slow: number of days over which to average for slow EMA
    :param time_period_macd: number of days over which to average for MACD EMA
    :return: modified dataframe
    """
    k_fast = 2 / (time_period_fast + 1)  # fast EMA smoothing factor
    ema_fast = 0
    k_slow = 2 / (time_period_slow + 1)  # slow EMA smoothing factor
    ema_slow = 0
    k_macd = 2 / (time_period_macd + 1)  # MACD EMA smoothing factor
    ema_macd = 0
    ema_fast_values = []
    ema_slow_values = []
    macd_values = []
    macd_signal_values = []  # EMA of MACD values
    macd_histogram_values = []  # MACD - MACD-EMA
    for value in df[column_source]:
        # compute MACD
        if ema_fast == 0:  # first observation
            ema_fast = value
            ema_slow = value
        else:
            ema_fast = (value - ema_fast) * k_fast + ema_fast
            ema_slow = (value - ema_slow) * k_slow + ema_slow
        ema_fast_values.append(ema_fast)
        ema_slow_values.append(ema_slow)
        macd = ema_fast - ema_slow
        macd_values.append(macd)

        # compute MACD signal and histogram
        if ema_macd == 0:  # first observation
            ema_macd = macd
        else:
            ema_macd = (macd - ema_macd) * k_macd + ema_macd  # signal is EMA of MACD values
        macd_signal_values.append(ema_macd)
        macd_histogram_values.append(macd - ema_macd)

    # add computed results back to dataframe
    time_fast = str(time_period_fast)
    time_slow = str(time_period_slow)
    time_fast_slow_macd = time_fast + "-" + time_slow + "-" + str(time_period_macd)
    key_ema_fast = column_target_ema + "-" + time_fast
    key_ema_slow = column_target_ema + "-" + time_slow
    key_macd = column_target_macd + "-" + time_fast_slow_macd
    key_macd_signal = column_target_macd_signal + "-" + time_fast_slow_macd
    key_macd_histogram = column_target_macd_histogram + "-" + time_fast_slow_macd
    df[key_ema_fast] = ema_fast_values
    df[key_ema_slow] = ema_slow_values
    df[key_macd] = macd_values
    df[key_macd_signal] = macd_signal_values
    df[key_macd_histogram] = macd_histogram_values

    return df


def compute_macd(
        df, column_source, column_target_ema,
        column_target_macd, column_target_macd_signal, column_target_macd_histogram,
        time_period_fast, time_period_slow, time_period_macd):
    """
    Compute Moving Average Convergence Divergence (MACD).

    When fast ema crosses above slow ema, it indicates a reversal from downtrend to uptrend.
    When fast ema crosses below slow ema, it indicates a reversal from uptrend to downtrend.

    When macd crosses above ema_macd (signal), it indicates a reversal from downtrend to uptrend.
    When macd crosses below ema_macd (signal), it indicates a reversal from uptrend to downtrend.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe  with values to compute MACD (e.g. close price)
    :param column_target_ema: prefix of target column in dataframe for EMA results
    :param column_target_macd: name of target column in dataframe for MACD results
    :param column_target_macd_signal: name of target column in dataframe for MACD signal results
    :param column_target_macd_histogram: name of target column in dataframe for MACD histogram results
    :param time_period_fast: number of days over which to average for fast EMA
    :param time_period_slow: number of days over which to average for slow EMA
    :param time_period_macd: number of days over which to average for MACD EMA
    :return: modified dataframe
    """
    time_fast_slow_macd = "{:d}-{:d}-{:d}".format(time_period_fast, time_period_slow, time_period_macd)
    key_ema_fast = column_target_ema + "-{:d}".format(time_period_fast)
    key_ema_slow = column_target_ema + "-{:d}".format(time_period_slow)
    key_macd = column_target_macd + "-" + time_fast_slow_macd
    key_macd_signal = column_target_macd_signal + "-" + time_fast_slow_macd
    key_macd_histogram = column_target_macd_histogram + "-" + time_fast_slow_macd

    # compute EMA and add results back to dataframe
    df[key_ema_fast] = df[column_source].ewm(span=time_period_fast, adjust=False).mean()
    df[key_ema_slow] = df[column_source].ewm(span=time_period_slow, adjust=False).mean()

    # compute MACD and add results back to dataframe
    df[key_macd] = df[key_ema_fast] - df[key_ema_slow]
    df[key_macd_signal] = df[key_macd].ewm(span=time_period_macd, adjust=False).mean()
    df[key_macd_histogram] = df[key_macd] - df[key_macd_signal]

    return df


def compute_rsi(df, column_source, column_target_avg_gain, column_target_avg_loss, column_target_rsi, time_periods):
    """
    Compute Relative Strength Indicator (RSI).

    RSI values over 50% indicate an uptrend, while values below 50% indicate a downtrend.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe  with values to compute RSI (e.g. close price)
    :param column_target_avg_gain: name of target column in dataframe for average gain results
    :param column_target_avg_loss: name of target column in dataframe for average loss results
    :param column_target_rsi: name of target column in dataframe for RSI results
    :param time_periods: list ot time periods (in days) over which to look back to compute gains and losses
    :return: modified dataframe
    """

    # compute RSI over time period and add results back to dataframe
    for time_period in time_periods:
        gain_history_values = []  # history of gains over look back period (0 if no gain, magnitude of gain if gain)
        loss_history_values = []  # history of loss over look back period (0 if no loss, magnitude if loss)
        avg_gain_values = []
        avg_loss_values = []
        rsi_values = []
        last_value = 0  # current_value - last_value > 0 ==> gain; current_value - last_value < 0 ==> loss
        for value in df[column_source]:
            if last_value == 0:  # first observation
                last_value = value

            # compute average gain and loss
            gain_history_values.append(max(0, value - last_value))
            loss_history_values.append(max(0, last_value - value))
            last_value = value
            if len(gain_history_values) > time_period:  # maximum observations is equal to look back period
                del (gain_history_values[0])
                del (loss_history_values[0])
            avg_gain = stats.mean(gain_history_values)  # average gain over look back period
            avg_loss = stats.mean(loss_history_values)  # average loss over look back period
            avg_gain_values.append(avg_gain)
            avg_loss_values.append(avg_loss)

            # compute RS and RSI
            rs = 0
            if avg_loss > 0:  # to avoid division by 0
                rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

        # add computed results back to dataframe
        key_avg_gain = column_target_avg_gain + "-{:d}".format(time_period)
        key_avg_loss = column_target_avg_loss + "-{:d}".format(time_period)
        key_rsi = column_target_rsi + "-{:d}".format(time_period)
        df[key_avg_gain] = avg_gain_values
        df[key_avg_loss] = avg_loss_values
        df[key_rsi] = rsi_values

    return df


def compute_change(df, column_source, column_target_change, column_target_change_pc, time_periods):
    """
    Compute the change and percentage change of the values in the source column for the specified period in (trading)
    days.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to compute change (e.g. close price)
    :param column_target_change: name of target column in dataframe for change to add to dataframe
    :param column_target_change_pc: name of target column in dataframe for change pc to add to dataframe
    :param time_periods: list of time periods in (trading) days
    :return: modified dataframe
    """

    # compute change over time period and add result back to dataframe
    for time_period in time_periods:
        key_change = column_target_change + "-{:d}".format(time_period)
        key_change_pc = column_target_change_pc + "-{:d}".format(time_period)
        #df2 = df[column_source].asfreq("D", method="ffill")
        #change_series = df2.diff(time_period)
        #change_pc_series = df2.pct_change(time_period)
        #df[key_change] = change_series
        #df[key_change_pc] = change_pc_series
        change_series = df[column_source].diff(time_period)
        change_pc_series = df[column_source].pct_change(time_period)
        df[key_change] = change_series
        df[key_change_pc] = change_pc_series

    return df


def compute_daily_change_between_current_and_previous(
        df, column_source_current, column_source_previous,
        column_target_daily_change, column_target_daily_change_pc):
    """
    Compute the daily change and daily percentage change between the current and previous values.

    :param df: dataframe (sorted in ascending time order)
    :param column_source_current: name of source column in dataframe with current values to compute (e.g. current open price)
    :param column_source_previous: name of source column in dataframe with previous values to compute (e.g. previous close price)
    :param column_target_daily_change: name of target column in dataframe for daily change to add to dataframe
    :param column_target_daily_change_pc: name of target column in dataframe for daily change pc to add to dataframe
    :return: modified dataframe
    """

    # NOT CORRECT?
    daily_change = df[column_source_current] - df[column_source_previous].shift(1)
    # daily_change_pc = daily_change.pct_change(1)
    daily_change_pc = df[column_source_previous].pct_change(1)

    # add computed results back to dataframe
    df = pd.concat([df, daily_change.rename(column_target_daily_change)], axis=1)
    df = pd.concat([df, daily_change_pc.rename(column_target_daily_change_pc)], axis=1)

    return df


def compute_52_week_range(df, column_source_low, column_source_high, column_target_low, column_target_high):
    """
    Compute 52 Week Range (Low~High).

    :param df: dataframe (sorted in ascending time order)
    :param column_source_low: name of source column in dataframe with low values to compute
    :param column_source_high: name of source column in dataframe with high values to compute
    :param column_target_low: name of target column in dataframe for low range results to add to dataframe
    :param column_target_high: name of target column in dataframe for high range results to add to dataframe
    :return: modified dataframe
    """

    # compute rolling 52 week range and add result back to dataframe
    df[column_target_low] = df[column_source_low].asfreq("D").rolling(window=52*7, min_periods=1).min();
    df[column_target_high] = df[column_source_high].asfreq("D").rolling(window=52*7, min_periods=1).max();

    return df


def compute_change_pc_above(df, column_source1, column_source2, column_target, column_target_pc):
    """
    Compute the percentage of source1 above source2 (e.g. close price above the 52 week low price).

    :param df: dataframe (sorted in ascending time order)
    :param column_source1: name of source1 column in dataframe with values to compute (e.g. close price)
    :param column_source2: name of source2 column in dataframe with values to compute (e.g. 52 week low price)
    :param column_target: name of target column in dataframe for change results to add to dataframe
    :param column_target_pc: name of target column in dataframe for percentage change results to add to dataframe
    :return: modified dataframe
    """

    change_above = df[column_source1] - df[column_source2]
    pc_above = (df[column_source1] / df[column_source2]) - 1

    # add computed results back to dataframe
    df = pd.concat([df, change_above.rename(column_target)], axis=1)
    df = pd.concat([df, pc_above.rename(column_target_pc)], axis=1)

    return df


def compute_change_pc_below(df, column_source1, column_source2, column_target, column_target_pc):
    """
    Compute the percentage of source1 below source2 (e.g. close price below the 52 week high price).

    :param df: dataframe (sorted in ascending time order)
    :param column_source1: name of source1 column in dataframe with values to compute (e.g. close price)
    :param column_source2: name of source2 column in dataframe with values to compute (e.g. 52 week high price)
    :param column_target: name of target column in dataframe for change results to add to dataframe
    :param column_target_pc: name of target column in dataframe for percentage change results to add to dataframe
    :return: modified dataframe
    """

    change_below = df[column_source2] - df[column_source1]
    pc_below = 1 - (df[column_source1] / df[column_source2])

    # add computed results back to dataframe
    df = pd.concat([df, change_below.rename(column_target)], axis=1)
    df = pd.concat([df, pc_below.rename(column_target_pc)], axis=1)

    return df


def compute_sharpe(df, column_source, column_target, N=252):
    """
    Compute the sharpe ratio of the (e.g. daily) return values in the source column.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column in dataframe with values to compute sharpe ratio (e.g. daily returns)
    :param column_target: name of target column in dataframe for sharpe ratio to add to dataframe
    :param N: number of trading periods (e.g. 252 = daily, 12 = monthly)
    :return: modified dataframe
    """

    # compute the sharpe ratio and add result back to dataframe
    return_series = df[column_source]
    df[column_target] = np.sqrt(N) * return_series.mean() / return_series.std()
    return df


def compute_cumulative_total_return(df, column_price):
    """
    Cumulative return on an investment is the aggregate amount that the investment has gained or lost over time,
    independent of the amount of time involved.

    cumulative total return = (price_end - price_start) / price_start = (price_end/price_start) - 1

    :param df: dataframe (sorted in ascending time order)
    :param column_price: name of source column in dataframe with price values (adjusted for splits and dividends) to
                         compute cumulative total return
    :return: cumulative total return
    """

    # compute cumulative total return
    price_start = df[column_price][0]
    price_end = df[column_price][-1]
    cumulative_return = (price_end - price_start)/price_start
    return cumulative_return


def compute_annualized_total_return_over_years(df, column_price, years):
    """
    Computed the annualized total return over the specified number of years.

    This is equivalent to Compound Annual Growth Rate (CAGR).

    Note: If the period is less than one year, it is best not to use annualized total return as it could result in a
    very large (positive or negative) number that is not meaningful.

    :param df: dataframe (sorted in ascending time order)
    :param column_price: name of source column in dataframe with price values (adjusted for splits and dividends) to
                         compute annualized total return
    :param years: time period in years (e.g. 1 = 1 year, 2 = 2 years, 2.5 = 1 year and 6 months, etc.)
    :return: annualized total return over years
    """

    # compute cumulative total return
    total_return = compute_cumulative_total_return(df, column_price)

    # compute annualized total returns over months
    annualized_total_return = ((1 + total_return)**(1/years)) - 1

    return annualized_total_return


def compute_annualized_total_return_over_months(df, column_price, months):
    """
    Computed the annualized total return over the specified number of months.

    This is equivalent to Compound Annual Growth Rate (CAGR).

    Note: If the period is less than one year, it is best not to use annualized total return as it could result in a
    very large (positive or negative) number that is not meaningful.

    :param df: dataframe (sorted in ascending time order)
    :param column_price: name of source column in dataframe with price values (adjusted for splits and dividends) to
                         compute annualized total return
    :param months: time period in months (e.g. 1 = 1 month, 2 = 2 months, 2.5 = 1 month and ~15 days, etc.)
    :return: annualized total return over months
    """

    # calculate cumulative total return
    total_return = compute_cumulative_total_return(df, column_price)

    # calculate annualized total returns over months
    annualized_total_return = ((1 + total_return)**(12/months)) - 1

    return annualized_total_return


def compute_annualized_total_return_over_calendar_days(df, column_price):
    """
    Computed the annualized total return over the provided number of calendar days.

    This is equivalent to Compound Annual Growth Rate (CAGR).

    Note: Using days (versus years or months) provides the most precise form of annualized return calculation.

    Note: If the period is less than one year, it is best not to use annualized total return as it could result in a
    very large (positive or negative) number that is not meaningful.

    :param df: dataframe (sorted in ascending time order)
    :param column_price: name of source column in dataframe with price values (adjusted for splits and dividends) to
                         compute annualized total return
    :return: annualized total return over days
    """

    # calculate cumulative total return
    total_return = compute_cumulative_total_return(df, column_price)

    # fill in missing calendar days
    index_filled = pd.date_range(min(df.index), max(df.index))
    df_filled = df.reindex(index_filled, method="ffill")

    # number of calendar days in data
    # note: dataframe includes one day before the desired range; for example, if we want to get the annualized total
    # return from 4/1/2000 to 3/31/2002, the dataframe will contain data from 3/31/2000 to 3/31/2002; as a result,
    # the number of calendar days is (len(df) - 1)
    calendar_days = len(df_filled) - 1

    # calculate annualized total returns over days
    annualized_total_return = ((1 + total_return)**(c.CALENDAR_DAYS/calendar_days)) - 1

    return annualized_total_return


def compute_annualized_total_return_over_trading_days(df, column_price):
    """
    Computed the (trailing) annualized total return over the provided number of trading days.

    This is equivalent to Compound Annual Growth Rate (CAGR).

    Note: Using days (versus years or months) provides the most precise form of annualized return calculation.

    Note: If the period is less than one year, it is best not to use annualized total return as it could result in a
    very large (positive or negative) number that is not meaningful.

    :param df: dataframe (sorted in ascending time order)
    :param column_price: name of source column in dataframe with price values (adjusted for splits and dividends) to
                         compute annualized total return
    :return: annualized total return over days
    """

    # calculate cumulative total return
    total_return = compute_cumulative_total_return(df, column_price)

    # number of trading days in data
    # note: dataframe includes one day before the desired range; for example, if we want to get the annualized total
    # return from 4/1/2000 to 3/31/2002, the dataframe will contain data from 3/31/2000 to 3/31/2002; as a result,
    # the number of trading days is (len(df) - 1)
    trading_days = len(df) - 1

    # calculate annualized total returns over number of trading days
    annualized_total_return = ((1 + total_return)**(c.TRADING_DAYS_YEAR/trading_days)) - 1

    return annualized_total_return


def compute_trailing_returns(df, symbol):

    dict_returns = {"symbol": symbol}

    # compute total return (trailing 1 month)
    end = df.index[-1]
    start = end - pd.DateOffset(months=1)
    dict_returns[c.TRAILING_RETURN + "-1m"] = compute_cumulative_total_return(df.loc[start:end], c.CLOSE)

    # compute total return (trailing 3 month)
    end = df.index[-1]
    start = end - pd.DateOffset(months=3)
    dict_returns[c.TRAILING_RETURN + "-3m"] = compute_cumulative_total_return(df.loc[start:end], c.CLOSE)

    # compute total return for YTD
    end = df.index[-1]
    start = end - pd.offsets.YearBegin() - pd.offsets.Day(1)
    dict_returns[c.TRAILING_RETURN + "-ytd"] = compute_cumulative_total_return(df.loc[start:end], c.CLOSE)

    # compute annualized total returns (trailing 1 year)
    end = df.index[-1]
    start = end - pd.DateOffset(years=1)
    dict_returns[c.TRAILING_RETURN + "-1y"] = compute_annualized_total_return_over_trading_days(df.loc[start:end], c.CLOSE)

    # compute annualized total returns (trailing 3 years)
    end = df.index[-1]
    start = end - pd.DateOffset(years=3)
    dict_returns[c.TRAILING_RETURN + "-3y"] = compute_annualized_total_return_over_trading_days(df.loc[start:end], c.CLOSE)

    # compute annualized total returns (trailing 5 years)
    end = df.index[-1]
    start = end - pd.DateOffset(years=5)
    dict_returns[c.TRAILING_RETURN + "-5y"] = compute_annualized_total_return_over_trading_days(df.loc[start:end], c.CLOSE)

    # compute annualized total returns (trailing 10 years)
    end = df.index[-1]
    start = end - pd.DateOffset(years=10)
    dict_returns[c.TRAILING_RETURN + "-10y"] = compute_annualized_total_return_over_trading_days(df.loc[start:end], c.CLOSE)

    # add computed returns to dataframe
    df_returns = pd.DataFrame()
    df_returns = df_returns.append(dict_returns, ignore_index=True)

    return df_returns


def compute_yearly_total_returns(df, symbol):
    # compute cumulative total return
    dict_returns = {"symbol": symbol}

    # compute annualized total returns for past 20 years, starting with previous year
    years_available = np.unique(df.index.year)
    if years_available.size > 1:
        years_previous_end = years_available[-22:-1]
        years_previous_start = years_previous_end[:-1]
        year_end = years_previous_end[-1]
        for year_start in reversed(years_previous_start):
            start = "{:d}-12-31".format(year_start)
            end = "{:d}-12-31".format(year_end)
            dict_returns[c.ANNUALIZED_RETURN + "-{:d}".format(year_end)] = compute_annualized_total_return_over_trading_days(
                df.loc[start:end], c.CLOSE)
            year_end = year_start

    # add computed returns to dataframe
    df_returns = pd.DataFrame()
    df_returns = df_returns.append(dict_returns, ignore_index=True)

    return df_returns


def compute_max_drawdowns(df, column_source, column_target):
    daily_pc_series = df[column_source].pct_change()

    # compute wealth index starting with $1000
    wealth_index_series = 1000 * (1 + daily_pc_series).cumprod()

    # compute previous peaks
    previous_peaks_series = wealth_index_series.cummax()

    # compute drawdowns
    drawdowns_series = (wealth_index_series - previous_peaks_series)/previous_peaks_series

    # add computed results back to dataframe
    df = pd.concat([df, drawdowns_series.rename(column_target)], axis=1)

    return df


def compute_max_drawdown_by_year(df, symbol):

    # get previously calculated max drawdowns
    mdd_series = df[c.MAX_DD]

    # find all-time max drawdown
    dict_mdd = {"symbol": symbol, c.MAX_DD + "-max": mdd_series.min()}

    # compute YTD max drawdown
    end = df.index[-1]
    start = end - pd.offsets.YearBegin()
    dict_mdd[c.MAX_DD + "-ytd"] = mdd_series.loc[start:end].min()

    # find max drawdown for past 20 years, starting with previous year
    years_available = np.unique(df.index.year)
    years_previous = years_available[-21:-1]
    for year in reversed(years_previous):
        start = "{:d}-01-01".format(year)
        end = "{:d}-12-31".format(year)
        dict_mdd[c.MAX_DD + "-{:d}".format(year)] = mdd_series.loc[start:end].min()

    # add computed returns to dataframe
    df_mdd_by_year = pd.DataFrame()
    df_mdd_by_year = df_mdd_by_year.append(dict_mdd, ignore_index=True)

    return df_mdd_by_year


def compute_all_ta(df):
    # add previous close
    df = copy_column_shift(df, c.CLOSE, c.PREV_CLOSE, 1)
    # compute change and percentage change of closing price
    df = compute_change(df, c.CLOSE, c.CHANGE, c.CHANGE_PC, (1,))
    # compute daily change between open price and previous closing price
    df = compute_daily_change_between_current_and_previous(df, c.OPEN, c.CLOSE, c.OPEN_PREV_CLOSE, c.OPEN_PREV_CLOSE_PC)
    # compute 52 week range (low~high)
    df = compute_52_week_range(df, c.LOW, c.HIGH, c.R52_WK_LOW, c.R52_WK_HIGH)
    # compute change and percentage change of close price above the 52 week low price
    df = compute_change_pc_above(df, c.CLOSE, c.R52_WK_LOW, c.CLOSE_ABOVE_52_WK_LOW, c.CLOSE_ABOVE_52_WK_LOW_PC)
    # compute change and percentage change of close price below the 52 week high price
    df = compute_change_pc_below(df, c.CLOSE, c.R52_WK_HIGH, c.CLOSE_BELOW_52_WK_HIGH, c.CLOSE_BELOW_52_WK_HIGH_PC)
    # compute SMA of close price
    df = compute_sma(df, c.CLOSE, c.SMA, (50, 100, 200))
    # compute EMA of close price
    df = compute_ema(df, c.CLOSE, c.EMA, (12, 26, 50, 200))
    # compute golden/death crosses for SMA
    df = compute_ma_cross(df, c.SMA, c.SMA_GOLDEN_CROSS, c.SMA_DEATH_CROSS, 50, 200)
    # compute golden/death crosses for EMA
    df = compute_ma_cross(df, c.EMA, c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, 12, 26)
    df = compute_ma_cross(df, c.EMA, c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, 50, 200)
    # compute average daily trading volume
    df = compute_sma(df, c.VOLUME, c.ADTV, (30,))
    # compute BB of close price with SMA period of 20 and standard deviation of 2
    df = compute_bb(df, c.CLOSE, c.BB, 20, 2)
    # compute MACD of close price
    df = compute_macd(df, c.CLOSE, c.EMA, c.MACD, c.MACD_SIGNAL, c.MACD_HISTOGRAM, 12, 26, 9)
    # compute RSI of close price
    df = compute_rsi(df, c.CLOSE, c.RSI_AVG_GAIN, c.RSI_AVG_LOSS, c.RSI, (7, 14, 21))
    # compute sharpe ratio
    df = compute_sharpe(df, "u_change_pc-1", c.SHARPE)
    # compute max drawdowns
    df = compute_max_drawdowns(df, c.CLOSE, c.MAX_DD)

    return df


def compute_all_perf(df, symbol_name):
    # compute trailing (annualized) returns
    df_trailing_returns = compute_trailing_returns(df, symbol_name)

    # compute yearly total returns
    df_yearly_total_returns = compute_yearly_total_returns(df, symbol_name)

    # compute max drawdown by year
    df_mdd = compute_max_drawdown_by_year(df, symbol_name)

    # merge performance data
    df_perf = pd.merge(df_trailing_returns, df_yearly_total_returns, on="symbol")
    df_perf = pd.merge(df_perf, df_mdd, on="symbol")

    return df_perf
