"""
Technical Analysis Library

Library of functions to compute various technical indicators.

@author: eyu
"""

import logging
import pandas as pd
import math as math
import statistics as stats

# create logger
logger = logging.getLogger("algo-trader")


def copy_column(df, column_source, column_target):
    """
    Copy an existing column to a new column in dataframe.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to copy
    :param column_target: name of target column in dataframe for copied values
    :return: modified dataframe
    """
    df[column_target] = df[column_source]
    return df


def compute_sma_custom(df, column_source, column_target_sma, time_period):
    """
    Compute Simple Moving Average (SMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to compute SMA (e.g. close price)
    :param column_target_sma: prefix of target column in dataframe for SMA results
    :param time_period: number of days over which to average
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


def compute_sma(df, column_source, column_target_sma, time_period):
    """
    Compute Simple Moving Average (SMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to compute SMA (e.g. close price)
    :param column_target_sma: prefix of target column in dataframe for SMA results
    :param time_period: number of days over which to average
    :return: modified dataframe
    """
    # compute SMA and add results back to dataframe
    key_sma = column_target_sma + "-{:d}".format(time_period)
    df[key_sma] = df[column_source].rolling(window=time_period).mean()

    return df


def compute_ema_custom(
        df, column_source, column_target_ema, column_target_golden_cross, column_target_death_cross,
        time_period_fast, time_period_slow):
    """
    Compute Exponential Moving Average (EMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to compute EMA (e.g. close price)
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


def compute_ema(
        df, column_source, column_target_ema, column_target_golden_cross, column_target_death_cross,
        time_period_fast, time_period_slow):
    """
    Compute Exponential Moving Average (EMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to compute EMA (e.g. close price)
    :param column_target_ema: prefix of target column in dataframe for EMA results
    :param column_target_golden_cross: name of target column in dataframe for golden cross results
    :param column_target_death_cross: name of target column in dataframe for death cross results
    :param time_period_fast: number of days over which to average for fast EMA
    :param time_period_slow: number of days over which to average for slow EMA
    :return: modified dataframe
    """
    # compute EMA and add results back to dataframe
    key_ema_fast = column_target_ema + "-{:d}".format(time_period_fast)
    key_ema_slow = column_target_ema + "-{:d}".format(time_period_slow)
    ema_fast_series = df[column_source].ewm(span=time_period_fast, adjust=False).mean()
    ema_slow_series = df[column_source].ewm(span=time_period_slow, adjust=False).mean()
    df[key_ema_fast] = ema_fast_series
    df[key_ema_slow] = ema_slow_series

    # compute golden cross / death cross and add results back to dataframe
    previous_fast_series = df[key_ema_fast].shift(1)
    previous_slow_series = df[key_ema_slow].shift(1)
    key_golden_cross = column_target_golden_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
    key_death_cross = column_target_death_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
    df[key_golden_cross] = (ema_fast_series >= ema_slow_series) & (previous_fast_series <= previous_slow_series)
    df[key_death_cross] = (ema_fast_series <= ema_slow_series) & (previous_fast_series >= previous_slow_series)

    return df


def compute_bb_custom(df, column_source, column_target_bb, time_period, stdev_factor=2):
    """
    Compute Bollinger Bands (BB) With Simple Moving Average (SMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to compute SMA (e.g. close price)
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
        upper_band_values.append(sma + stdev_factor * stdev)
        lower_band_values.append(sma - stdev_factor * stdev)

    # add computed BB results back to dataframe
    key_sma = column_target_bb + "-sma-{:d}".format(time_period)
    key_upper_band = column_target_bb + "-upper-{:d}".format(time_period)
    key_lower_band = column_target_bb + "-lower-{:d}".format(time_period)
    df[key_sma] = sma_values
    df[key_upper_band] = upper_band_values
    df[key_lower_band] = lower_band_values

    return df


def compute_bb(df, column_source, column_target_bb, time_period, stdev_factor=2):
    """
    Compute Bollinger Bands (BB) With Simple Moving Average (SMA).

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to compute SMA (e.g. close price)
    :param column_target_bb: prefix of target column in dataframe for BB results
    :param time_period: number of days over which to average
    :param stdev_factor: standard deviation scaling factor for upper and lower bands
    :return: modified dataframe
    """
    # compute BB and add results back to dataframe
    key_sma = column_target_bb + "-sma-{:d}".format(time_period)
    key_upper_band = column_target_bb + "-upper-{:d}".format(time_period)
    key_lower_band = column_target_bb + "-lower-{:d}".format(time_period)
    df[key_sma] = df[column_source].rolling(window=time_period).mean()
    sma_stdev = df[column_source].rolling(window=time_period).std(ddof=0)
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
    :param column_source: name of source column with values to compute MACD (e.g. close price)
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
    :param column_source: name of source column with values to compute MACD (e.g. close price)
    :param column_target_ema: prefix of target column in dataframe for EMA results
    :param column_target_macd: name of target column in dataframe for MACD results
    :param column_target_macd_signal: name of target column in dataframe for MACD signal results
    :param column_target_macd_histogram: name of target column in dataframe for MACD histogram results
    :param time_period_fast: number of days over which to average for fast EMA
    :param time_period_slow: number of days over which to average for slow EMA
    :param time_period_macd: number of days over which to average for MACD EMA
    :return: modified dataframe
    """
    time_fast = str(time_period_fast)
    time_slow = str(time_period_slow)
    time_fast_slow_macd = time_fast + "-" + time_slow + "-" + str(time_period_macd)
    key_ema_fast = column_target_ema + "-" + time_fast
    key_ema_slow = column_target_ema + "-" + time_slow
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


def compute_rsi(df, column_source, column_target_avg_gain, column_target_avg_loss, column_target_rsi, time_period):
    """
    Compute Relative Strength Indicator (RSI).

    RSI values over 50% indicate an uptrend, while values below 50% indicate a downtrend.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to compute RSI (e.g. close price)
    :param column_target_avg_gain: name of target column in dataframe for average gain results
    :param column_target_avg_loss: name of target column in dataframe for average loss results
    :param column_target_rsi: name of target column in dataframe for RSI results
    :param time_period: number of days over which to look back to compute gains and losses
    :return: modified dataframe
    """
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

        # compute RSI
        rs = 0
        if avg_loss > 0:  # to avoid division by 0
            rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    # add computed results back to dataframe
    key_avg_gain = column_target_avg_gain + "-" + str(time_period)
    key_avg_loss = column_target_avg_loss + "-" + str(time_period)
    key_rsi = column_target_rsi + "-" + str(time_period)
    df[key_avg_gain] = avg_gain_values
    df[key_avg_loss] = avg_loss_values
    df[key_rsi] = rsi_values

    return df


def compute_daily_change(df, column_source, column_target_daily_change, column_target_daily_change_pc):
    """
    Compute the daily change and daily percentage change of the values in the source column.

    :param df: dataframe (sorted in ascending time order)
    :param column_source: name of source column with values to compute daily change (e.g. close price)
    :param column_target_daily_change: name of target column for daily change to add to dataframe
    :param column_target_daily_change_pc: name of target column for daily change pc to add to dataframe
    :return: modified dataframe
    """

    # NOT CORRECT?
    daily_change = df[column_source].diff(1)
    # daily_change_pc = daily_change.pct_change(1)
    daily_change_pc = df[column_source].pct_change(1)

    # add computed results back to dataframe
    df = pd.concat([df, daily_change.rename(column_target_daily_change)], axis=1)
    df = pd.concat([df, daily_change_pc.rename(column_target_daily_change_pc)], axis=1)

    return df


def compute_daily_change_between_current_and_previous(
        df, column_source_current, column_source_previous,
        column_target_daily_change, column_target_daily_change_pc):
    """
    Compute the daily change and daily percentage change between the current and previous values.

    :param df: dataframe (sorted in ascending time order)
    :param column_source_current: name of source column with current values to compute (e.g. current open price)
    :param column_source_previous: name of source column with previous values to compute (e.g. previous close price)
    :param column_target_daily_change: name of target column for daily change to add to dataframe
    :param column_target_daily_change_pc: name of target column for daily change pc to add to dataframe
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
    :param column_source_low: name of source column with low values to compute
    :param column_source_high: name of source column with high values to compute
    :param column_target_low: name of target column for low range results to add to dataframe
    :param column_target_high: name of target column for high range results to add to dataframe
    :return: modified dataframe
    """

    # (for reference) alternative calculation method
    # symbol_high = symbol_df["High"].asfreq('D').rolling(window=52*7, min_periods=1).max();
    # symbol_low = symbol_df["Low"].asfreq('D').rolling(window=52*7, min_periods=1).min();

    time_period = 252  # approximate number of trading days in a year

    low_history = []
    high_history = []
    low_values = []
    high_values = []
    for low_value in df[column_source_low]:
        low_history.append(low_value)
        if len(low_history) > time_period:
            del (low_history[0])
        low_values.append(min(low_history))
    for high_value in df[column_source_high]:
        high_history.append(high_value)
        if len(high_history) > time_period:
            del (high_history[0])
        high_values.append(max(high_history))

    # add computed results back to dataframe
    df = pd.concat([df, pd.Series(low_values, index=df.index).rename(column_target_low)], axis=1)
    df = pd.concat([df, pd.Series(high_values, index=df.index).rename(column_target_high)], axis=1)

    return df


def compute_pc_above(df, column_source1, column_source2, column_target):
    """
    Compute the percentage of source1 above source2 (e.g. close price above the 52 week low price).

    :param df: dataframe (sorted in ascending time order)
    :param column_source1: name of source1 column with values to compute (e.g. close price)
    :param column_source2: name of source2 column with values to compute (e.g. 52 week low price)
    :param column_target: name of target column for results to add to dataframe
    :return: modified dataframe
    """

    # symbol_df['u_close_above_52-wk-low'] = (symbol_df['u_close'] / symbol_df['u_52_wk_low']) - 1
    # return symbol_df
    # return (close_data / close_52_wk_low_data) - 1
    pc_above = (df[column_source1] / df[column_source2]) - 1

    # add computed results back to dataframe
    df = pd.concat([df, pc_above.rename(column_target)], axis=1)

    return df


def compute_pc_below(df, column_source1, column_source2, column_target):
    """
    Compute the percentage of source1 below source2 (e.g. close price below the 52 week high price).

    :param df: dataframe (sorted in ascending time order)
    :param column_source1: name of source1 column with values to compute (e.g. close price)
    :param column_source2: name of source2 column with values to compute (e.g. 52 week high price)
    :param column_target: name of target column for results to add to dataframe
    :return: modified dataframe
    """

    # symbol_df['u_close_below_52-wk-high'] = 1 - (symbol_df['u_close'] / symbol_df['u_52_wk_high'])
    # return symbol_df
    # return 1 - (close_data / close_52_wk_high_data)

    pc_below = 1 - (df[column_source1] / df[column_source2])

    # add computed results back to dataframe
    df = pd.concat([df, pc_below.rename(column_target)], axis=1)

    return df
