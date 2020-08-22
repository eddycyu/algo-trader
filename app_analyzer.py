"""
Analyzer

@author: eyu
"""

import os
import logging
import click
import pandas as pd
from datetime import date

import constants as c
from data_reader_stooq import StooqDataReader
from data_reader_yahoo import YahooDataReader
import talib as talib
from taplot import TAPlot

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

# column constants
CLOSE = "u_close"
OPEN = "Open"
LOW = "Low"
HIGH = "High"
VOLUME = "Volume"
DAILY_CHG = "u_daily_change"
DAILY_CHG_PC = "u_daily_change_pc"
OPEN_PREV_CLOSE = "u_open_prev_close"
OPEN_PREV_CLOSE_PC = "u_open_prev_close_pc"
R52_WK_LOW = "u_52_wk_low"
R52_WK_HIGH = "u_52_wk_high"
CLOSE_ABOVE_52_WK_LOW = "u_close_above_52_wk_low"
CLOSE_BELOW_52_WK_HIGH = "u_close_below_52_wk_high"
SMA = "u_sma"
EMA_FAST = "u_ema_fast"
EMA_SLOW = "u_ema_slow"
EMA_GOLDEN_CROSS = "u_ema_golden"
EMA_DEATH_CROSS = "u_ema_death"
BB = "u_bb"
MACD_EMA_FAST = "u_macd_ema_fast"
MACD_EMA_SLOW = "u_macd_ema_slow"
MACD = "u_macd"
MACD_SIGNAL = "u_macd_signal"
MACD_HISTOGRAM = "u_macd_histogram"
RSI_AVG_GAIN = "u_rs_avg_gain"
RSI_AVG_LOSS = "u_rs_avg_loss"
RSI = "u_rsi"


def compute_indices(symbols, start_date, end_date):
    """
    ^SPX = S&P500 = cap-weighted index of the 500 largest U.S. publicly traded companies
    ^TWSE = cap-weighted index of all listed common shares traded on the Taiwan Stock Exchange
    ^KOSPI = cap-weighted index of all listed common shares traded on the Korea Exchange
    ^NKX = price-weighted index of top 225 blue-chip companies traded on the Tokyo Stock Exchange
    ^HSI = cap-weighted index of the largest companies on the Hong Kong Exchange
    ^STI = cap-weighted index of top 30 companies on the Singapore Exchange
    ^SHC = cap-weighted index of all stocks (A-shares and B-shares) traded on the Shanghai Stock Exchange
    ^SHBS = cap-weighted index of all B-shares traded on the Shanghai Stock Exchange

    :param symbols: list of index symbols to fetch, analyze and plot
    :param start_date: earliest date to fetch
    :param end_date:: latest date to fetch
    :return:
    """
    data_reader = StooqDataReader()
    plotter = TAPlot(c.CHART_TA_DIR)
    for symbol in symbols:
        symbol_name = symbol[1:]  # strip off the ^ character
        # load data
        df = data_reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Close' (no adjusted close for indices) as our close price
        df = talib.copy_column(df, "Close", CLOSE)
        # calculate and plot
        compute(symbol_name, df, plotter)


def compute_equities(symbols, start_date, end_date):
    """
    :param symbols: list of equity symbols to fetch, analyze and plot
    :param start_date: earliest date to fetch
    :param end_date:: latest date to fetch
    :return:
    """
    data_reader = YahooDataReader()
    plotter = TAPlot(c.CHART_TA_DIR)
    for symbol in symbols:
        symbol_name = symbol
        # load data
        df = data_reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Adj Close' as our close price
        df = talib.copy_column(df, "Adj Close", CLOSE)
        # calculate and plot
        compute(symbol_name, df, plotter)


def compute(symbol_name, df, plotter):
    # calculate daily change and daily percentage change of closing price
    df = talib.compute_daily_change(df, CLOSE, DAILY_CHG, DAILY_CHG_PC)
    # calculate daily change between open price and previous closing price
    df = talib.compute_daily_change_between_current_and_previous(
        df, OPEN, CLOSE,
        OPEN_PREV_CLOSE, OPEN_PREV_CLOSE_PC)
    # calculate 52 week range (low~high)
    df = talib.compute_52_week_range(df, LOW, HIGH, R52_WK_LOW, R52_WK_HIGH)
    # calculate percentage change of close price above the 52 week low price
    df = talib.compute_pc_above(df, CLOSE, R52_WK_LOW, CLOSE_ABOVE_52_WK_LOW)
    # calculate percentage change of close price below the 52 week high price
    df = talib.compute_pc_below(df, CLOSE, R52_WK_HIGH, CLOSE_BELOW_52_WK_HIGH)
    # compute SMA of close price
    df = talib.compute_sma(df, CLOSE, SMA, 20)
    # compute EMA of close price
    df = talib.compute_ema(df, CLOSE, EMA_FAST, EMA_SLOW, EMA_GOLDEN_CROSS, EMA_DEATH_CROSS, 12, 26)  # short term
    df = talib.compute_ema(df, CLOSE, EMA_FAST, EMA_SLOW, EMA_GOLDEN_CROSS, EMA_DEATH_CROSS, 50, 200)  # long term
    # compute BB of close price with SMA period of 20 and standard deviation of 2
    df = talib.compute_bb(df, CLOSE, BB, 20, 2)
    # compute MACD of close price
    df = talib.compute_macd(
        df, CLOSE,
        MACD_EMA_FAST, MACD_EMA_SLOW,
        MACD, MACD_SIGNAL, MACD_HISTOGRAM,
        12, 26, 9)
    # compute RSI of close price
    df = talib.compute_rsi(df, CLOSE, RSI_AVG_GAIN, RSI_AVG_LOSS, RSI, 14)

    # plot charts
    plotter.plot_change(df.tail(252), CLOSE, DAILY_CHG, DAILY_CHG_PC, symbol_name)
    plotter.plot_change_between_current_and_previous(
        df.tail(252), OPEN, CLOSE,
        OPEN_PREV_CLOSE, OPEN_PREV_CLOSE_PC,
        symbol_name)
    plotter.plot_pc_above(df.tail(252), CLOSE, R52_WK_LOW, CLOSE_ABOVE_52_WK_LOW, symbol_name)
    plotter.plot_pc_below(df.tail(252), CLOSE, R52_WK_HIGH, CLOSE_BELOW_52_WK_HIGH, symbol_name)
    plotter.plot_sma(df.tail(252), CLOSE, SMA, VOLUME, symbol_name, 20)
    plotter.plot_ema(
        df.tail(252), CLOSE, EMA_FAST, EMA_SLOW,
        EMA_GOLDEN_CROSS, EMA_DEATH_CROSS, VOLUME,
        symbol_name, 12, 26)
    plotter.plot_ema(
        df.tail(252), CLOSE, EMA_FAST, EMA_SLOW,
        EMA_GOLDEN_CROSS, EMA_DEATH_CROSS, VOLUME,
        symbol_name, 50, 200)
    plotter.plot_bb(df.tail(252), CLOSE, BB, VOLUME, symbol_name, 20, 2)
    plotter.plot_macd(
        df.tail(120), CLOSE,
        MACD_EMA_FAST, MACD_EMA_SLOW,
        MACD, MACD_SIGNAL, MACD_HISTOGRAM,
        symbol_name, 12, 26, 9)
    plotter.plot_rsi(df.tail(120), CLOSE, RSI_AVG_GAIN, RSI_AVG_LOSS, RSI, symbol_name, 14)
    plotter.plot_bb_macd_rsi(
        df.tail(120), CLOSE,
        MACD_EMA_FAST, MACD_EMA_SLOW,
        BB, MACD, MACD_SIGNAL, RSI,
        symbol_name, 12, 26, 20, 2, 9, 14)


@click.command(help="Run app_analyzer.")
@click.option('--indices', '-i', multiple=True, help="Indices to analyze (e.g. ^SPX, ^TWSE, ^KOSPI, etc.)")
@click.option('--equities', '-e', multiple=True, help="Equities to analyze (e.g. SCHB, AMZN, TSLA, etc.)")
@click.option('--start', multiple=False, help="Use data from start date (YYY-MM-DD). Period will be (start,end].")
@click.option('--end', multiple=False, help="Use data from end date (YYY-MM-DD). Period will be (start,end].")
def main(indices, equities, start, end):
    # defaults
    symbols_indices = ("^SPX", "^NDQ", "^NDX", "^DJI", "^TWSE", "^KOSPI", "^NKX", "^HSI", "^STI", "^SHC", "^SHBS")
    symbols_equities = ("SCHB", "SCHX", "AMZN", "GOOG", "MSFT", "FB", "AAPL", "NFLX", "TSLA")
    start_date = "1980-01-01"
    end_date = date.today()

    # initialize symbols (indices and equities) from command line
    if indices:
        symbols_indices = indices
    if equities:
        symbols_equities = equities

    # initialize start (inclusive) and end (inclusive) date range from command line
    if start:
        start_date = start
    if end:
        end_date = end

    # analyze
    compute_indices(symbols_indices, start_date, end_date)
    compute_equities(symbols_equities, start_date, end_date)


if __name__ == "__main__":
    main()
