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


def process_indices(symbols, start_date, end_date):
    """
    ^SPX = S&P500 = cap-weighted index of the 500 largest U.S. publicly traded companies
    ^TWSE = cap-weighted index of all listed common shares traded on the Taiwan Stock Exchange
    ^KOSPI = cap-weighted index of all listed common shares traded on the Korea Exchange
    ^NKX = price-weighted index of top 225 blue-chip companies traded on the Tokyo Stock Exchange
    ^HSI = cap-weighted index of the largest companies on the Hong Kong Exchange
    ^STI = cap-weighted index of top 30 companies on the Singapore Exchange
    ^SHC = cap-weighted index of all stocks (A-shares and B-shares) traded on the Shanghai Stock Exchange
    ^SHBS = cap-weighted index of all B-shares traded on the Shanghai Stock Exchange

    :param symbols: list of index symbols to fetch, compute and plot
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
        df = talib.copy_column(df, "Close", c.CLOSE)
        # compute all technical indicators
        df = compute_all(df)
        # plot all charts
        plot_all(symbol_name, df, plotter)


def process_equities(symbols, start_date, end_date):
    """
    :param symbols: list of equity symbols to fetch, compute and plot
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
        df = talib.copy_column(df, "Adj Close", c.CLOSE)
        # compute all technical indicators
        df = compute_all(df)
        # plot all charts
        plot_all(symbol_name, df, plotter)


def compute_all(df):
    # compute daily change and daily percentage change of closing price
    df = talib.compute_daily_change(df, c.CLOSE, c.DAILY_CHG, c.DAILY_CHG_PC)
    # compute daily change between open price and previous closing price
    df = talib.compute_daily_change_between_current_and_previous(
        df, c.OPEN, c.CLOSE,
        c.OPEN_PREV_CLOSE, c.OPEN_PREV_CLOSE_PC)
    # compute 52 week range (low~high)
    df = talib.compute_52_week_range(df, c.LOW, c.HIGH, c.R52_WK_LOW, c.R52_WK_HIGH)
    # compute percentage change of close price above the 52 week low price
    df = talib.compute_pc_above(df, c.CLOSE, c.R52_WK_LOW, c.CLOSE_ABOVE_52_WK_LOW)
    # compute percentage change of close price below the 52 week high price
    df = talib.compute_pc_below(df, c.CLOSE, c.R52_WK_HIGH, c.CLOSE_BELOW_52_WK_HIGH)
    # compute SMA of close price
    df = talib.compute_sma(df, c.CLOSE, c.SMA, 20)
    # compute EMA of close price
    df = talib.compute_ema(df, c.CLOSE, c.EMA_FAST, c.EMA_SLOW, c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, 12, 26)  # short
    df = talib.compute_ema(df, c.CLOSE, c.EMA_FAST, c.EMA_SLOW, c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, 50, 200)  # long
    # compute BB of close price with SMA period of 20 and standard deviation of 2
    df = talib.compute_bb(df, c.CLOSE, c.BB, 20, 2)
    # compute MACD of close price
    df = talib.compute_macd(
        df, c.CLOSE,
        c.MACD_EMA_FAST, c.MACD_EMA_SLOW,
        c.MACD, c.MACD_SIGNAL, c.MACD_HISTOGRAM,
        12, 26, 9)
    # compute RSI of close price
    df = talib.compute_rsi(df, c.CLOSE, c.RSI_AVG_GAIN, c.RSI_AVG_LOSS, c.RSI, 14)

    return df


def plot_all(symbol_name, df, plotter):
    plotter.plot_change(df.tail(252), c.CLOSE, c.DAILY_CHG, c.DAILY_CHG_PC, symbol_name)
    plotter.plot_change_between_current_and_previous(
        df.tail(252), c.OPEN, c.CLOSE,
        c.OPEN_PREV_CLOSE, c.OPEN_PREV_CLOSE_PC,
        symbol_name)
    plotter.plot_pc_above(df.tail(252), c.CLOSE, c.R52_WK_LOW, c.CLOSE_ABOVE_52_WK_LOW, symbol_name)
    plotter.plot_pc_below(df.tail(252), c.CLOSE, c.R52_WK_HIGH, c.CLOSE_BELOW_52_WK_HIGH, symbol_name)
    plotter.plot_sma(df.tail(252), c.CLOSE, c.SMA, c.VOLUME, symbol_name, 20)
    plotter.plot_ema(
        df.tail(252), c.CLOSE, c.EMA_FAST, c.EMA_SLOW,
        c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, c.VOLUME,
        symbol_name, 12, 26)
    plotter.plot_ema(
        df.tail(252), c.CLOSE, c.EMA_FAST, c.EMA_SLOW,
        c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, c.VOLUME,
        symbol_name, 50, 200)
    plotter.plot_bb(df.tail(252), c.CLOSE, c.BB, c.VOLUME, symbol_name, 20, 2)
    plotter.plot_macd(
        df.tail(120), c.CLOSE,
        c.MACD_EMA_FAST, c.MACD_EMA_SLOW,
        c.MACD, c.MACD_SIGNAL, c.MACD_HISTOGRAM,
        symbol_name, 12, 26, 9)
    plotter.plot_rsi(df.tail(120), c.CLOSE, c.RSI_AVG_GAIN, c.RSI_AVG_LOSS, c.RSI, symbol_name, 14)
    plotter.plot_bb_macd_rsi(
        df.tail(120), c.CLOSE,
        c.MACD_EMA_FAST, c.MACD_EMA_SLOW,
        c.BB, c.MACD, c.MACD_SIGNAL, c.RSI,
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

    # process
    process_indices(symbols_indices, start_date, end_date)
    process_equities(symbols_equities, start_date, end_date)


if __name__ == "__main__":
    main()
