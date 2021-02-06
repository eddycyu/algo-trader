"""
Analyuze

@author: eyu
"""

import os
import logging
import click
import pandas as pd
from datetime import date

from data_reader_stooq import StooqDataReader
from data_reader_yahoo import YahooDataReader
import talib as talib
import palib as palib
from taplot import TAPlot
from paplot import PAPlot
import network as net
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
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_formatter)
console_handler.setFormatter(log_formatter)

# add handler to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def process_indices(symbols, start_date, end_date, reader, db_dir, plot_ta):
    """
    :param symbols: list of index symbols to fetch, compute and plot
    :param start_date: earliest date to fetch
    :param end_date: latest date to fetch
    :param reader: data reader (e.g. yahoo, stooq, etc.)
    :param db_dir: directory location to save processed dataframe
    :param plot_ta: true = plot TA charts, false = do not plot TA charts
    :return:
    """
    df_all = {}
    df_perf_all = pd.DataFrame()
    plotter_ta = TAPlot(c.CHART_TA_DIR)
    for symbol in symbols:
        symbol_name = symbol[1:]  # strip off the ^ character
        # load data
        df = reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Close' (no adjusted close for indices) as our close price
        df = talib.copy_column(df, "Close", c.CLOSE)
        # compute all
        df, df_perf = compute_all(df, symbol_name)
        # aggregate symbol dataframes
        df_all[symbol] = df
        df_perf_all = df_perf_all.append(df_perf, ignore_index=True)
        # plot TA charts
        #plot_ta_charts(symbol_name, df, plot_ta, plotter_ta)

    # analyze
    analyze(df_all, df_perf_all)


def process_equities(symbols, start_date, end_date, reader, db_dir, plot_ta):
    """
    :param symbols: list of equity symbols to fetch, compute and plot
    :param start_date: earliest date to fetch
    :param end_date: latest date to fetch
    :param reader: data reader (e.g. yahoo, stooq, etc.)
    :param db_dir: directory location to save processed dataframe
    :param plot_ta: true = plot TA charts, false = do not plot TA charts
    :return:
    """
    df_all = {}
    df_perf_all = pd.DataFrame()
    plotter_ta = TAPlot(c.CHART_TA_DIR)
    for symbol in symbols:
        symbol_name = symbol
        # load data
        df = reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Adj Close' as our close price
        df = talib.copy_column(df, "Adj Close", c.CLOSE)
        # compute all
        df, df_perf = compute_all(df, symbol_name)
        # aggregate symbol dataframes
        df_all[symbol] = df
        df_perf_all = df_perf_all.append(df_perf, ignore_index=True)
        # plot TA charts
        #plot_ta_charts(symbol_name, df, plot_ta, plotter_ta)

    # analyze
    analyze(df_all, df_perf_all)


def compute_all(df, symbol_name):

    # compute all technical indicators
    df = talib.compute_all_ta(df)

    # compute all performance
    df_perf = talib.compute_all_perf(df, symbol_name)

    return df, df_perf


def analyze(df, df_perf):

    df_perf1 = df_perf.sort_values(by=["u_annualized_return-2020"], ascending=False).reset_index(drop=True)
    df_perf1 = df_perf1[["symbol", "u_annualized_return-2020"]]
    print(df_perf1.head(10))

    df_perf2 = df_perf.sort_values(by=["u_annualized_return-2019"], ascending=False).reset_index(drop=True)
    df_perf2 = df_perf2[["symbol", "u_annualized_return-2019"]]
    print(df_perf2.head(10))

    df_perf3 = df_perf.sort_values(by=["u_annualized_return-2018"], ascending=False).reset_index(drop=True)
    df_perf3 = df_perf3[["symbol", "u_annualized_return-2018"]]
    print(df_perf3.head(10))

    df_perf4 = df_perf.sort_values(by=["u_annualized_return-2017"], ascending=False).reset_index(drop=True)
    df_perf4 = df_perf4[["symbol", "u_annualized_return-2017"]]
    print(df_perf4.head(10))

    df_perf5 = df_perf.sort_values(by=["u_annualized_return-2016"], ascending=False).reset_index(drop=True)
    df_perf5 = df_perf5[["symbol", "u_annualized_return-2016"]]
    print(df_perf5.head(10))

    df_perf6 = df_perf.sort_values(by=["u_annualized_return-2015"], ascending=False).reset_index(drop=True)
    df_perf6 = df_perf6[["symbol", "u_annualized_return-2015"]]
    print(df_perf6.head(10))

    df_perf7 = df_perf.sort_values(by=["u_annualized_return-2014"], ascending=False).reset_index(drop=True)
    df_perf7 = df_perf7[["symbol", "u_annualized_return-2014"]]
    print(df_perf7.head(10))

    df_perf8 = df_perf.sort_values(by=["u_annualized_return-2013"], ascending=False).reset_index(drop=True)
    df_perf8 = df_perf8[["symbol", "u_annualized_return-2013"]]
    print(df_perf8.head(10))

    df_perf9 = df_perf.sort_values(by=["u_annualized_return-2012"], ascending=False).reset_index(drop=True)
    df_perf9 = df_perf9[["symbol", "u_annualized_return-2012"]]
    print(df_perf9.head(10))

    df_perf10 = df_perf.sort_values(by=["u_annualized_return-2011"], ascending=False).reset_index(drop=True)
    df_perf10 = df_perf10[["symbol", "u_annualized_return-2011"]]
    print(df_perf10.head(10))


def plot_ta_charts(symbol_name, df, plot, plotter):
    if not plot:
        return

    # plotter.plot_change(df.tail(252), c.CLOSE, c.DAILY_CHG, c.DAILY_CHG_PC, symbol_name)
    # plotter.plot_change_between_current_and_previous(
    #     df.tail(252), c.OPEN, c.CLOSE,
    #     c.OPEN_PREV_CLOSE, c.OPEN_PREV_CLOSE_PC,
    #     symbol_name)
    # plotter.plot_pc_above(df.tail(252), c.CLOSE, c.R52_WK_LOW, c.CLOSE_ABOVE_52_WK_LOW, symbol_name)
    # plotter.plot_pc_below(df.tail(252), c.CLOSE, c.R52_WK_HIGH, c.CLOSE_BELOW_52_WK_HIGH, symbol_name)
    plotter.plot_sma(df.tail(252), c.CLOSE, c.SMA, c.VOLUME, symbol_name, (50, 100, 200))
    plotter.plot_sma_cross(
        df.tail(252), c.CLOSE, c.SMA, c.SMA_GOLDEN_CROSS, c.SMA_DEATH_CROSS, c.VOLUME, symbol_name, 50, 200)
    plotter.plot_ema(df.tail(252), c.CLOSE, c.EMA, c.VOLUME, symbol_name, (12, 26, 50, 200))
    plotter.plot_ema_cross(
        df.tail(252), c.CLOSE, c.EMA, c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, c.VOLUME, symbol_name, 12, 26)
    plotter.plot_ema_cross(
        df.tail(252), c.CLOSE, c.EMA, c.EMA_GOLDEN_CROSS, c.EMA_DEATH_CROSS, c.VOLUME, symbol_name, 50, 200)
    plotter.plot_adtv(df.tail(252), c.CLOSE, c.ADTV, c.VOLUME, symbol_name, (30, 90, 180, 365))
    plotter.plot_bb(df.tail(252), c.CLOSE, c.BB, c.VOLUME, symbol_name, 20, 2)
    plotter.plot_macd(df.tail(252), c.CLOSE, c.EMA, c.MACD, c.MACD_SIGNAL, c.MACD_HISTOGRAM, symbol_name, 12, 26, 9)
    plotter.plot_rsi(df.tail(252), c.CLOSE, c.RSI_AVG_GAIN, c.RSI_AVG_LOSS, c.RSI, symbol_name, 7)
    plotter.plot_rsi(df.tail(252), c.CLOSE, c.RSI_AVG_GAIN, c.RSI_AVG_LOSS, c.RSI, symbol_name, 14)
    plotter.plot_rsi(df.tail(252), c.CLOSE, c.RSI_AVG_GAIN, c.RSI_AVG_LOSS, c.RSI, symbol_name, 21)
    plotter.plot_bb_macd_rsi(
        df.tail(252), c.CLOSE, c.EMA, c.BB, c.MACD, c.MACD_SIGNAL, c.RSI, symbol_name, 12, 26, 20, 2, 9, 7)
    plotter.plot_bb_macd_rsi(
        df.tail(252), c.CLOSE, c.EMA, c.BB, c.MACD, c.MACD_SIGNAL, c.RSI, symbol_name, 12, 26, 20, 2, 9, 14)
    plotter.plot_bb_macd_rsi(
        df.tail(252), c.CLOSE, c.EMA, c.BB, c.MACD, c.MACD_SIGNAL, c.RSI, symbol_name, 12, 26, 20, 2, 9, 21)


@click.command(help="Run app_analyzer.")
@click.option('--indices', '-i', multiple=True, help="Indices to analyze (e.g. ^SPX, ^DJI, ^TWSE, ^KOSPI, etc.)")
@click.option('--stocks', '-s', multiple=True, help="Stocks to analyze (e.g. AAPL, AMZN, GOOGL, TSLA, etc.)")
@click.option('--etfs', '-e', multiple=True, help="ETFs to analyze (e.g. SCHB, SCHX, VTI, VOO, etc.)")
@click.option('--start', multiple=False, help="Use data from start date (YYY-MM-DD). Period will be (start,end].")
@click.option('--end', multiple=False, help="Use data from end date (YYY-MM-DD). Period will be (start,end].")
@click.option('--dbi', multiple=False, help="Location to save processed index data.")
@click.option('--dbs', multiple=False, help="Location to save processed stock data.")
@click.option('--dbe', multiple=False, help="Location to save processed ETF data.")
@click.option('--plot_ta/--no-plot_ta', multiple=False, help="plot technical analysis charts", default=True)
def main(indices, stocks, etfs, start, end, dbi, dbs, dbe, plot_ta):
    # defaults
    symbols_indices = c.SYMBOLS_INDICES
    symbols_stocks = c.SYMBOLS_STOCKS
    symbols_etfs = c.SYMBOLS_ETFS
    start_date = "1980-01-01"
    end_date = date.today()
    dbi_dir = c.DB_SYMBOL_DIR
    dbs_dir = c.DB_SYMBOL_DIR
    dbe_dir = c.DB_SYMBOL_DIR

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

    # initialize dbi_dir (for indices)
    if dbi:
        dbi_dir = dbi

    # initialize dbs_dir (for stocks)
    if dbs:
        dbs_dir = dbs

    # initialize dbe_dir (for ETFs)
    if dbe:
        dbe_dir = dbe

    # check if the directories exist
    if not os.path.exists(dbi_dir):
        os.makedirs(dbi_dir)
    if not os.path.exists(dbs_dir):
        os.makedirs(dbs_dir)
    if not os.path.exists(dbe_dir):
        os.makedirs(dbe_dir)

    reader_stooq = StooqDataReader()
    reader_yahoo = YahooDataReader()

    # process indices (non-US)
    process_indices(c.SYMBOLS_INDICES_NON_US, start_date, end_date, reader_stooq, dbi_dir, plot_ta)

    # process indices
    process_equities(symbols_indices, start_date, end_date, reader_yahoo, dbs_dir, plot_ta)

    # process stocks
    process_equities(symbols_stocks, start_date, end_date, reader_yahoo, dbs_dir, plot_ta)

    # process ETFs
    process_equities(symbols_etfs, start_date, end_date, reader_yahoo, dbs_dir, plot_ta)


if __name__ == "__main__":
    main()
