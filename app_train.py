"""
Train networks to generate models for prediction.

https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
https://www.datacamp.com/community/tutorials/lstm-python-stock-market

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
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_formatter)
console_handler.setFormatter(log_formatter)

# add handler to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def process_indices(symbols, start_date, end_date, model_dir):
    """
    :param symbols: list of index symbols to fetch, predict and plot
    :param start_date: earliest date to fetch
    :param end_date: latest date to fetch
    :param model_dir: location to save model
    :return:
    """
    data_reader = StooqDataReader()
    plotter = PAPlot(c.CHART_TRAIN_DIR)
    for symbol in symbols:
        symbol_name = symbol[1:]  # strip off the ^ character
        # load data
        df = data_reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Close' (no adjusted close for indices) as our close price
        df = talib.copy_column(df, "Close", c.CLOSE)
        # train and plot all
        train_all(df, symbol, symbol_name, model_dir, plotter)


def process_equities(symbols, start_date, end_date, model_dir):
    """
    :param symbols: list of equity symbols to fetch, predict and plot
    :param start_date: earliest date to fetch
    :param end_date: latest date to fetch
    :param model_dir: location to save model
    :return:
    """
    data_reader = YahooDataReader()
    plotter = PAPlot(c.CHART_TRAIN_DIR)
    for symbol in symbols:
        symbol_name = symbol
        # load data
        df = data_reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Adj Close' as our close price
        df = talib.copy_column(df, "Adj Close", c.CLOSE)
        # train and plot all
        train_all(df, symbol, symbol_name, model_dir, plotter)


def train_all(df, symbol, symbol_name, model_dir, plotter):
    # compute technical indicators
    df = talib.compute_all(df)

    # remove rows with missing data (in specified columns)
    df = df.dropna(subset=[c.CLOSE, c.LOW, c.HIGH, "u_macd-12-26-9", "u_macd_signal-12-26-9", "u_rsi-14"])

    # make sure there is sufficient data to train model for prediction (min = 252 = trading days in a year)
    df_size = len(df.index)
    if df_size < 252:
        logger.info("Insufficient data (size: {size}) to train model! Skipping ({symbol}).".format(
            size=df_size, symbol=symbol))
        return df

    # split data to create training (80%), validation (10%) and testing (10%) datasets
    # NOTE: split before normalizing to avoid introducing future information into training
    df_train, df_val_test = palib.split_df_by_ratio(df, test_ratio=0.2)
    df_val, df_test = palib.split_df_by_ratio(df_val_test, test_ratio=0.5)

    # verify settings exists for training symbol
    settings = c.TRAIN_SETTINGS.get(symbol)
    if settings is None:
        logging.error("No training settings found for [{symbol}]! Skipping.".format(symbol=symbol))
        return

    # get settings for training symbol
    epochs = settings["epochs"]
    steps_in = settings["steps_in"]
    steps_out = settings["steps_out"]

    # train and generate model for each specific feature
    features = ("close",)  # "close", "low", "high", "macd", "macd-signal", "rsi")
    for feature in features:
        # create datasets
        df_train, sequences_train, x_train, y_train, scaler_train = palib.get_normalized_train_dataset_xy(
            df_train, feature, steps_in, steps_out)
        df_val, sequences_val, x_val, y_val, scaler_val = palib.get_normalized_train_dataset_xy(
            df_val, feature, steps_in, steps_out)
        df_test, sequences_test, x_test, y_test, scaler_test = palib.get_normalized_train_dataset_xy(
            df_test, feature, steps_in, steps_out)
        # train
        network_list = [net.lstm1]
        palib.train_and_test_networks(
            x_train, y_train, x_val, y_val, x_test, y_test, scaler_train,
            steps_out, epochs, symbol_name, feature, network_list, model_dir, plotter)


@click.command(help="Run app_predictor.")
@click.option('--indices', '-i', multiple=True, help="Indices to analyze (e.g. ^SPX, ^DJI, ^TWSE, ^KOSPI, etc.)")
@click.option('--stocks', '-s', multiple=True, help="Stocks to analyze (e.g. AAPL, AMZN, GOOGL, TSLA, etc.)")
@click.option('--etfs', '-e', multiple=True, help="ETFs to analyze (e.g. SCHB, SCHX, VTI, VOO, etc.)")
@click.option('--start', multiple=False, help="Use data from start date (YYY-MM-DD). Period will be (start,end].")
@click.option('--end', multiple=False, help="Use data from end date (YYY-MM-DD). Period will be (start,end].")
@click.option('--model', multiple=False, required=True, help="Location to save model.")
def main(indices, stocks, etfs, start, end, model):
    # defaults
    symbols_indices = c.SYMBOLS_INDICES
    symbols_stocks = c.SYMBOLS_STOCKS
    symbols_etfs = c.SYMBOLS_ETFS
    start_date = "1980-01-01"
    end_date = date.today()
    model_dir = c.MODEL_DIR

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

    # initialize model_dir
    if model:
        model_dir = model

    # check if the directories exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # process
    process_indices(symbols_indices, start_date, end_date, model_dir)
    process_equities(symbols_stocks, start_date, end_date, model_dir)
    process_equities(symbols_etfs, start_date, end_date, model_dir)


if __name__ == "__main__":
    main()
