"""
Use generated models to perform predictions.

@author: eyu
"""

import os
import logging
import click
import glob
import pandas as pd
from datetime import date

from keras.models import load_model

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


def process_indices(
        symbols, start_date, end_date, reader,
        model_dir, db_dir, db_perf_dir, db_predict_dir, plot_predict, plot_ta):
    """
    :param symbols: list of index symbols to fetch, compute and plot
    :param start_date: earliest date to fetch
    :param end_date: latest date to fetch
    :param reader: data reader (e.g. yahoo, stooq, etc.)
    :param model_dir: directory location containing model for prediction
    :param db_dir: directory location to save processed dataframe
    :param db_perf_dir: directory location to save performance dataframe
    :param db_predict_dir: directory location to save future prediction dataframe
    :param plot_predict: true = plot prediction charts, false = do not plot prediction charts
    :param plot_ta: true = plot TA charts, false = do not plot TA charts
    :return:
    """
    plotter_predict = PAPlot(c.CHART_PREDICT_DIR)
    plotter_ta = TAPlot(c.CHART_TA_DIR)
    for symbol in symbols:
        symbol_name = symbol[1:]  # strip off the ^ character
        # load data
        df = reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Close' (no adjusted close for indices) as our close price
        df = talib.copy_column(df, "Close", c.CLOSE)
        # compute ta
        df = compute_ta(df)
        # compute performance
        df_perf = compute_perf(df, symbol_name)
        # compute prediction
        df, df_predict = predict_all(df, symbol_name, model_dir, plot_predict, plotter_predict)
        # save all
        save_all(symbol_name, df, df_perf, df_predict, db_dir, db_perf_dir, db_predict_dir)
        # plot TA charts
        plot_ta_charts(symbol_name, df, plot_ta, plotter_ta)


def process_equities(
        symbols, start_date, end_date, reader,
        model_dir, db_dir, db_perf_dir, db_predict_dir, plot_predict, plot_ta):
    """
    :param symbols: list of equity symbols to fetch, compute and plot
    :param start_date: earliest date to fetch
    :param end_date: latest date to fetch
    :param reader: data reader (e.g. yahoo, stooq, etc.)
    :param model_dir: directory location containing model for prediction
    :param db_dir: directory location to save processed dataframe
    :param db_perf_dir: directory location to save performance dataframe
    :param db_predict_dir: directory location to save future prediction dataframe
    :param plot_predict: true = plot prediction charts, false = do not plot prediction charts
    :param plot_ta: true = plot TA charts, false = do not plot TA charts
    :return:
    """
    plotter_predict = PAPlot(c.CHART_PREDICT_DIR)
    plotter_ta = TAPlot(c.CHART_TA_DIR)
    for symbol in symbols:
        symbol_name = symbol
        # load data
        df = reader.load(symbol, start_date, end_date, symbol_name)
        # use 'Adj Close' as our close price
        df = talib.copy_column(df, "Adj Close", c.CLOSE)
        # compute ta
        df = compute_ta(df)
        # compute performance
        df_perf = compute_perf(df, symbol_name)
        # compute prediction
        df, df_predict = predict_all(df, symbol_name, model_dir, plot_predict, plotter_predict)
        # save all
        save_all(symbol_name, df, df_perf, df_predict, db_dir, db_perf_dir, db_predict_dir)
        # plot TA charts
        plot_ta_charts(symbol_name, df, plot_ta, plotter_ta)


def compute_ta(df):
    # compute all technical indicators
    df = talib.compute_all_ta(df)
    return df


def compute_perf(df, symbol_name):
    # compute all performance
    df_perf = talib.compute_all_perf(df, symbol_name)

    return df_perf


def predict_all(df, symbol_name, model_dir, plot, plotter):
    # predict for each specific feature
    d_date = df.index[-1]
    df_predict = pd.DataFrame()
    features = ("close",)  # ("close", "low", "high", "macd", "macd-signal", "rsi")
    for feature in features:
        model_file_pattern = os.path.join(model_dir, "model_{symbol_name}_{feature}_*".format(
            symbol_name=symbol_name, feature=feature).lower())
        for model_file in glob.glob(model_file_pattern):
            # instantiate model from file
            tokens = model_file.split('_')
            network_name = tokens[3]
            #hidden_nodes = int(tokens[4])
            #steps_in = int(tokens[5])
            #steps_out = int(tokens[6])
            #features = int(tokens[7])
            #network_function = getattr(net, network_name)
            #model = network_function(hidden_nodes, steps_in, steps_out, features, model_file)

            # alternative way to instantiate model from file
            model = load_model(model_file)
            steps_in = model.input_shape[1]
            steps_out = model.output_shape[1]

            # predict for known data
            df, sequences, x, y, scaler = palib.get_normalized_train_dataset_xy(df, feature, steps_in, steps_out)
            y_pred = model.predict(x)

            # add prediction to dataframe (only if predicting one step)
            if steps_out == 1:
                y_pred_denormalized = scaler.inverse_transform(y_pred.reshape(-1, 1))
                y_pred_denormalized = y_pred_denormalized.flatten() # flatten from 2-D to 1-D
                pred_series = pd.Series(y_pred_denormalized, index=df.index[steps_in:])
                df["p_" + feature] = pred_series
                if plot:
                    plotter.plot_prediction(
                        y_pred[-90:], y[-90:], scaler, network_name, symbol_name, feature, steps_in, steps_out)

            # predict future 'y' values beyond the last known 'y' values;
            # the date of last known 'y' value is 'p_d_date', and future 'y' values will be D+1, D+2, etc.)
            x_future = palib.get_normalized_dataset_x(sequences, feature, steps_in)
            y_future = model.predict(x_future[-steps_out:])
            y_future_denormalized = scaler.inverse_transform(y_future.reshape(-1, 1))
            y_future_denormalized = y_future_denormalized.flatten()  # flatten from 2-D to 1-D
            future_series = pd.Series(y_future_denormalized)
            df_predict = pd.concat([df_predict, future_series.rename("p_future_" + feature)], axis=1)
    df_predict["p_d_date"] = d_date

    return df, df_predict


def save_all(symbol_name, df, df_perf, df_predict, db_dir, db_perf_dir, db_predict_dir):
    # check if the directories exist
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    if not os.path.exists(db_perf_dir):
        os.makedirs(db_perf_dir)
    if not os.path.exists(db_predict_dir):
        os.makedirs(db_predict_dir)

    # save "processed" dataframe to csv
    csv_file = os.path.join(db_dir, symbol_name + ".csv")
    df.to_csv(csv_file, index_label="Date")

    # save performance dataframe to csv
    csv_file = os.path.join(db_perf_dir, symbol_name + ".csv")
    df_perf.to_csv(csv_file, index=False)

    # save "future" prediction dataframe to csv
    csv_file = os.path.join(db_predict_dir, symbol_name + ".csv")
    df_predict.to_csv(csv_file, index_label="p_index")


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
@click.option('--db_index', multiple=False, help="Location to save processed index data.")
@click.option('--db_stock', multiple=False, help="Location to save processed stock data.")
@click.option('--db_etf', multiple=False, help="Location to save processed ETF data.")
@click.option('--db_perf', multiple=False, help="Location to save performance data.")
@click.option('--db_predict', multiple=False, help="Location to save future prediction data.")
@click.option('--plot_predict/--no-plot_predict', multiple=False, help="plot prediction charts", default=True)
@click.option('--plot_ta/--no-plot_ta', multiple=False, help="plot technical analysis charts", default=True)
@click.option('--model', multiple=False, required=True, help="Location of model for predicting.")
def main(indices, stocks, etfs, start, end, db_index, db_stock, db_etf, db_perf, db_predict,
         plot_predict, plot_ta, model):
    # defaults
    symbols_indices = c.SYMBOLS_INDICES
    symbols_stocks = c.SYMBOLS_STOCKS
    symbols_etfs = c.SYMBOLS_ETFS
    start_date = "1980-01-01"
    end_date = date.today()
    db_index_dir = c.DB_SYMBOL_DIR
    db_stock_dir = c.DB_SYMBOL_DIR
    db_etf_dir = c.DB_SYMBOL_DIR
    db_perf_dir = c.DB_PERF_DIR
    db_predict_dir = c.DB_PREDICT_DIR

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

    # initialize db_index_dir (for index data)
    if db_index:
        db_index_dir = db_index

    # initialize db_stock_dir (for stock data)
    if db_stock:
        db_stock_dir = db_stock

    # initialize db_etf_dir (for ETF data)
    if db_etf:
        db_etf_dir = db_etf

    # initialize db_perf_dir (for performance data)
    if db_perf:
        db_perf_dir = db_perf

    # initialize db_predict_dir (for predictions data)
    if db_predict:
        db_predict_dir = db_predict

    reader_stooq = StooqDataReader()
    reader_yahoo = YahooDataReader()

    # process indices (non-US)
    process_indices(c.SYMBOLS_INDICES_NON_US, start_date, end_date, reader_stooq,
                    model, db_index_dir, db_perf_dir, db_predict_dir, plot_predict, plot_ta)

    # process indices
    process_indices(symbols_indices, start_date, end_date, reader_yahoo,
                    model, db_index_dir, db_perf_dir, db_predict_dir, plot_predict, plot_ta)

    # process stocks
    process_equities(symbols_stocks, start_date, end_date, reader_yahoo,
                     model, db_stock_dir, db_perf_dir, db_predict_dir, plot_predict, plot_ta)

    # process ETFs
    process_equities(symbols_etfs, start_date, end_date, reader_yahoo,
                     model, db_etf_dir, db_perf_dir, db_predict_dir, plot_predict, plot_ta)


if __name__ == "__main__":
    main()
