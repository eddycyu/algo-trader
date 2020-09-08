"""
Predictive Analysis (PA) Plotter

Library of functions to plot various predictive analysis charts.

@author: eyu
"""

import os
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import constants as c


class PAPlot(object):
    """
    Class for PA plots.
    """

    def __init__(self, chart_dir=c.CHART_PA_DIR):
        self.chart_dir = chart_dir
        self.logger = logging.getLogger("algo-trader")

        # check if the chart (for predictive analysis) directory exists; if not, make it
        if not os.path.exists(chart_dir):
            os.makedirs(chart_dir)

    def plot_multiple(self, df, *column_sources, symbol_name, xlabel="", ylabel="Price"):
        # generate chart
        fig = plt.figure(figsize=(16, 10))
        subplot_count = len(column_sources)
        ax = fig.subplots(subplot_count)
        i = 0
        columns = ""
        for column_source in column_sources:
            earliest_date = str(df[column_source].head(1).index.date[0])
            latest_date = str(df[column_source].tail(1).index.date[0])
            ax[i].set_title("[" + symbol_name + "] - [" + column_source + "] - ["
                            + earliest_date + "~" + latest_date + "]")
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)
            df[column_source].plot(ax=ax[i], color="green", legend=True, label=column_source, zorder=2)

            # set major tick locator
            ax[i].xaxis.set_major_locator(mdates.MonthLocator())

            # set major tick labels with no rotation
            plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=0)

            # turn on grid
            ax[i].grid(color="lightgray", alpha=0.5, zorder=1)

            # append column_source to filename
            columns = columns + "[" + column_source + "]"

            # increment to next subplot
            i = i + 1

        # save chart
        fig.tight_layout()
        output_file = os.path.join(
            self.chart_dir, "plot-" + columns + "-[" + symbol_name + "]" + ".png")
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_loss(self, network, model_name, symbol_name, look_back, foresight, xlabel="Epoch", ylabel=""):
        # generate chart
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_title("[{symbol}] - Loss [Model: {model_name}][LB: {lb:d}][F: {fs:d}]]".format(
            symbol=symbol_name, model_name=model_name, lb=look_back, fs=foresight))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # plot loss
        loss_min = min(network.history["loss"])
        val_loss_min = min(network.history["val_loss"])
        ax.plot(network.history["loss"], label="training loss ({:.8f})".format(loss_min), zorder=2)
        ax.plot(network.history["val_loss"], label="validation loss ({:.8f})".format(val_loss_min), zorder=2)
        plt.grid(color="lightgray", alpha=0.5, zorder=1)
        plt.legend()

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "loss-[{model_name}]-[{symbol_name}]-[{lb:d}-{fs:d}].png".format(
            model_name=model_name, symbol_name=symbol_name, lb=look_back, fs=foresight))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_prediction(
            self, y_pred, y_test, scaler, model_name, symbol_name, look_back, foresight, xlabel="Date", ylabel="Price"):
        # generate chart
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_title("[{symbol}] - Prediction [Model: {model_name}][LB: {lb:d}][F: {fs:d}]]".format(
            symbol=symbol_name, model_name=model_name, lb=look_back, fs=foresight))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # plot actual vs prediction
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        predicted = scaler.inverse_transform(y_pred.reshape(-1, 1))
        last_actual = actual[-1][0]
        last_predicted = predicted[-1][0]
        ax.plot(actual, label="Actual ({:.4f})".format(last_actual),
                color="green", zorder=2)
        ax.plot(predicted, label="Predicted ({:.4f})".format(last_predicted),
                color="blue", linestyle="dotted", zorder=2)
        # ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.grid(color="lightgray", alpha=0.5, zorder=1)
        plt.legend()

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "pred-[{model_name}]-[{symbol_name}]-[{lb:d}-{fs:d}].png".format(
            model_name=model_name, symbol_name=symbol_name, lb=look_back, fs=foresight))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)
