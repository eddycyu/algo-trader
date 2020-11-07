"""
Technical Analysis (TA) Plotter

Library of functions to plot various technical indicator charts.

@author: eyu
"""

import os
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import constants as c


class TAPlot(object):
    """
    Class for TA plots.
    """

    def __init__(self, chart_dir=c.CHART_TA_DIR):
        self.chart_dir = chart_dir
        self.logger = logging.getLogger("algo-trader")

        # check if the chart (for technical analysis) directory exists; if not, make it
        if not os.path.exists(chart_dir):
            os.makedirs(chart_dir)

    def plot_sma(self, df, column_close, column_sma, column_volume, symbol_name, time_periods):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot close and SMA
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3, colspan=1)
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        time_periods_string = "-".join(map(str, time_periods))
        ax1.set_title("[{symbol}] - SMA [Period: {period}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, period=time_periods_string, earliest=earliest_date, latest=latest_date))
        ax1.set_ylabel("Price")
        last_close = df[column_close].tail(1)[0]
        df[column_close].plot(ax=ax1, label="Close ({:.4f})".format(last_close), color="green", zorder=3)
        for time_period in time_periods:
            key_sma = column_sma + "-{:d}".format(time_period)
            last_sma = df[key_sma].tail(1)[0]
            df[key_sma].plot(ax=ax1, label="SMA-{:d} ({:.4f})".format(time_period, last_sma), ls="dotted", zorder=2)
            # testing start
            # last_date = df[column_close].tail(1).index.date[0]
            # last_price = df[column_close].tail(1)[0]
            # last_point = pd.DataFrame({'x': [last_date], 'y': [last_price]})
            # last_point.plot(ax=ax1, x="x", y="y", style="bD", markersize=10, label="latest price")
            # testing end
        ax1.legend()

        # plot volume
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1, sharex=ax1)
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax2.set_ylabel("Volume")
        ax2.bar(df[column_volume].index, df[column_volume], color="red", zorder=2)

        # remove x-label
        ax1.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "sma-[{period}]-[{symbol}].png".format(
            period=time_periods_string, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_sma_cross(
            self, df, column_close, column_sma, column_golden_cross, column_death_cross, column_volume,
            symbol_name, time_period_fast, time_period_slow):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot SMA pair
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3, colspan=1)
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        ax1.set_title("[{symbol}] - SMA [Fast: {fast:d}][Slow: {slow:d}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, fast=time_period_fast, slow=time_period_slow,
            earliest=earliest_date, latest=latest_date))
        ax1.set_ylabel("Price")
        key_sma_fast = column_sma + "-{:d}".format(time_period_fast)
        key_sma_slow = column_sma + "-{:d}".format(time_period_slow)
        last_close = df[column_close].tail(1)[0]
        last_sma_fast = df[key_sma_fast].tail(1)[0]
        last_sma_slow = df[key_sma_slow].tail(1)[0]
        df[column_close].plot(
            ax=ax1, label="Close ({:.4f})".format(last_close),
            color="green", zorder=3)
        df[key_sma_fast].plot(
            ax=ax1, label="SMA-{:d} ({:.4f})".format(time_period_fast, last_sma_fast),
            color="blue", linestyle="dotted", zorder=2)
        df[key_sma_slow].plot(
            ax=ax1, label="SMA-{:d} ({:.4f})".format(time_period_slow, last_sma_slow),
            color="red", linestyle="dotted", zorder=2)

        # plot crosses
        key_golden_cross = column_golden_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
        key_death_cross = column_death_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
        for i in range(0, len(df)):
            if df[key_golden_cross].iloc[i]:
                ax1.axvline(
                    x=df[key_golden_cross].index.date[i], color="gold", ls="-", lw=2,
                    label="Golden-X ({})".format(df[key_golden_cross].index.date[i]), zorder=2)
            if df[key_death_cross].iloc[i]:
                ax1.axvline(
                    x=df[key_death_cross].index.date[i], color="magenta", ls="-", lw=2,
                    label="Death-X ({})".format(df[key_death_cross].index.date[i]), zorder=2)
        ax1.legend()

        # plot volume
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1, sharex=ax1)
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax2.set_ylabel("Volume")
        ax2.bar(df[column_volume].index, df[column_volume], color="red", zorder=2)

        # remove x-label
        ax1.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "sma-cross-[{:d}-{:d}]-[{symbol}].png".format(
            time_period_fast, time_period_slow, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_ema(self, df, column_close, column_ema, column_volume, symbol_name, time_periods):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot close and EMA
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3, colspan=1)
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        time_periods_string = "-".join(map(str, time_periods))
        ax1.set_title("[{symbol}] - EMA [Period: {period}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, period=time_periods_string, earliest=earliest_date, latest=latest_date))
        ax1.set_ylabel("Price")
        last_close = df[column_close].tail(1)[0]
        df[column_close].plot(ax=ax1, label="Close ({:.4f})".format(last_close), color="green", zorder=3)
        for time_period in time_periods:
            key_ema = column_ema + "-{:d}".format(time_period)
            last_ema = df[key_ema].tail(1)[0]
            df[key_ema].plot(ax=ax1, label="EMA-{:d} ({:.4f})".format(time_period, last_ema), ls="dotted", zorder=2)
        ax1.legend()

        # plot volume
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1, sharex=ax1)
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax2.set_ylabel("Volume")
        ax2.bar(df[column_volume].index, df[column_volume], color="red", zorder=2)

        # remove x-label
        ax1.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "ema-[{period}]-[{symbol}].png".format(
            period=time_periods_string, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_ema_cross(
            self, df, column_close, column_ema, column_golden_cross, column_death_cross, column_volume,
            symbol_name, time_period_fast, time_period_slow):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot EMA pair
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3, colspan=1)
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        ax1.set_title("[{symbol}] - EMA [Fast: {fast:d}][Slow: {slow:d}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, fast=time_period_fast, slow=time_period_slow,
            earliest=earliest_date, latest=latest_date))
        ax1.set_ylabel("Price")
        key_ema_fast = column_ema + "-{:d}".format(time_period_fast)
        key_ema_slow = column_ema + "-{:d}".format(time_period_slow)
        last_close = df[column_close].tail(1)[0]
        last_ema_fast = df[key_ema_fast].tail(1)[0]
        last_ema_slow = df[key_ema_slow].tail(1)[0]
        df[column_close].plot(
            ax=ax1, label="Close ({:.4f})".format(last_close),
            color="green", zorder=3)
        df[key_ema_fast].plot(
            ax=ax1, label="EMA-{:d} ({:.4f})".format(time_period_fast, last_ema_fast),
            color="blue", linestyle="dotted", zorder=2)
        df[key_ema_slow].plot(
            ax=ax1, label="EMA-{:d} ({:.4f})".format(time_period_slow, last_ema_slow),
            color="red", linestyle="dotted", zorder=2)

        # plot crosses
        key_golden_cross = column_golden_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
        key_death_cross = column_death_cross + "-{:d}-{:d}".format(time_period_fast, time_period_slow)
        for i in range(0, len(df)):
            if df[key_golden_cross].iloc[i]:
                ax1.axvline(
                    x=df[key_golden_cross].index.date[i], color="gold", ls="-", lw=2,
                    label="Golden-X ({})".format(df[key_golden_cross].index.date[i]), zorder=2)
            if df[key_death_cross].iloc[i]:
                ax1.axvline(
                    x=df[key_death_cross].index.date[i], color="magenta", ls="-", lw=2,
                    label="Death-X ({})".format(df[key_death_cross].index.date[i]), zorder=2)
        ax1.legend()

        # plot volume
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1, sharex=ax1)
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax2.set_ylabel("Volume")
        ax2.bar(df[column_volume].index, df[column_volume], color="red", zorder=2)

        # remove x-label
        ax1.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "ema-cross-[{:d}-{:d}]-[{symbol}].png".format(
            time_period_fast, time_period_slow, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_adtv(self, df, column_close, column_adtv, column_volume, symbol_name, time_periods):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot close
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        time_periods_string = "-".join(map(str, time_periods))
        ax1.set_title("[{symbol}] - ADTV [Period: {period}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, period=time_periods_string, earliest=earliest_date, latest=latest_date))
        ax1.set_ylabel("Price")
        last_close = df[column_close].tail(1)[0]
        df[column_close].plot(ax=ax1, label="Close ({:.4f})".format(last_close), color="green", zorder=3)
        ax1.legend()

        # plot volume and ADTV
        ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2, colspan=1, sharex=ax1)
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax2.set_ylabel("Volume")
        ax2.bar(df[column_volume].index, df[column_volume], color="red", zorder=2)
        for time_period in time_periods:
            key_adtv = column_adtv + "-{:d}".format(time_period)
            last_adtv = df[key_adtv].tail(1)[0]
            df[key_adtv].plot(ax=ax2, label="ADTV-{:d} ({:.4f})".format(time_period, last_adtv), ls="dotted", zorder=2)
        ax2.legend()

        # remove x-label
        ax1.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "adtv-[{period}]-[{symbol}].png".format(
            period=time_periods_string, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_bb(self, df, column_close, column_bb, column_volume, symbol_name, time_period, stdev_factor):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot bollinger bands
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3, colspan=1)
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        ax1.set_title(
            "[{symbol}] - Bollinger Bands [SMA Period: {period:d}][Stdev: {stdev}] - [{earliest} ~ {latest}]".format(
                symbol=symbol_name, period=time_period, stdev=stdev_factor, earliest=earliest_date, latest=latest_date))
        ax1.set_ylabel("Price")
        key_sma = column_bb + "-sma-{:d}-{:d}".format(time_period, stdev_factor)
        key_upper = column_bb + "-upper-{:d}-{:d}".format(time_period, stdev_factor)
        key_lower = column_bb + "-lower-{:d}-{:d}".format(time_period, stdev_factor)
        last_close = df[column_close].tail(1)[0]
        last_sma = df[key_sma].tail(1)[0]
        last_upper = df[key_upper].tail(1)[0]
        last_lower = df[key_lower].tail(1)[0]
        df[column_close].plot(
            ax=ax1, label="Close ({:.4f})".format(last_close),
            color="green", legend=True, zorder=4)
        df[key_sma].plot(
            ax=ax1, label="SMA-{:d} ({:.4f})".format(time_period, last_sma),
            color="maroon", legend=True, zorder=3)
        df[key_upper].plot(
            ax=ax1, label="Upper-{:d} ({:.4f})".format(time_period, last_upper),
            color="black", linestyle="dotted", legend=True, zorder=2)
        df[key_lower].plot(
            ax=ax1, label="Lower-{:d} ({:.4f})".format(time_period, last_lower),
            color="black", linestyle="dotted", legend=True, zorder=2)
        plt.fill_between(df.index, df[key_upper], df[key_lower], color="beige", alpha=0.5)

        # plot volume
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1, sharex=ax1)
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax2.set_ylabel("Volume")
        ax2.bar(df[column_volume].index, df[column_volume], color="red", zorder=2)

        # remove x-label
        ax1.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "bb-[{:d}-{:d}]-[{symbol}].png".format(
            time_period, stdev_factor, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_macd(
            self, df, column_close,
            column_ema, column_macd, column_macd_signal, column_macd_histogram,
            symbol_name, time_period_fast, time_period_slow, time_period_macd):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot price and EMA
        ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=4, colspan=1)
        ax1.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax1.set_ylabel("Price")
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        ax1.set_title(
            "[{symbol}] - MACD [Fast: {fast:d}][Slow: {slow:d}][MACD: {macd:d}] - [{earliest} ~ {latest}]".format(
                symbol=symbol_name, fast=time_period_fast, slow=time_period_slow, macd=time_period_macd,
                earliest=earliest_date, latest=latest_date))
        key_ema_fast = column_ema + "-{:d}".format(time_period_fast)
        key_ema_slow = column_ema + "-{:d}".format(time_period_slow)
        last_close = df[column_close].tail(1)[0]
        last_ema_fast = df[key_ema_fast].tail(1)[0]
        last_ema_slow = df[key_ema_slow].tail(1)[0]
        df[column_close].plot(
            ax=ax1, label="Close ({:.4f})".format(last_close),
            color="green", legend=True, zorder=3)
        df[key_ema_fast].plot(
            ax=ax1, label="EMA-{:d} ({:.4f})".format(time_period_fast, last_ema_fast),
            color="blue", linestyle="dotted", legend=True, zorder=2)
        df[key_ema_slow].plot(
            ax=ax1, label="EMA-{:d} ({:.4f})".format(time_period_slow, last_ema_slow),
            color="red", linestyle="dotted", legend=True, zorder=2)

        # plot MACD and signal
        ax2 = plt.subplot2grid((7, 1), (4, 0), rowspan=2, colspan=1, sharex=ax1)
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax2.set_ylabel("MACD")
        time_fast_slow_macd = "-{:d}-{:d}-{:d}".format(time_period_fast, time_period_slow, time_period_macd)
        key_macd = column_macd + time_fast_slow_macd
        key_signal = column_macd_signal + time_fast_slow_macd
        df[key_macd].plot(ax=ax2, color="blue", legend=True, label="MACD", zorder=2)
        df[key_signal].plot(ax=ax2, label="Signal-{:d}".format(time_period_macd), color="red", legend=True, zorder=2)

        # plot histogram
        ax3 = plt.subplot2grid((7, 1), (6, 0), rowspan=1, colspan=1, sharex=ax1)
        ax3.set_ylabel("Histogram")
        key_macd_histogram = column_macd_histogram + time_fast_slow_macd
        ax3.bar(df[key_macd_histogram].index, df[key_macd_histogram], label="Histogram", color="red", zorder=2)

        # remove x-label
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax3.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)
        ax3.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "macd-[{:d}-{:d}-{:d}]-[{symbol}].png".format(
            time_period_fast, time_period_slow, time_period_macd, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_rsi(self, df, column_close, column_avg_gain, column_avg_loss, column_rsi, symbol_name, time_period):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot price
        ax1 = fig.add_subplot(311, ylabel="Price")
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        ax1.set_title("[{symbol}] - RSI [Period: {period:d}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, period=time_period, earliest=earliest_date, latest=latest_date))
        last_close = df[column_close].tail(1)[0]
        df[column_close].plot(ax=ax1, label="Close ({:.4f})".format(last_close), color="green", legend=True, zorder=2)

        # plot average gain and loss
        ax2 = fig.add_subplot(312, ylabel="Avg Gain / Loss")
        key_avg_gain = column_avg_gain + "-{:d}".format(time_period)
        key_avg_loss = column_avg_loss + "-{:d}".format(time_period)
        df[key_avg_gain].plot(ax=ax2, label="Avg Gain", color="green", legend=True, zorder=2)
        df[key_avg_loss].plot(ax=ax2, label="Avg Loss", color="red", legend=True, zorder=2)

        # plot RSI
        ax3 = fig.add_subplot(313, ylabel="RSI")
        key_rsi = column_rsi + "-" + str(time_period)
        df[key_rsi].plot(ax=ax3, color="blue", legend=False, label="RSI-{:d}".format(time_period), zorder=3)
        ax3.axhline(y=70, label="Overbought", color="red", linestyle="dotted", zorder=2)
        ax3.axhline(y=30, label="Oversold", color="green", linestyle="dotted", zorder=2)
        ax3.legend()

        # remove x-label
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax3.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)
        ax3.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "rsi-[{:d}]-[{symbol}].png".format(time_period, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_bb_macd_rsi(
            self, df, column_close,
            column_ema, column_bb, column_macd, column_macd_signal, column_rsi,
            symbol_name, time_period_fast, time_period_slow,
            time_period_bb, stddev_factor, time_period_macd, time_period_rsi):
        # generate chart
        fig = plt.figure(figsize=(16, 10))

        # plot BB with price and EMA
        ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=4, colspan=1)
        ax1.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        earliest_date = str(df[column_close].head(1).index.date[0])
        latest_date = str(df[column_close].tail(1).index.date[0])
        ax1.set_title(
            "[{symbol}] - BB_MACD_RSI [Fast: {fast:d}][Slow: {slow:d}]"
            "[BB: {bb:d}-{stdev:d}][MACD: {macd:d}][RSI: {rsi:d}] - "
            "[{earliest} ~ {latest}]".format(
                symbol=symbol_name, fast=time_period_fast, slow=time_period_slow,
                bb=time_period_bb, stdev=stddev_factor, macd=time_period_macd, rsi=time_period_rsi,
                earliest=earliest_date, latest=latest_date))
        ax1.set_ylabel("Price")
        key_ema_fast = column_ema + "-{:d}".format(time_period_fast)
        key_ema_slow = column_ema + "-{:d}".format(time_period_slow)
        last_close = df[column_close].tail(1)[0]
        last_ema_fast = df[key_ema_fast].tail(1)[0]
        last_ema_slow = df[key_ema_slow].tail(1)[0]
        df[column_close].plot(
            ax=ax1, label="Close ({:.4f})".format(last_close),
            color="green", legend=True, zorder=4)
        df[key_ema_fast].plot(
            ax=ax1, label="EMA-{:d} ({:.4f})".format(time_period_fast, last_ema_fast),
            color="blue", legend=True, linestyle="dotted", zorder=2)
        df[key_ema_slow].plot(
            ax=ax1, label="EMA-{:d} ({:.4f})".format(time_period_slow, last_ema_slow),
            color="red", legend=True, linestyle="dotted", zorder=2)

        # plot BB
        key_bb_sma = column_bb + "-sma-{:d}-{:d}".format(time_period_bb, stddev_factor)
        key_bb_upper = column_bb + "-upper-{:d}-{:d}".format(time_period_bb, stddev_factor)
        key_bb_lower = column_bb + "-lower-{:d}-{:d}".format(time_period_bb, stddev_factor)
        last_bb_sma = df[key_bb_sma].tail(1)[0]
        last_bb_upper = df[key_bb_upper].tail(1)[0]
        last_bb_lower = df[key_bb_lower].tail(1)[0]
        df[key_bb_sma].plot(
            ax=ax1, label="SMA-{:d} ({:.4f})".format(time_period_bb, last_bb_sma),
            color="maroon", legend=True, zorder=3)
        df[key_bb_upper].plot(
            ax=ax1, label="Upper-{:d} ({:.4f})".format(time_period_bb, last_bb_upper),
            color="black", linestyle="dotted", legend=True, zorder=2)
        df[key_bb_lower].plot(
            ax=ax1, label="Lower-{:d} ({:.4f})".format(time_period_bb, last_bb_lower),
            color="black", linestyle="dotted", legend=True, zorder=2)
        plt.fill_between(df.index, df[key_bb_upper], df[key_bb_lower], color="beige", alpha=0.5)

        # plot MACD and signal
        ax2 = plt.subplot2grid((7, 1), (4, 0), rowspan=2, colspan=1)
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        ax2.set_ylabel("MACD")
        time_fast_slow_macd = "-{:d}-{:d}-{:d}".format(time_period_fast, time_period_slow, time_period_macd)
        key_macd = column_macd + time_fast_slow_macd
        key_signal = column_macd_signal + time_fast_slow_macd
        df[key_macd].plot(ax=ax2, label="MACD", color="blue", legend=True, zorder=2)
        df[key_signal].plot(ax=ax2, label="Signal-{:d}".format(time_period_macd), color="red", legend=True, zorder=2)

        # plot RSI
        ax3 = plt.subplot2grid((7, 1), (6, 0), rowspan=1, colspan=1)
        ax3.set_ylabel("RSI")
        key_rsi = column_rsi + "-{:d}".format(time_period_rsi)
        df[key_rsi].plot(ax=ax3, color="blue", legend=False, label="RSI-{:d}".format(time_period_rsi), zorder=3)
        ax3.axhline(y=70, label="Overbought", color="red", linestyle="dotted", zorder=2)
        ax3.axhline(y=30, label="Oversold", color="green", linestyle="dotted", zorder=2)
        ax3.legend()

        # remove x-label
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax3.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)
        ax3.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "bb-macd-rsi-[{:d}-{:d}-{:d}-{:d}-{:d}-{:d}]-[{symbol}].png".format(
            time_period_fast, time_period_slow, time_period_bb, stddev_factor, time_period_macd, time_period_rsi,
            symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_change(self, df, column_name, column_change, column_change_pc, symbol_name):
        # generate chart
        fig = plt.figure(figsize=(16, 10))
        plt.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        earliest_date = str(df[column_name].head(1).index.date[0])
        latest_date = str(df[column_name].tail(1).index.date[0])
        plt.title("[{symbol}] - Change [{column}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, column=column_name, earliest=earliest_date, latest=latest_date))

        # plot column_name (e.g. price)
        ax1 = fig.add_subplot(311)
        ax1.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        last_value = df[column_name].tail(1)[0]
        df[column_name].plot(ax=ax1, label="{} ({:.4f})".format(column_name, last_value),
                             color="green", legend=True, zorder=2)

        # plot change
        ax2 = fig.add_subplot(312, ylabel="Change")
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        df[column_change].plot(ax=ax2, color="blue", legend=False, zorder=2)
        ax2.axhline(y=0, color="black", linestyle="dotted", zorder=2)

        # plot percent change
        ax3 = fig.add_subplot(313, ylabel="% Change")
        df[column_change_pc].plot(ax=ax3, color="blue", legend=False, zorder=2)
        ax3.axhline(y=0, color="black", linestyle="dotted", zorder=2)

        # remove x-label
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax3.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)
        ax3.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "change-for-[{}]-[{symbol}].png".format(
            column_name, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_change_between_current_and_previous(
            self, df, column_current, column_previous, column_change, column_change_pc, symbol_name):
        # generate chart
        fig = plt.figure(figsize=(16, 10))
        plt.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        earliest_date = str(df[column_current].head(1).index.date[0])
        latest_date = str(df[column_current].tail(1).index.date[0])
        plt.title(
            "[{symbol}] - Change B/W Current [{current}] VS Previous [{previous}] - [{earliest} ~ {latest}]".format(
                symbol=symbol_name, current=column_current, previous=column_previous,
                earliest=earliest_date, latest=latest_date))

        # plot current vs previous (e.g. current price vs previous price)
        ax1 = fig.add_subplot(311)
        ax1.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        last_current_value = df[column_current].tail(1)[0]
        last_previous_value = df[column_previous].tail(1)[0]
        df[column_current].plot(
            ax=ax1, label="{} ({:.4f})".format(column_current, last_current_value),
            color="green", legend=True, zorder=2)
        df[column_previous].plot(
            ax=ax1, label="{} ({:.4f})".format(column_previous, last_previous_value),
            color="blue", legend=True, zorder=2)

        # plot change
        ax2 = fig.add_subplot(312, ylabel="Change")
        ax2.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        df[column_change].plot(ax=ax2, color="blue", legend=False, zorder=2)
        ax2.axhline(y=0, color="black", linestyle="dotted", zorder=2)

        # plot percent change
        ax3 = fig.add_subplot(313, ylabel="% Change")
        ax1.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        df[column_change_pc].plot(ax=ax3, color="blue", legend=False, zorder=2)
        ax3.axhline(y=0, color="black", linestyle="dotted", zorder=2)

        # remove x-label
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax3.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)
        ax3.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "change-pc-bw-current-[{}]-previous-[{}]-[{symbol}].png".format(
            column_current, column_previous, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_pc_above(self, df, column_name1, column_name2, column_change_pc, symbol_name):
        # generate chart
        fig = plt.figure(figsize=(16, 10))
        plt.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        earliest_date = str(df[column_name1].head(1).index.date[0])
        latest_date = str(df[column_name1].tail(1).index.date[0])
        plt.title("[{symbol}] - % Change Of [{column1}] Above [{column2}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, column1=column_name1, column2=column_name2,
            earliest=earliest_date, latest=latest_date))

        # plot column_name1 vs column_name2
        ax1 = fig.add_subplot(211)
        ax1.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        last_name1_value = df[column_name1].tail(1)[0]
        last_name2_value = df[column_name2].tail(1)[0]
        df[column_name1].plot(
            ax=ax1, label="{} ({:.4f})".format(column_name1, last_name1_value), color="green", legend=True, zorder=2)
        df[column_name2].plot(
            ax=ax1, label="{} ({:.4f})".format(column_name2, last_name2_value), color="blue", legend=True, zorder=2)

        # plot percent change
        ax2 = fig.add_subplot(212, ylabel="% Change")
        df[column_change_pc].plot(ax=ax2, color="blue", legend=False, zorder=2)

        # remove x-label
        ax1.set_xlabel("")
        ax2.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "pc-above-[{}]-[{}]-[{symbol}].png".format(
            column_name1, column_name2, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)

    def plot_pc_below(self, df, column_name1, column_name2, column_change_pc, symbol_name):
        # generate chart
        fig = plt.figure(figsize=(16, 10))
        plt.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        earliest_date = str(df[column_name1].head(1).index.date[0])
        latest_date = str(df[column_name1].tail(1).index.date[0])
        plt.title("[{symbol}] - % Change Of [{column1}] Below [{column2}] - [{earliest} ~ {latest}]".format(
            symbol=symbol_name, column1=column_name1, column2=column_name2,
            earliest=earliest_date, latest=latest_date))

        # plot column_name1 vs column_name2
        ax1 = fig.add_subplot(211)
        ax1.tick_params(axis="both", which="both", bottom=False, labelbottom=False)
        last_name1_value = df[column_name1].tail(1)[0]
        last_name2_value = df[column_name2].tail(1)[0]
        df[column_name1].plot(
            ax=ax1, label="{} ({:.4f})".format(column_name1, last_name1_value),
            color="green", legend=True, zorder=2)
        df[column_name2].plot(
            ax=ax1, label="{} ({:.4f})".format(column_name2, last_name2_value),
            color="blue", legend=True, zorder=2)

        # plot percent change
        ax2 = fig.add_subplot(212, ylabel="% Change")
        df[column_change_pc].plot(ax=ax2, color="blue", legend=False, zorder=2)

        # remove x-label
        ax1.set_xlabel("")
        ax2.set_xlabel("")

        # set major tick locator
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # set major tick labels with no rotation
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # turn on grid
        ax1.grid(color="lightgray", alpha=0.5, zorder=1)
        ax2.grid(color="lightgray", alpha=0.5, zorder=1)

        # save chart
        fig.tight_layout()
        output_file = os.path.join(self.chart_dir, "pc-below-[{}]-[{}]-[{symbol}].png".format(
            column_name1, column_name2, symbol=symbol_name))
        plt.savefig(output_file.lower(), format="png", bbox_inches="tight", transparent=False)
        plt.close(fig)
