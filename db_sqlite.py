import logging
import os
import sqlite3
from db import DB


class SQLiteDB(DB):
    """
    Class for SQLite database.
    """

    def __init__(self, db_dir):
        super().__init__(db_dir)
        self.logger = logging.getLogger("algo-trader")
        # register adapter and converter
        #sqlite3.register_adapter(bool, int)
        #sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))

    def connect(self, db_filename):
        try:
            db_file = os.path.join(self.db_dir, db_filename)
            connection = sqlite3.connect(db_file)
            self.logger.info("Connected to database [{db}].".format(db=db_filename))
            return connection
        except sqlite3.Error as error:
            self.logger.error("Failed to connect to database [{db}]!".format(db=db_filename), error)

    def disconnect(self, connection):
        try:
            connection.close()
            self.logger.info("Closed connection to database.")
        except sqlite3.Error as error:
            self.logger.error("Failed to close connection to database!", error)

    def create_tables(self, connection):
        sql_create_data = "CREATE TABLE IF NOT EXISTS data (" \
                          "symbol TEXT NOT NULL, " \
                          "date datetime NOT NULL, " \
                          "Open REAL NOT NULL, " \
                          "High REAL NOT NULL, " \
                          "Low REAL NOT NULL, " \
                          "Close REAL NOT NULL, " \
                          "Volume INTEGER NOT NULL, " \
                          "u_close REAL NOT NULL, " \
                          "u_daily_change REAL NOT NULL, " \
                          "u_daily_change_pc REAL NOT NULL, " \
                          "u_open_prev_close REAL NOT NULL, " \
                          "u_open_prev_close_pc REAL NOT NULL, " \
                          "u_52_wk_low REAL NOT NULL, " \
                          "u_52_wk_high REAL NOT NULL, " \
                          "u_close_above_52_wk_low REAL NOT NULL, " \
                          "u_close_below_52_wk_high REAL NOT NULL, " \
                          "u_sma-50 REAL NOT NULL, " \
                          "u_sma-100 REAL NOT NULL, " \
                          "u_sma-200 REAL NOT NULL, " \
                          "u_ema-12 REAL NOT NULL, " \
                          "u_ema-26 REAL NOT NULL, " \
                          "u_ema_golden-12-26 INTEGER NOT NULL, " \
                          "u_ema_death-12-26 INTEGER NOT NULL, " \
                          "u_ema-50 REAL NOT NULL, " \
                          "u_ema-200 REAL NOT NULL, " \
                          "u_ema_golden-50-200 INTEGER NOT NULL, " \
                          "u_ema_death-50-200 INTEGER NOT NULL, " \
                          "u_bb-sma-20 REAL NOT NULL, " \
                          "u_bb-upper-20 REAL NOT NULL, " \
                          "u_bb-lower-20 REAL NOT NULL, " \
                          "u_macd-12-26-9 REAL NOT NULL, " \
                          "u_macd_signal-12-26-9 REAL NOT NULL, " \
                          "u_macd-histogram-12-26-9 REAL NOT NULL, " \
                          "u_macd-50-200-9 REAL NOT NULL, " \
                          "u_macd_signal-50-200-9 REAL NOT NULL, " \
                          "u_macd_histogram-50-200-9 REAL NOT NULL, " \
                          "u_rs_avg_gain-14 REAL NOT NULL, " \
                          "u_rs_avg_loss-14 REAL NOT NULL, " \
                          "u_rsi-14 REAL NOT NULL, " \
                          "PRIMARY KEY(symbol, date))"

        cursor = connection.cursor()
        try:
            cursor.execute(sql_create_data)
            self.logger.info("Create data table in database.")
        except sqlite3.Error as error:
            self.logger.error("Failed to create data table in database!", error)
        finally:
            cursor.close()
