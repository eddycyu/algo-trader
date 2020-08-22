import logging
import os
import pandas as pd
from pandas_datareader import data as pdr
import constants as c


class DataReader(object):
    """
    Base class for data readers (e.g. YahooDataReader, StooqDataReader, etc.)
    """

    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(__name__)

        # check if the data directory exists; if not, make it
        if not os.path.exists(c.DATA_DIR):
            os.makedirs(c.DATA_DIR)

    def _load(self, symbol_name, source, start_date, end_date, output_file):
        pickle_file = os.path.join(
            c.DATA_DIR, str(start_date) + "_" + str(end_date) + "_" + output_file + "_" + source + ".pkl").lower()
        try:
            df = pd.read_pickle(pickle_file)
            self.logger.info("Existing file [{file}] found.".format(file=pickle_file))
        except FileNotFoundError:
            self.logger.info("File [{file}] not found... downloading [{symbol}] from [{source}].".format(
                file=pickle_file, symbol=symbol_name, source=source))
            df = pdr.DataReader(symbol_name, source, start_date, end_date)
            df.to_pickle(pickle_file)
            actual_start = df.head(1).index
            actual_end = df.tail(1).index
            self.logger.info(symbol_name + " : [" + str(actual_start.date[0]) + "," + str(actual_end.date[0]) + "]")

        return df

    def load(self, symbol_name, start_date, end_date, output_file):
        self.logger.error("You should override this method!")
