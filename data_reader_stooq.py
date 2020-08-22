import logging
import os
import pandas as pd
from pandas_datareader import data as pdr
import constants as c
from data_reader import DataReader


class StooqDataReader(DataReader):
    """
    Class for reading data from Stooq.
    """

    def __init__(self):
        super().__init__('stooq_data_reader')
        self.logger = logging.getLogger("algo-trader")

    def load(self, symbol_name, start_date, end_date, output_file):
        pickle_file = os.path.join(c.DATA_DIR, "{start}_{end}_{file}_stooq.pkl".format(
            start=start_date, end=end_date, file=output_file).lower())
        try:
            df = pd.read_pickle(pickle_file)
            self.logger.info("Existing file [{file}] found.".format(file=pickle_file))
        except FileNotFoundError:
            self.logger.info("File [{file}] not found... downloading [{symbol}] from [stooq].".format(
                file=pickle_file, symbol=symbol_name))
            df = pdr.StooqDailyReader(symbol_name, start_date, end_date).read()
            df.sort_index(inplace=True)  # stooq returns data in descending order; re-sort into ascending order
            df.to_pickle(pickle_file)
            actual_start = df.head(1).index
            actual_end = df.tail(1).index
            self.logger.info(symbol_name +
                             " : Requested [{requested_start} ~ {requested_end}]"
                             " / Actual [{actual_start} ~ {actual_end}]".format(
                                 requested_start=start_date, requested_end=end_date,
                                 actual_start=actual_start.date[0], actual_end=actual_end.date[0]))

        return df
