import logging
from data_reader import DataReader


class YahooDataReader(DataReader):
    """
    Class for reading data from Yahoo.
    """

    def __init__(self):
        super().__init__('yahoo_data_reader')
        self.logger = logging.getLogger("algo-trader")

    def load(self, symbol_name, start_date, end_date, output_file):
        return self._load(symbol_name, "yahoo", start_date, end_date, output_file)
