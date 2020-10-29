import logging
import os
import constants as c


class DB(object):
    """
    Base class for databases (e.g. SQLiteDB, etc.)
    """

    def __init__(self, db_dir=c.CHART_PA_DIR):
        self.db_dir = db_dir
        self.logger = logging.getLogger("algo-trader")

        # check if the database directory exists; if not, make it
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

    def connect(self, db_filename):
        self.logger.error("You should override this method!")

    def disconnect(self, connection):
        self.logger.error("You should override this method!")

    def create_tables(self, connection):
        self.logger.error("You should override this method!")
