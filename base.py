import logging
import sys
import os

logging.basicConfig(
    format="[%(levelname)s] [%(asctime)s] [%(name)s] - %(message)s",
    level = logging.INFO,
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ])

class Base:
    """
    Implements predominantly logging for all classes in finder matcher
    """

    def __init__(self, subcls):
        self.logger = logging.getLogger(subcls)
        self.logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))