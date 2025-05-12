import logging
import os
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

def get_timestamp():
    """Generate a standardized timestamp string for the application.
    Format: YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_logging(results_dir, tensorboard_logging=False):
    """Configure logging for the brain age prediction application.

    Args:
        results_dir (str): Directory where log files will be stored
        tensorboard_logging (bool): Whether to enable tensorboard logging
    """
    logger = logging.getLogger('brainage')
    logger.setLevel(logging.DEBUG)

    # Create formatters with standardized datetime format
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y%m%d_%H%M%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Create and configure file handler
    log_file = os.path.join(results_dir, 'training.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial setup information
    logger.info(f"Logging initialized - Log file: {log_file}")
    logger.info(f"TensorBoard logging enabled: {tensorboard_logging}")

    return logger

class TensorBoardFilter(logging.Filter):
    """Filter to control TensorBoard logging output"""
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled

    def filter(self, record):
        if not self.enabled and 'tensorboard' in record.name.lower():
            return False
        return True
