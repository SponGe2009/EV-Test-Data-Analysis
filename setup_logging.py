import logging


def setup_logging():
    # Create a logger
    # 'AppLogger' is a custom name for your logger
    logger = logging.getLogger('AppLogger')
    logger.setLevel(logging.DEBUG)  # Set the minimum logging level

    # Create file handler which logs even debug messages
    fh = logging.FileHandler('./log/logs.log')
    fh.setLevel(logging.DEBUG)  # Set the minimum logging level for the file

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)  # Only log errors and above to the console

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def handle_exception(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger('AppLogger')
    logger.error("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))
