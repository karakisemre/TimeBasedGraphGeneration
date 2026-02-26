import logging

# Let's define ANSI escape codes for colors
class LogColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
    CRITICAL = '\033[95m'  # A shade above dark red/a color for burgundy

class CustomFormatter(logging.Formatter):
    format_dict = {
        logging.INFO: LogColors.BLUE + "%(asctime)s - %(levelname)s - %(message)s" + LogColors.ENDC,
        logging.DEBUG: LogColors.GREEN + "%(asctime)s - %(levelname)s - %(message)s" + LogColors.ENDC,
        logging.WARNING: LogColors.WARNING + "%(asctime)s - %(levelname)s - %(message)s" + LogColors.ENDC,
        logging.ERROR: LogColors.FAIL + "%(asctime)s - %(levelname)s - %(message)s" + LogColors.ENDC,
        logging.CRITICAL: LogColors.CRITICAL + "%(asctime)s - %(levelname)s - %(message)s" + LogColors.ENDC,
    }

    def format(self, record):
        log_fmt = self.format_dict.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        try:
            return formatter.format(record)
        except TypeError:
            msg = str(record.msg)
            if record.args:
                msg = msg + " " + " ".join(map(str, record.args))
            record.msg = msg
            record.args = ()
            return formatter.format(record)


class Logger:
    def __init__(self):
        self.logger = self.configure_logger()

    def configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        #logger.setLevel(logging.WARNING)

        # Use CustomFormatter for console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CustomFormatter())

        file_handler = logging.FileHandler('app.log')
        file_handler.setLevel(logging.INFO)
        # Use a simple formatter for the file (without color)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def __getattr__(self, attr):
        """

        Thanks to this magic method, the methods called from the Logger class (for example logger.info, logger.debug) are directed directly to the self.logger object.
        Thus, the Logger class acts as if it supports all the methods of the logger object in it.

        """
        return getattr(self.logger, attr)
