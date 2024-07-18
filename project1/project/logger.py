import logging.config
import os
from colorama import init

init()


class LoggerFormatter:
    CRITICAL = "\033[1;31m"
    ERROR = "\033[0;31m"
    WARNING = "\033[0;33m"
    INFO = "\033[0;37m"
    DEBUG = "\033[1;37m"
    NOTSET = "\033[0;37m"
    RESET = "\033[0m"
    format = "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.CRITICAL: CRITICAL + format + RESET,
        logging.ERROR: ERROR + format + RESET,
        logging.WARNING: WARNING + format + RESET,
        logging.INFO: INFO + format + RESET,
        logging.DEBUG: DEBUG + format + RESET,
        logging.NOTSET: NOTSET + format + RESET
    }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)


class Logger:
    logger = logging.getLogger('Project1')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(LoggerFormatter())
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.INFO)
    logLength = 80

    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    def __init__(self):
        if "LOG_LEVEL" in os.environ and hasattr(Logger, os.environ["LOG_LEVEL"]):
            self.logger.setLevel(logging.getLevelName(os.environ["LOG_LEVEL"]))

    @staticmethod
    def print_line_of_char(char, level=INFO):
        Logger.logger.log(level, char * Logger.logLength)

    @staticmethod
    def print_var(var_name, var, level=INFO):
        separator = ": "
        var_name_length = len(var_name)
        wording_length = int(Logger.logLength / 2)
        var_name_rem_length = wording_length - var_name_length
        rem_length = Logger.logLength - var_name_length - len(separator)
        Logger.logger.log(level, f"%-{wording_length}s{separator}%-{rem_length}s", var_name + "." * var_name_rem_length,
                          var)

    @staticmethod
    def print_stat(stat, stat_value, level=INFO):
        separator = ":"
        stat_length = len(stat)
        rem_length = Logger.logLength - stat_length - len(separator)
        Logger.logger.log(level, f"%-{stat_length}s{separator}%{rem_length}s", stat, stat_value)

    @staticmethod
    def print_center(msg, delimiter="|", level=INFO):
        delimiter_length = len(delimiter) + 1
        max_msg_length = Logger.logLength - (2 * delimiter_length)
        Logger.logger.log(level, f"%-{delimiter_length}s%{max_msg_length}s%{delimiter_length}s", delimiter,
                          msg.center(max_msg_length), delimiter)

    @staticmethod
    def print(msg, level=INFO, **kwargs):
        Logger.logger.log(level, msg, **kwargs)

    @staticmethod
    def print_error(msg):
        Logger.logger.exception(msg)


logger = Logger.logger
