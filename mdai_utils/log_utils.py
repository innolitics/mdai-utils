import gzip
import logging
import os
import shutil
from logging.handlers import RotatingFileHandler
from pathlib import Path


def rotating_zip_file_handler(log_file, maxBytes, backupCount):
    """
    Create a RotatingFileHandler that compresses the rotated files.
    Recommended to have a large backupCount, to avoid deleting the log files.
    Instead we compress them.
    """

    def namer(name):
        return name + ".gz"

    def rotator(source, dest):
        with open(source, "rb") as f_in:
            with gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)  # type: ignore
        os.remove(source)

    file_handler = RotatingFileHandler(
        log_file, mode="a", maxBytes=maxBytes, backupCount=backupCount
    )
    file_handler.namer = namer
    file_handler.rotator = rotator
    return file_handler


def set_dual_logger(logger_name, logs_dir, log_level=logging.INFO, verbose=True):
    """
    Set a logger that writes to a file and to the console.

    The file is rotated when it reaches 250 MB. The rotated files are compressed.
    The logs directory is at the root of the project if not in a docker container,
    otherwise it is in the processing directory / logs.

    logger_name: str, name of the logger
    logs_dir: str or Path, should point to processing_directory/logs
    log_level: logging level, defaults to INFO
    verbose: bool, prints to console the path of the log file
    """

    MAX_LOG_SIZE = 250 * 1024 * 1024  # 250 MB in bytes
    BACKUP_COUNT = 999999999  # number of rotated files to keep ({log_file}.1, .2, ...)

    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True)
    log_file = logs_dir / f"{logger_name}.log"
    file_handler = rotating_zip_file_handler(
        log_file=log_file,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    if verbose:
        print("Logging into", log_file)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
