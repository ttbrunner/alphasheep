"""
A logging architecture that allows for easy creation of Loggers, even when run from different __main__s and with different modules.
This is similar to the old log4j / log4net structure.

I feel like this is not the best way to do things... I'm happy if you can suggest something more elegant.
"""

import logging

ROOT_NAME = ""


def init_logging():
    """
    Initializes a root logger and defines the log outputs. The root logger (and all others) have a default loglevel of INFO.
    """

    try:
        import absl.logging
        # TensorFlow uses Abseil logging, which interferes with our logging. Apparently they have fixed it, but it's not yet live.
        # Using this workaround in the meantime.
        # https://github.com/abseil/abseil-py/issues/99
        # https://github.com/abseil/abseil-py/issues/102
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
    except Exception:
        pass

    # Get root logger
    logger = logging.getLogger(ROOT_NAME)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Init console output
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Initialized logging.")

    # Don't return the root logger - everybody should use their own class logger.


def get_class_logger(c):
    # Can provide either a class or an instance of it (lazy)
    if c.__class__ is not type:
        c = c.__class__

    # Only using class name now, not the class+module name as that just clutters the output.
    logger_name = ROOT_NAME + "." if len(ROOT_NAME) > 0 else ""
    logger_name += c.__name__   # full_class_name(c)
    return logging.getLogger(logger_name)


def get_named_logger(name):
    # Custom name: For everything that's not inside a class
    logger_name = ROOT_NAME + "." if len(ROOT_NAME) > 0 else ""
    logger_name += name
    return logging.getLogger(logger_name)


def full_class_name(c: type):
    # Concats module and class name. This can get pretty long!
    module = c.__module__
    if module is None or module == str.__class__.__module__:
        return c.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + c.__name__
