# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #

# Adapted from https://github.com/vllm-project/vllm/blob/0ce0539d4750f9ebcd9b19d7085ca3b934b9ec67/vllm/logger.py
"""Logging configuration for sllm."""

import logging
import os
import sys

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("sllm")
_default_handler = None
_logger_initialized = False


def _setup_logger():
    """Setup root logger only once. This is critical for Ray multi-process/actor
    environments to prevent duplicate log output (was causing 2x/3x duplication).
    """
    global _default_handler, _logger_initialized
    if _logger_initialized or _default_handler is not None:
        return
    _logger_initialized = True

    _root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to prevent duplicates from Ray worker inheritance
    for handler in list(_root_logger.handlers):
        _root_logger.removeHandler(handler)

    _default_handler = logging.StreamHandler(sys.stdout)
    _default_handler.flush = sys.stdout.flush  # type: ignore
    _default_handler.setLevel(logging.DEBUG)
    _root_logger.addHandler(_default_handler)

    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    _root_logger.propagate = False


_setup_logger()


def init_logger(name: str):
    """Initialize a named logger. Critical fix for Ray: always ensure only one handler
    exists and propagate=False. The SLLM_SUPPRESS_LOGS=1 env var (set on CPU routers)
    reduces CPU node INFO spam while control node (gpu_node) outputs fully.
    """
    logger = logging.getLogger(name)

    # Support suppressing logs from CPU node to achieve single overall output
    suppress_logs = os.getenv("SLLM_SUPPRESS_LOGS", "0").lower() in ("1", "true", "yes", "on")
    if suppress_logs:
        log_level = "WARNING"
    else:
        log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    logger.setLevel(log_level)

    if _default_handler:
        _default_handler.setLevel(log_level)

        # CRITICAL: Remove ALL existing handlers first (Ray workers often inherit duplicates)
        for h in list(logger.handlers):
            logger.removeHandler(h)

        # Add our controlled handler exactly once
        logger.addHandler(_default_handler)

    # Never propagate to root/parent loggers (this was a major source of duplication)
    logger.propagate = False

    return logger
