"""Centralised logging setup for MuRaL.

Three output channels, each with independent level filters:

    stdout              — INFO+  (always visible)
    training.log        — INFO+  (clean, production — always)
    training_debug.log  — DEBUG+ (verbose, only when --debug)

When --debug is passed, all DEBUG messages go to the separate debug log
AND to stdout (with function/line info).  The main training.log stays
clean (INFO only).

Usage::

    from MuRaL.utils.log_utils import setup_logging, get_logger

    setup_logging(logdir='./results/exp/trial_01', debug=True)
    logger = get_logger(__name__)
    logger.info("training started")
    logger.debug("batch 1000 timing: 1.2s")  # → only in debug log + stdout
"""

import logging
import sys
import os


LOG_FORMAT = '%(asctime)s [%(levelname)-5s] %(name)s: %(message)s'
LOG_FORMAT_DEBUG = ('%(asctime)s [%(levelname)-5s] %(name)s|%(funcName)s:%(lineno)d: '
                    '%(message)s')
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

_configured = False


def setup_logging(logdir=None, debug=False, use_ray=False):
    """Configure MuRaL logging once.  Idempotent — safe to call multiple times.

    Args:
        logdir:  directory for log files (None or use_ray=True = stdout only).
        debug:   if True, also create training_debug.log and show DEBUG
                 messages on stdout (with function / line number).
        use_ray: if True, skip file handlers (Ray captures stdout).
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger('mural')
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.handlers.clear()
    root.propagate = False

    info_fmt = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    debug_fmt = logging.Formatter(LOG_FORMAT_DEBUG, DATE_FORMAT)

    # ── stdout: INFO+ (normal) or DEBUG+ (--debug) ──────────────────
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(debug_fmt if debug else info_fmt)
    stdout_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(stdout_handler)

    if logdir and not use_ray:
        os.makedirs(logdir, exist_ok=True)

        # ── training.log: INFO+  (always clean) ────────────────────
        _add_file_handler(root, os.path.join(logdir, 'training.log'),
                          info_fmt, logging.INFO)

        # ── training_debug.log: DEBUG+  (only when --debug) ────────
        if debug:
            _add_file_handler(root, os.path.join(logdir, 'training_debug.log'),
                              debug_fmt, logging.DEBUG)

    root.info("Logging configured (debug=%s, logdir=%s)", debug, logdir or 'none')


def _add_file_handler(root, path, fmt, level):
    """Add a file handler, silently falling back if creation fails."""
    try:
        h = logging.FileHandler(path, mode='a', encoding='utf-8')
        h.setFormatter(fmt)
        h.setLevel(level)
        root.addHandler(h)
    except OSError:
        root.warning("Cannot create log file: %s (continuing with stdout only)", path)


def get_logger(name='mural'):
    """Get a logger for a specific sub-module.

    The name is always prefixed with 'mural.' so output routes to the
    handlers configured by setup_logging().

    >>> logger = get_logger(__name__)
    >>> logger.info("hello")
    """
    if name == 'mural':
        return logging.getLogger('mural')
    return logging.getLogger('mural.' + name)
