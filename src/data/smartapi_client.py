"""Suppress SDK request dumps containing authentication headers and login payloads."""
from pathlib import Path
import logging
import logzero


class _SmartApiLogFilter(logging.Filter):
    def filter(self, record):
        return "smartapi" not in {part.lower() for part in Path(record.pathname).parts}


logzero.logger.addFilter(_SmartApiLogFilter())
from SmartApi import SmartConnect
