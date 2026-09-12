"""Core: shared contracts, settings, numeric normalisation, ids, errors and logging.

Every other secqa module imports shared types from :mod:`secqa.core.contracts` and errors from
:mod:`secqa.core.errors`; nothing in ``core`` imports from any other secqa module.
"""

from secqa.core import contracts, errors, ids, logging, settings, textnum
from secqa.core.contracts import Frozen
from secqa.core.errors import (
    CalcRejected,
    CassetteMiss,
    ConfigError,
    IndexMismatch,
    ProviderError,
    ScenarioExhausted,
    SecqaError,
    SqlRejected,
)
from secqa.core.ids import chunk_id, request_id
from secqa.core.logging import configure_logging, get_logger
from secqa.core.settings import Settings, get_settings
from secqa.core.textnum import extract_numbers, normalize_text, numbers_equal, parse_number

__all__ = [
    "CalcRejected",
    "CassetteMiss",
    "ConfigError",
    "Frozen",
    "IndexMismatch",
    "ProviderError",
    "ScenarioExhausted",
    "SecqaError",
    "Settings",
    "SqlRejected",
    "chunk_id",
    "configure_logging",
    "contracts",
    "errors",
    "extract_numbers",
    "get_logger",
    "get_settings",
    "ids",
    "logging",
    "normalize_text",
    "numbers_equal",
    "parse_number",
    "request_id",
    "settings",
    "textnum",
]
