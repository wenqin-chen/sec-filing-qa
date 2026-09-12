"""secqa: grounded question answering over SEC filings with a reproducible evaluation harness."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("secqa")
except PackageNotFoundError:  # pragma: no cover - source checkout without an installed dist
    __version__ = "0.1.0"

__all__ = ["__version__"]
