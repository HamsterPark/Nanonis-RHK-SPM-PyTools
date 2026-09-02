"""Binary readers for scanning-probe-microscopy file formats."""

from __future__ import annotations


class SpmFormatError(ValueError):
    """Raised when a file does not follow the expected binary layout."""


__all__ = ["SpmFormatError"]
