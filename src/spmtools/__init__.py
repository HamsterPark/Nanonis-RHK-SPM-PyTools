"""spmtools: preview, align and crop tools for Nanonis SXM and RHK SM4 SPM data.

Public entry points:

* :mod:`spmtools.io` - binary readers for ``.sxm`` (Nanonis) and ``.sm4`` (RHK) files
* :mod:`spmtools.preview` - per-channel mosaic previews of whole datasets
* :mod:`spmtools.align` - sub-pixel registration and difference of two scans
* :mod:`spmtools.crop` - removal of solid-colour (unscanned) borders from exported images
"""

from __future__ import annotations

import os
import sys

__version__ = "1.0.0"


def _disable_windows_wmi() -> None:
    """Opt-in workaround for ``import numpy`` hanging on some Windows machines.

    ``platform.machine()`` (called during the NumPy import) goes through a WMI query
    that can block forever when the Winmgmt service is unhealthy.  Setting the
    environment variable ``SPMTOOLS_DISABLE_WMI=1`` makes :mod:`platform` fall back to
    ``sys.getwindowsversion()`` instead.  This must run before NumPy is imported, which
    is why it lives in the package ``__init__``.
    """
    import platform

    platform._wmi = None  # type: ignore[attr-defined]

    def _wmi_query(*_args: object, **_kwargs: object) -> str:
        raise OSError("WMI disabled by SPMTOOLS_DISABLE_WMI")

    platform._wmi_query = _wmi_query  # type: ignore[attr-defined]


if sys.platform == "win32" and os.environ.get("SPMTOOLS_DISABLE_WMI") == "1":
    _disable_windows_wmi()

__all__ = ["__version__"]
