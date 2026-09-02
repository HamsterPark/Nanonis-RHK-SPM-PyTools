"""Shared fixtures: writers for small synthetic ``.sxm`` and ``.sm4`` files.

The writers follow the file-format descriptions in :mod:`spmtools.io.sxm` and
:mod:`spmtools.io.sm4` independently of the reader code, so the tests check the readers
against the documented layout rather than against themselves.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

Channel = tuple[str, str, str]  # (name, unit, direction)


def sxm_header_text(
    channels: list[Channel],
    nx: int,
    ny: int,
    *,
    scan_dir: str = "down",
    scan_range: tuple[float, float] | None = (2.0e-8, 1.5e-8),
    byteorder: str = "MSBFIRST",
    comment: str = "synthetic test file",
) -> str:
    lines = [
        ":NANONIS_VERSION:",
        "2",
        ":SCANIT_TYPE:",
        f"              FLOAT               {byteorder}",
        ":REC_DATE:",
        " 01.01.2026",
        ":SCAN_PIXELS:",
        f"       {nx}       {ny}",
    ]
    if scan_range is not None:
        lines += [":SCAN_RANGE:", f"           {scan_range[0]:E}           {scan_range[1]:E}"]
    lines += [
        ":SCAN_DIR:",
        scan_dir,
        ":Z-CONTROLLER:",
        "\tName\ton\tSetpoint\tP-gain\tI-gain\tT-const",
        "\tCurrent\t1\t1.000E-10 A\t1.000E-12 m\t2.000E-8 m/s\t5.000E-5 s",
        ":COMMENT:",
        comment,
        ":DATA_INFO:",
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset",
    ]
    for number, (name, unit, direction) in enumerate(channels):
        lines.append(f"\t{number}\t{name}\t{unit}\t{direction}\t1.000E+0\t0.000E+0")
    lines += ["", ":SCANIT_END:", "", ""]
    return "\n".join(lines)


def write_sxm(
    path: Path,
    channels: list[Channel],
    images: dict[tuple[str, str], np.ndarray],
    *,
    scan_dir: str = "down",
    scan_range: tuple[float, float] | None = (2.0e-8, 1.5e-8),
    byteorder: str = "MSBFIRST",
    truncate: int = 0,
) -> Path:
    """Write a synthetic ``.sxm`` file.

    ``images`` maps ``(channel name, "forward" | "backward")`` to the array *as stored*
    (no orientation applied).  Every image a channel's direction requires must be given.
    """
    shapes = {img.shape for img in images.values()}
    assert len(shapes) == 1, "all images must share one shape"
    ny, nx = shapes.pop()
    header = sxm_header_text(
        channels, nx, ny, scan_dir=scan_dir, scan_range=scan_range, byteorder=byteorder
    )
    dtype = np.dtype(">f4" if byteorder == "MSBFIRST" else "<f4")
    payload = bytearray()
    for name, _unit, direction in channels:
        if direction in ("both", "fwd"):
            payload += np.ascontiguousarray(images[(name, "forward")], dtype=dtype).tobytes()
        if direction in ("both", "bwd"):
            payload += np.ascontiguousarray(images[(name, "backward")], dtype=dtype).tobytes()
    data = header.encode("latin-1") + b"\x1a\x04" + bytes(payload)
    if truncate:
        data = data[:-truncate]
    path.write_bytes(data)
    return path


def ramp(ny: int, nx: int, offset: float = 0.0) -> np.ndarray:
    """A deterministic non-symmetric test image."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (yy * 100 + xx + offset).astype(np.float64)


def _utf16(text: str) -> bytes:
    return text.encode("utf-16-le")


def _sm4_string(text: str) -> bytes:
    return struct.pack("<H", len(text)) + _utf16(text)


def write_sm4(path: Path, pages: list[dict[str, Any]]) -> Path:
    """Write a synthetic ``.sm4`` file with the given image pages.

    Each page dict may contain: ``data`` (2-D array as stored, shape ``(y_size, x_size)``,
    required), ``page_type`` (default 1 = topography), ``scan_type`` (default 0 =
    forward), ``line_type`` (default 0), ``data_type`` (default 0 = image), ``x_scale``,
    ``y_scale``, ``z_scale``, ``z_offset``, ``label`` and ``z_unit``.
    """
    file_header_size = 2 + 36 + 5 * 4  # 58 bytes including the 2-byte size field
    object_size = 12
    page_index_header_offset = file_header_size + object_size
    page_index_array_offset = page_index_header_offset + 16 + object_size
    entry_size = 16 + 16 + 2 * object_size
    body_offset = page_index_array_offset + entry_size * len(pages)

    blobs: list[bytes] = []
    entries: list[bytes] = []
    cursor = body_offset
    for spec in pages:
        data = np.asarray(spec["data"])
        y_size, x_size = data.shape
        line_type = spec.get("line_type", 0)
        dtype = np.dtype("<f4") if line_type in {1, 6, 9, 10, 11, 13, 18, 19, 21, 22} else "<i4"
        raw = np.ascontiguousarray(data, dtype=dtype).tobytes()
        strings = [spec.get("label", "Topography")] + [""] * 8 + [spec.get("z_unit", "m")]
        string_table = b"".join(_sm4_string(s) for s in strings)
        header = struct.pack(
            "<HHIIIiiIIIIIIiifffffffffffIIIIB3s60s",
            180,
            len(strings),
            spec.get("page_type", 1),
            0,
            line_type,
            0,
            0,
            x_size,
            y_size,
            0,
            spec.get("scan_type", 0),
            0,
            len(raw),
            0,
            0,
            spec.get("x_scale", 1.0e-9),
            spec.get("y_scale", -1.0e-9),
            spec.get("z_scale", 1.0e-12),
            0.0,
            0.0,
            0.0,
            spec.get("z_offset", 0.0),
            0.0,
            spec.get("bias", 0.0),
            spec.get("current", 0.0),
            spec.get("angle", 0.0),
            0,
            0,
            0,
            0,  # no page-level objects
            0,
            b"\0" * 3,
            b"\0" * 60,
        )
        header_blob = header + string_table
        header_offset = cursor
        data_offset = header_offset + len(header_blob)
        cursor = data_offset + len(raw)
        blobs.append(header_blob + raw)
        entries.append(
            struct.pack("<16sIIII", b"\0" * 16, spec.get("data_type", 0), 0, 2, 0)
            + struct.pack("<III", 3, header_offset, len(header_blob))
            + struct.pack("<III", 4, data_offset, len(raw))
        )

    signature = _utf16("STiMage 005.005 1").ljust(36, b"\0")
    file_header = struct.pack(
        "<H36sIIIII", file_header_size - 2, signature, len(pages), 1, object_size, 0, 0
    )
    file_objects = struct.pack("<III", 1, page_index_header_offset, 16)
    page_index_header = struct.pack("<IIII", len(pages), 1, 0, 0)
    pih_objects = struct.pack("<III", 2, page_index_array_offset, entry_size * len(pages))
    out = file_header + file_objects + page_index_header + pih_objects + b"".join(entries)
    assert len(out) == body_offset
    path.write_bytes(out + b"".join(blobs))
    return path


@pytest.fixture
def simple_sxm(tmp_path: Path) -> Path:
    """One ``down`` scan with Z (both), Current (both) and a forward-only Freq Shift."""
    channels: list[Channel] = [
        ("Z", "m", "both"),
        ("Current", "A", "both"),
        ("OC_M1_Freq._Shift", "Hz", "fwd"),
    ]
    ny, nx = 6, 8
    images = {
        ("Z", "forward"): ramp(ny, nx, 0),
        ("Z", "backward"): ramp(ny, nx, 1000),
        ("Current", "forward"): ramp(ny, nx, 2000),
        ("Current", "backward"): ramp(ny, nx, 3000),
        ("OC_M1_Freq._Shift", "forward"): ramp(ny, nx, 4000),
    }
    return write_sxm(tmp_path / "sample_0001.sxm", channels, images)
