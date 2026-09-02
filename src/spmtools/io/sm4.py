"""Reader for RHK Technology ``.sm4`` files (``STiMage 005.005`` layout).

An SM4 file is a tree of *objects*.  Every object list is a sequence of
``(object type, offset, size)`` triples of little-endian ``uint32`` that point at other
structures::

    file header
      +-- object list --> page index header (type 1)
                            +-- object list --> page index array (type 2)
                                                  +-- one entry per page, each with its
                                                      own object list:
                                                        +--> page header (type 3)
                                                        +--> page data   (type 4)

Only image pages (page data type 0) are decoded.  All integers are little-endian.  The
field layout follows the RHK SM4 specification and was cross-checked against Gwyddion's
``rhk-sm4.c`` and the ``spym`` reader.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

from spmtools.io import SpmFormatError

SIGNATURE_PREFIX = "STiMage"

OBJECT_PAGE_INDEX_HEADER = 1
OBJECT_PAGE_INDEX_ARRAY = 2
OBJECT_PAGE_HEADER = 3
OBJECT_PAGE_DATA = 4

DATA_TYPE_IMAGE = 0
PAGE_TYPE_TOPOGRAPHIC = 1
PAGE_TYPE_CURRENT = 2
SCAN_DIR_FORWARD = 0  # "right"; 1 = left (backward), 2 = up, 3 = down

#: ``line_type`` values whose samples are stored as float32 rather than int32.
FLOAT_LINE_TYPES = frozenset({1, 6, 9, 10, 11, 13, 18, 19, 21, 22})

# header_size, signature (18 UTF-16 chars), page_count, object_list_count,
# object_field_size, reserved x2
_FILE_HEADER = struct.Struct("<H36sIIIII")
_OBJECT = struct.Struct("<III")
# page_count, object_list_count, reserved x2
_PAGE_INDEX_HEADER = struct.Struct("<IIII")
# guid, data_type, source_type, object_list_count, minor_version
_PAGE_INDEX_ENTRY = struct.Struct("<16sIIII")
# field_size, string_count, page_type, data_sub_source, line_type, x_coord, y_coord,
# x_size, y_size, image_type, scan_type, group_id, data_size, min_z, max_z,
# x_scale, y_scale, z_scale, xy_scale, x_offset, y_offset, z_offset, period, bias,
# current, angle, color_info_count, grid_x_size, grid_y_size, object_list_count,
# data_flag, reserved flags (3), reserved (60)
_PAGE_HEADER = struct.Struct("<HHIIIiiIIIIIIiifffffffffffIIIIB3s60s")
PAGE_HEADER_SIZE = _PAGE_HEADER.size
_STRING_LENGTH = struct.Struct("<H")

# Index of well-known entries in the page string table.
STRING_LABEL = 0
STRING_X_UNIT = 7
STRING_Y_UNIT = 8
STRING_Z_UNIT = 9


@dataclass(frozen=True)
class Sm4Page:
    """Metadata of one page (one channel/direction of a scan, or a spectrum)."""

    index: int
    data_type: int
    source_type: int
    page_type: int
    line_type: int
    x_size: int
    y_size: int
    image_type: int
    scan_type: int
    x_scale: float
    y_scale: float
    z_scale: float
    x_offset: float
    y_offset: float
    z_offset: float
    period: float
    bias: float
    current: float
    angle: float
    header_offset: int
    data_offset: int
    data_size: int
    strings: tuple[str, ...] = field(default=())

    @property
    def is_image(self) -> bool:
        return self.data_type == DATA_TYPE_IMAGE

    @property
    def is_topography(self) -> bool:
        return self.page_type == PAGE_TYPE_TOPOGRAPHIC

    @property
    def is_forward(self) -> bool:
        return self.scan_type == SCAN_DIR_FORWARD

    @property
    def label(self) -> str:
        return self._string(STRING_LABEL)

    @property
    def z_unit(self) -> str:
        return self._string(STRING_Z_UNIT)

    @property
    def sample_dtype(self) -> np.dtype:
        return np.dtype("<f4" if self.line_type in FLOAT_LINE_TYPES else "<i4")

    def _string(self, index: int) -> str:
        return self.strings[index] if index < len(self.strings) else ""


def _read_exact(handle: BinaryIO, offset: int, size: int, what: str) -> bytes:
    handle.seek(offset)
    buf = handle.read(size)
    if len(buf) != size:
        raise SpmFormatError(f"unexpected end of file while reading {what}")
    return buf


def _read_objects(handle: BinaryIO, offset: int, count: int) -> dict[int, tuple[int, int]]:
    """Read ``count`` object triples; return ``{type: (offset, size)}`` (first wins)."""
    if count > 4096:
        raise SpmFormatError(f"implausible object list count {count}")
    buf = _read_exact(handle, offset, _OBJECT.size * count, "object list")
    objects: dict[int, tuple[int, int]] = {}
    for obj_type, obj_offset, obj_size in _OBJECT.iter_unpack(buf):
        objects.setdefault(obj_type, (obj_offset, obj_size))
    return objects


def _read_strings(handle: BinaryIO, offset: int, count: int) -> tuple[str, ...]:
    """Read ``count`` length-prefixed UTF-16LE strings starting at ``offset``."""
    strings: list[str] = []
    pos = offset
    for _ in range(count):
        (length,) = _STRING_LENGTH.unpack(_read_exact(handle, pos, 2, "string length"))
        pos += 2
        raw = _read_exact(handle, pos, 2 * length, "string")
        pos += 2 * length
        strings.append(raw.decode("utf-16-le", errors="replace"))
    return tuple(strings)


def _read_page(
    handle: BinaryIO,
    index: int,
    data_type: int,
    source_type: int,
    objects: dict[int, tuple[int, int]],
) -> Sm4Page:
    header_offset, _ = objects[OBJECT_PAGE_HEADER]
    data_offset, data_size_obj = objects[OBJECT_PAGE_DATA]
    (
        field_size,
        string_count,
        page_type,
        _data_sub_source,
        line_type,
        _x_coord,
        _y_coord,
        x_size,
        y_size,
        image_type,
        scan_type,
        _group_id,
        data_size,
        _min_z,
        _max_z,
        x_scale,
        y_scale,
        z_scale,
        _xy_scale,
        x_offset,
        y_offset,
        z_offset,
        period,
        bias,
        current,
        angle,
        _color_info_count,
        _grid_x,
        _grid_y,
        object_list_count,
        _flag,
        _reserved_flags,
        _reserved,
    ) = _PAGE_HEADER.unpack(_read_exact(handle, header_offset, PAGE_HEADER_SIZE, "page header"))
    # The page's own object list follows the fixed fields, then the string table.
    objects_start = header_offset + max(field_size, PAGE_HEADER_SIZE)
    strings_start = objects_start + object_list_count * _OBJECT.size
    try:
        strings = _read_strings(handle, strings_start, string_count)
    except SpmFormatError:
        strings = ()
    return Sm4Page(
        index=index,
        data_type=data_type,
        source_type=source_type,
        page_type=page_type,
        line_type=line_type,
        x_size=x_size,
        y_size=y_size,
        image_type=image_type,
        scan_type=scan_type,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        period=period,
        bias=bias,
        current=current,
        angle=angle,
        header_offset=header_offset,
        data_offset=data_offset,
        data_size=data_size or data_size_obj,
        strings=strings,
    )


def read_pages(handle: BinaryIO) -> list[Sm4Page]:
    """Walk the object tree of an open ``.sm4`` file and return all page descriptors."""
    header_size, signature, file_page_count, object_count, _field_size, _r1, _r2 = (
        _FILE_HEADER.unpack(_read_exact(handle, 0, _FILE_HEADER.size, "file header"))
    )
    text = signature.decode("utf-16-le", errors="replace").strip("\x00 ")
    if not text.startswith(SIGNATURE_PREFIX):
        raise SpmFormatError(f"not an RHK SM4 file (signature {text!r})")
    # ``header_size`` excludes its own two bytes; the object list follows the header.
    file_objects = _read_objects(handle, header_size + 2, object_count)
    if OBJECT_PAGE_INDEX_HEADER not in file_objects:
        raise SpmFormatError("missing page index header object")
    pih_offset, _ = file_objects[OBJECT_PAGE_INDEX_HEADER]
    index_page_count, index_object_count, _, _ = _PAGE_INDEX_HEADER.unpack(
        _read_exact(handle, pih_offset, _PAGE_INDEX_HEADER.size, "page index header")
    )
    index_objects = _read_objects(handle, pih_offset + _PAGE_INDEX_HEADER.size, index_object_count)
    if OBJECT_PAGE_INDEX_ARRAY not in index_objects:
        raise SpmFormatError("missing page index array object")
    pos, _ = index_objects[OBJECT_PAGE_INDEX_ARRAY]
    page_count = index_page_count or file_page_count

    pages: list[Sm4Page] = []
    for index in range(page_count):
        _guid, data_type, source_type, obj_count, _minor = _PAGE_INDEX_ENTRY.unpack(
            _read_exact(handle, pos, _PAGE_INDEX_ENTRY.size, "page index entry")
        )
        pos += _PAGE_INDEX_ENTRY.size
        objects = _read_objects(handle, pos, obj_count)
        pos += obj_count * _OBJECT.size
        if OBJECT_PAGE_HEADER not in objects or OBJECT_PAGE_DATA not in objects:
            continue
        pages.append(_read_page(handle, index, data_type, source_type, objects))
    return pages


def read_image(handle: BinaryIO, page: Sm4Page) -> np.ndarray:
    """Decode an image page into physical units as a float64 ``(y_size, x_size)`` array.

    Raw samples (int32, or float32 for the line types in :data:`FLOAT_LINE_TYPES`) are
    stored row by row.  A negative ``x_scale`` means the image was stored right-to-left
    and a positive ``y_scale`` that it was stored bottom-up; both are undone so that
    ``[0, 0]`` is the top-left corner.  Values are converted with
    ``z = raw * z_scale + z_offset``.
    """
    if not page.is_image:
        raise SpmFormatError(f"page {page.index} is not an image page (data type {page.data_type})")
    if page.x_size <= 0 or page.y_size <= 0:
        raise SpmFormatError(
            f"page {page.index} has an empty image ({page.x_size} x {page.y_size})"
        )
    count = page.x_size * page.y_size
    dtype = page.sample_dtype
    nbytes = count * dtype.itemsize
    if page.data_size < nbytes:
        raise SpmFormatError(
            f"page {page.index}: data block holds {page.data_size} bytes, need {nbytes}"
        )
    buf = _read_exact(handle, page.data_offset, nbytes, f"page {page.index} data")
    data = np.frombuffer(buf, dtype=dtype).reshape(page.y_size, page.x_size)
    if page.x_scale < 0:
        data = data[:, ::-1]
    if page.y_scale > 0:
        data = data[::-1, :]
    return data.astype(np.float64) * page.z_scale + page.z_offset


class Sm4File:
    """Reader for one ``.sm4`` file: page metadata is parsed once, images on demand."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as handle:
            self.pages = read_pages(handle)

    def image_pages(self) -> list[Sm4Page]:
        return [page for page in self.pages if page.is_image]

    def read_image(self, page: Sm4Page) -> np.ndarray:
        with open(self.path, "rb") as handle:
            return read_image(handle, page)

    def first_topography(self, *, prefer_forward: bool = True) -> Sm4Page | None:
        """The first topographic image page, preferring the forward scan direction."""
        candidates = [page for page in self.image_pages() if page.is_topography]
        if not candidates:
            return None
        if prefer_forward:
            for page in candidates:
                if page.is_forward:
                    return page
        return candidates[0]


__all__ = [
    "DATA_TYPE_IMAGE",
    "FLOAT_LINE_TYPES",
    "PAGE_HEADER_SIZE",
    "PAGE_TYPE_CURRENT",
    "PAGE_TYPE_TOPOGRAPHIC",
    "SCAN_DIR_FORWARD",
    "Sm4File",
    "Sm4Page",
    "read_image",
    "read_pages",
]
