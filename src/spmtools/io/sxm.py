"""Reader for Nanonis ``.sxm`` scan files.

File layout as written by the Nanonis SPM controller::

    :NANONIS_VERSION:
    2
    :SCANIT_TYPE:
                  FLOAT               MSBFIRST
    ...
    :DATA_INFO:
    <tab>Channel<tab>Name<tab>Unit<tab>Direction<tab>Calibration<tab>Offset
    <tab>14<tab>Z<tab>m<tab>both<tab>-3.900E-9<tab>0.000E+0

    :SCANIT_END:

    \\x1a\\x04<binary images>

* The header is a sequence of ``:TAG:`` lines, each followed by its value line(s), and ends
  with ``:SCANIT_END:``.  The binary block starts right after the ``\\x1a\\x04`` marker.
* ``:SCANIT_TYPE:`` gives the sample type and byte order (``FLOAT MSBFIRST`` is big-endian
  float32, the only sample type Nanonis writes).
* ``:SCAN_PIXELS:`` gives ``nx ny``; ``:SCAN_RANGE:`` the physical size in metres.
* ``:DATA_INFO:`` lists every recorded channel with its unit and direction (``both``,
  ``fwd`` or ``bwd``).  Images follow in that order, ``ny * nx`` samples each; a ``both``
  channel stores its forward image first, then the backward one.
* Lines are stored in acquisition order.  For ``:SCAN_DIR: down`` the first stored line is
  the top of the image, for ``up`` it is the bottom.  Backward images are stored
  right-to-left.  :func:`orient_image` undoes both so that ``array[0, 0]`` is the top-left
  corner of the scan frame for every channel and direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np

from spmtools.io import SpmFormatError

DATA_MARKER = b"\x1a\x04"
HEADER_END = b":SCANIT_END:"
FIRST_TAG = b":NANONIS_VERSION:"
_CHUNK = 64 * 1024
_MAX_HEADER = 32 * 1024 * 1024

Direction = Literal["forward", "backward"]
DIRECTIONS: tuple[Direction, ...] = ("forward", "backward")


@dataclass(frozen=True)
class SxmChannel:
    """One row of the ``DATA_INFO`` table."""

    index: int
    """Position in the ``DATA_INFO`` table (0-based)."""
    name: str
    unit: str
    direction: str
    """``both``, ``fwd`` or ``bwd`` as written by Nanonis."""
    image_index: int
    """Index of the first image of this channel in the binary block."""

    @property
    def has_forward(self) -> bool:
        return self.direction in ("both", "fwd")

    @property
    def has_backward(self) -> bool:
        return self.direction in ("both", "bwd")

    def image_number(self, direction: Direction) -> int:
        """Index of the requested image in the binary block."""
        if direction == "forward":
            if not self.has_forward:
                raise KeyError(f"channel {self.name!r} has no forward image")
            return self.image_index
        if direction == "backward":
            if not self.has_backward:
                raise KeyError(f"channel {self.name!r} has no backward image")
            return self.image_index + (1 if self.direction == "both" else 0)
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")


@dataclass(frozen=True)
class SxmHeader:
    """Parsed header of an ``.sxm`` file."""

    version: str
    nx: int
    ny: int
    scan_range_m: tuple[float, float] | None
    """Physical scan size ``(x, y)`` in metres, or ``None`` when ``SCAN_RANGE`` is absent."""
    scan_dir: str
    dtype: np.dtype
    channels: tuple[SxmChannel, ...]
    data_offset: int
    """Absolute byte offset of the first image sample."""
    tags: dict[str, list[str]]
    """Every header tag with its raw value lines."""

    @property
    def n_images(self) -> int:
        return sum(2 if ch.direction == "both" else 1 for ch in self.channels)

    @property
    def pixel_size_nm(self) -> tuple[float, float]:
        """Pixel pitch ``(x, y)`` in nanometres."""
        if self.scan_range_m is None:
            raise SpmFormatError("header has no SCAN_RANGE tag")
        return (self.scan_range_m[0] / self.nx * 1e9, self.scan_range_m[1] / self.ny * 1e9)

    def find_channel(self, keyword: str) -> SxmChannel:
        """Find a channel by name.

        An exact, case-insensitive name match wins.  Otherwise every whitespace-separated
        word of ``keyword`` must occur in the channel name (``"Freq Shift"`` matches
        ``"OC_M1_Freq._Shift"``).  The first channel in ``DATA_INFO`` order is returned.
        """
        wanted = keyword.strip().lower()
        for ch in self.channels:
            if ch.name.lower() == wanted:
                return ch
        words = wanted.split()
        for ch in self.channels:
            lowered = ch.name.lower()
            if words and all(word in lowered for word in words):
                return ch
        available = ", ".join(ch.name for ch in self.channels) or "<none>"
        raise KeyError(f"no channel matches {keyword!r}; available: {available}")


def parse_header_text(text: str) -> dict[str, list[str]]:
    """Split the ASCII header into ``{tag: [value line, ...]}``.

    A line that starts and ends with ``:`` opens a new tag; all following lines up to the
    next tag are its value.
    """
    tags: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) > 2 and stripped.startswith(":") and stripped.endswith(":"):
            current = stripped[1:-1]
            tags[current] = []
        elif current is not None:
            tags[current].append(line)
    return tags


def parse_data_info(rows: list[str]) -> tuple[SxmChannel, ...]:
    """Turn the tab-separated ``DATA_INFO`` rows into :class:`SxmChannel` objects."""
    table = []
    for row in rows:
        cells = [cell.strip() for cell in row.split("\t")]
        cells = [cell for cell in cells if cell]
        if cells:
            table.append(cells)
    if not table:
        return ()
    head = [cell.lower() for cell in table[0]]
    if "name" in head:
        body = table[1:]
        name_col = head.index("name")
        unit_col = head.index("unit") if "unit" in head else -1
        dir_col = head.index("direction") if "direction" in head else -1
    else:  # no header row: assume the standard column order
        body = table
        name_col, unit_col, dir_col = 1, 2, 3
    channels: list[SxmChannel] = []
    image_index = 0
    for index, cells in enumerate(body):
        if len(cells) <= max(name_col, dir_col):
            raise SpmFormatError(f"malformed DATA_INFO row: {cells}")
        direction = cells[dir_col].lower() if dir_col >= 0 else "both"
        if direction not in ("both", "fwd", "bwd"):
            raise SpmFormatError(f"unknown scan direction {direction!r} in DATA_INFO")
        unit = cells[unit_col] if 0 <= unit_col < len(cells) else ""
        channels.append(SxmChannel(index, cells[name_col], unit, direction, image_index))
        image_index += 2 if direction == "both" else 1
    return tuple(channels)


def _read_header_bytes(handle: BinaryIO) -> tuple[bytes, int]:
    """Read up to the data marker; return the raw header and the data offset."""
    buf = bytearray()
    while True:
        chunk = handle.read(_CHUNK)
        if chunk:
            buf += chunk
        if len(buf) >= len(FIRST_TAG) and not buf.startswith(FIRST_TAG):
            raise SpmFormatError("not a Nanonis SXM file (missing :NANONIS_VERSION:)")
        end = buf.find(HEADER_END)
        if end != -1:
            marker = buf.find(DATA_MARKER, end)
            if marker != -1:
                return bytes(buf[:marker]), marker + len(DATA_MARKER)
        if not chunk:
            raise SpmFormatError("data marker \\x1a\\x04 not found; file truncated or corrupt")
        if len(buf) > _MAX_HEADER:
            raise SpmFormatError("header end not found within the first 32 MiB")


def _tag_value(tags: dict[str, list[str]], name: str) -> str:
    return " ".join(tags.get(name, [])).strip()


def read_header(source: str | Path | BinaryIO) -> SxmHeader:
    """Parse the header of an ``.sxm`` file given a path or a binary stream at offset 0."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as handle:
            raw, data_offset = _read_header_bytes(handle)
    else:
        raw, data_offset = _read_header_bytes(source)
    tags = parse_header_text(raw.decode("latin-1"))

    type_tokens = _tag_value(tags, "SCANIT_TYPE").split()
    if len(type_tokens) < 2 or type_tokens[0].upper() != "FLOAT":
        raise SpmFormatError(f"unsupported SCANIT_TYPE {type_tokens}; only FLOAT is supported")
    dtype = np.dtype(">f4" if type_tokens[1].upper() == "MSBFIRST" else "<f4")

    pixels = _tag_value(tags, "SCAN_PIXELS").split()
    if len(pixels) < 2:
        raise SpmFormatError("missing or malformed SCAN_PIXELS tag")
    nx, ny = int(float(pixels[0])), int(float(pixels[1]))
    if nx <= 0 or ny <= 0:
        raise SpmFormatError(f"invalid SCAN_PIXELS {nx} x {ny}")

    scan_range: tuple[float, float] | None = None
    range_tokens = _tag_value(tags, "SCAN_RANGE").split()
    if len(range_tokens) >= 2:
        scan_range = (float(range_tokens[0]), float(range_tokens[1]))

    return SxmHeader(
        version=_tag_value(tags, "NANONIS_VERSION"),
        nx=nx,
        ny=ny,
        scan_range_m=scan_range,
        scan_dir=_tag_value(tags, "SCAN_DIR").lower(),
        dtype=dtype,
        channels=parse_data_info(tags.get("DATA_INFO", [])),
        data_offset=data_offset,
        tags=tags,
    )


def read_image(handle: BinaryIO, header: SxmHeader, image_number: int) -> np.ndarray:
    """Read one raw image (``ny`` x ``nx``, native float32) from the binary block."""
    count = header.nx * header.ny
    nbytes = count * header.dtype.itemsize
    handle.seek(header.data_offset + image_number * nbytes)
    buf = handle.read(nbytes)
    if len(buf) != nbytes:
        raise SpmFormatError("unexpected end of file while reading image data")
    data = np.frombuffer(buf, dtype=header.dtype).reshape(header.ny, header.nx)
    return data.astype(np.float32)


def orient_image(image: np.ndarray, scan_dir: str, direction: Direction = "forward") -> np.ndarray:
    """Return ``image`` with ``[0, 0]`` at the top-left corner of the scan frame.

    Backward images are mirrored horizontally (they are stored right-to-left); images of
    an ``up`` scan are flipped vertically (the first stored line is the bottom line).
    """
    out = image
    if direction == "backward":
        out = out[:, ::-1]
    if scan_dir.lower() != "down":
        out = out[::-1, :]
    return np.ascontiguousarray(out)


class SxmFile:
    """Lazy reader for one ``.sxm`` file: the header is parsed once, images on demand."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header = read_header(self.path)

    @property
    def channels(self) -> tuple[SxmChannel, ...]:
        return self.header.channels

    def find_channel(self, keyword: str) -> SxmChannel:
        return self.header.find_channel(keyword)

    def read(
        self,
        channel: str | SxmChannel,
        direction: Direction = "forward",
        *,
        orient: bool = True,
    ) -> np.ndarray:
        """Read one image as a native float32 array of shape ``(ny, nx)``.

        With ``orient=True`` (default) the array is returned in display orientation, see
        :func:`orient_image`; with ``orient=False`` it is returned exactly as stored.
        """
        if isinstance(channel, str):
            channel = self.find_channel(channel)
        number = channel.image_number(direction)
        with open(self.path, "rb") as handle:
            data = read_image(handle, self.header, number)
        if orient:
            data = orient_image(data, self.header.scan_dir, direction)
        return data

    def read_all(self, *, orient: bool = True) -> dict[str, dict[Direction, np.ndarray]]:
        """Read every image, keyed by channel name and direction."""
        out: dict[str, dict[Direction, np.ndarray]] = {}
        with open(self.path, "rb") as handle:
            for ch in self.channels:
                entry: dict[Direction, np.ndarray] = {}
                for direction in DIRECTIONS:
                    if direction == "forward" and not ch.has_forward:
                        continue
                    if direction == "backward" and not ch.has_backward:
                        continue
                    data = read_image(handle, self.header, ch.image_number(direction))
                    if orient:
                        data = orient_image(data, self.header.scan_dir, direction)
                    entry[direction] = data
                out[ch.name] = entry
        return out


__all__ = [
    "DATA_MARKER",
    "DIRECTIONS",
    "Direction",
    "SxmChannel",
    "SxmFile",
    "SxmHeader",
    "orient_image",
    "parse_data_info",
    "parse_header_text",
    "read_header",
    "read_image",
]
