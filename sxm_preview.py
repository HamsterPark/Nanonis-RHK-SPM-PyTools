#!/usr/bin/env python3
import argparse
import hashlib
import math
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm

SKIP_CANONICAL = {
    "bias",
    "x",
    "y",
    "oc_d1_phase",
    "oc_d1_amplitude",
    "oc_d1_excitation",
    "oc_m1_excitation",
    "li_demod_1_x",
    "li_demod_1_y",
    "li_demod_2_x",
    "li_demod_2_y",
}

FREQ_SHIFT_CANONICAL = {
    "oc_d1_freq_shift",
    "oc_m1_freq_shift",
}

Z_CANONICAL = {"z"}
CURRENT_CANONICAL = {"current"}

CMAP_FIXED = {
    "z": "copper",
    "z_line": "cividis",
    "current": "inferno",
    "oc_d1_freq_shift": "gray",
    "oc_m1_freq_shift": "gray",
}

CMAP_CYCLE = [
    "viridis",
    "magma",
    "plasma",
    "inferno",
    "cividis",
    "turbo",
    "cubehelix",
    "coolwarm",
]

GRID_COLOR = (48, 48, 48)
GRID_WIDTH = 1

SUPPORTED_EXTS = {".sxm", ".sm4"}
SM4_COLLECT_DEFAULT = "SM4_PreviewPy"
SM4_PREVIEW_PATTERNS = ("Z_*.png", "Z_line_*.png")

SM4_OBJECT_PAGE_INDEX_HEADER = 1
SM4_OBJECT_PAGE_INDEX_ARRAY = 2
SM4_OBJECT_PAGE_HEADER = 3
SM4_OBJECT_PAGE_DATA = 4

SM4_PAGE_DATA_IMAGE = 0
SM4_PAGE_TOPOGRAPHIC = 1
SM4_SCAN_FORWARD = 0
SM4_FLOAT_LINE_TYPES = {1, 6, 9, 10, 11, 13, 18, 19, 21, 22}


@dataclass
class ChannelInfo:
    index: int
    name: str
    direction: str


def canonical_name(name: str) -> str:
    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def safe_filename(name: str) -> str:
    value = name.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_") or "channel"


def extract_tail_id(stem: str, digits: int) -> str:
    match = re.search(r"(\d+)$", stem)
    if not match:
        return stem
    tail = match.group(1)
    tail = tail[-digits:]
    return tail.zfill(digits)


def read_line(handle):
    line = handle.readline()
    if not line:
        return None
    return line.decode("latin-1").strip()


def find_data_offset(handle) -> int:
    prev = None
    while True:
        b = handle.read(1)
        if not b:
            raise ValueError("Missing data marker")
        if prev == b"\x1a" and b == b"\x04":
            return handle.tell()
        prev = b


def parse_header(path: Path):
    with path.open("rb") as handle:
        first = read_line(handle)
        if first != ":NANONIS_VERSION:":
            raise ValueError(f"Not a Nanonis SXM file: {path}")
        version = read_line(handle)
        scan_pixels = None
        scan_dir = None
        data_info = []
        read_tag = True
        while True:
            if read_tag:
                s1 = read_line(handle)
                if s1 is None:
                    break
            tag = s1.strip()
            if tag.startswith(":") and tag.endswith(":"):
                tag = tag[1:-1]
            read_tag = True
            if tag == "SCANIT_END":
                break
            if tag == "SCAN_DIR":
                scan_dir = (read_line(handle) or "").strip()
            elif tag == "SCAN_PIXELS":
                line = read_line(handle) or ""
                parts = line.split()
                if len(parts) >= 2:
                    scan_pixels = (int(float(parts[0])), int(float(parts[1])))
            elif tag == "DATA_INFO":
                while True:
                    s2 = read_line(handle)
                    if s2 is None:
                        break
                    cols = s2.strip().split("\t")
                    if len(cols) <= 2:
                        if s2.startswith(":") and s2.endswith(":"):
                            s1 = s2
                            read_tag = False
                        break
                    data_info.append(cols)
            elif tag == "COMMENT":
                while True:
                    s2 = read_line(handle)
                    if s2 is None:
                        break
                    if s2.startswith(":"):
                        s1 = s2
                        read_tag = False
                        break
            elif tag == "Z-CONTROLLER":
                read_line(handle)
                read_line(handle)
            else:
                s_line = read_line(handle)
                if s_line is None:
                    break
                if s_line.startswith(":"):
                    s1 = s_line
                    read_tag = False
                else:
                    while True:
                        s_line = read_line(handle)
                        if s_line is None:
                            break
                        if s_line.startswith(":"):
                            s1 = s_line
                            read_tag = False
                            break
        data_offset = find_data_offset(handle)
    if scan_pixels is None:
        raise ValueError(f"Missing SCAN_PIXELS in header: {path}")
    return {
        "version": version,
        "scan_pixels": scan_pixels,
        "scan_dir": scan_dir or "",
        "data_info": data_info,
        "data_offset": data_offset,
    }


def parse_channels(data_info):
    rows = data_info
    if rows and rows[0][0].lower() == "channel":
        rows = rows[1:]
    channels = []
    for idx, row in enumerate(rows):
        name = row[1] if len(row) > 1 else ""
        direction = row[3] if len(row) > 3 else ""
        channels.append(ChannelInfo(index=idx, name=name, direction=direction))
    return channels


def read_channel(handle, header, channel_index: int):
    nx, ny = header["scan_pixels"]
    count = nx * ny
    offset = header["data_offset"] + channel_index * 2 * count * 4
    handle.seek(offset)
    data = np.fromfile(handle, dtype=">f4", count=count)
    if data.size != count:
        raise ValueError("Unexpected end of file while reading data")
    return data.reshape((ny, nx))


def _read_sm4_array(handle, dtype, count):
    data = np.fromfile(handle, dtype=dtype, count=count)
    if data.size != count:
        raise ValueError("Unexpected end of file while reading SM4 data")
    return data


def _read_sm4_value(handle, dtype):
    return _read_sm4_array(handle, dtype, 1)[0]


def _read_sm4_objects(handle, count):
    objects = []
    for _ in range(count):
        obj_id = int(_read_sm4_value(handle, np.uint32))
        offset = int(_read_sm4_value(handle, np.uint32))
        size = int(_read_sm4_value(handle, np.uint32))
        objects.append((obj_id, offset, size))
    return objects


def _find_sm4_object_offset(objects, target_id):
    for obj_id, offset, _size in objects:
        if obj_id == target_id:
            return offset
    return None


def _read_sm4_page_header(handle, offset):
    handle.seek(offset, 0)
    _ = _read_sm4_value(handle, np.uint16)  # field size
    _ = _read_sm4_value(handle, np.uint16)  # string count
    page_type = int(_read_sm4_value(handle, np.uint32))
    _ = _read_sm4_value(handle, np.uint32)  # data sub source
    line_type = int(_read_sm4_value(handle, np.uint32))
    _ = _read_sm4_value(handle, np.uint32)  # xcorner
    _ = _read_sm4_value(handle, np.uint32)  # ycorner
    xsize = int(_read_sm4_value(handle, np.uint32))
    ysize = int(_read_sm4_value(handle, np.uint32))
    _ = _read_sm4_value(handle, np.uint32)  # image type
    scan_type = int(_read_sm4_value(handle, np.uint32))
    _ = _read_sm4_value(handle, np.uint32)  # group id
    page_data_size = int(_read_sm4_value(handle, np.uint32))
    _ = _read_sm4_value(handle, np.uint32)  # min z
    _ = _read_sm4_value(handle, np.uint32)  # max z
    xscale = float(_read_sm4_value(handle, np.float32))
    yscale = float(_read_sm4_value(handle, np.float32))
    zscale = float(_read_sm4_value(handle, np.float32))
    _ = _read_sm4_value(handle, np.float32)  # xyscale
    _ = _read_sm4_value(handle, np.float32)  # xoffset
    _ = _read_sm4_value(handle, np.float32)  # yoffset
    zoffset = float(_read_sm4_value(handle, np.float32))
    _ = _read_sm4_value(handle, np.float32)  # period
    _ = _read_sm4_value(handle, np.float32)  # bias
    _ = _read_sm4_value(handle, np.float32)  # current
    _ = _read_sm4_value(handle, np.uint32)  # color info count
    _ = _read_sm4_value(handle, np.uint32)  # grid xsize
    _ = _read_sm4_value(handle, np.uint32)  # grid ysize
    _ = _read_sm4_value(handle, np.uint32)  # object list count
    _ = _read_sm4_value(handle, np.uint8)  # data flag
    _read_sm4_array(handle, np.uint8, 3)  # reserved flags
    _read_sm4_array(handle, np.uint8, 60)  # reserved
    if xsize <= 0 or ysize <= 0:
        return None
    return {
        "page_type": page_type,
        "scan_type": scan_type,
        "line_type": line_type,
        "xsize": xsize,
        "ysize": ysize,
        "xscale": xscale,
        "yscale": yscale,
        "zscale": zscale,
        "zoffset": zoffset,
        "page_data_size": page_data_size,
    }


def _read_sm4_image_data(handle, data_offset, info):
    page_data_size = info["page_data_size"]
    if page_data_size <= 0:
        return None
    xsize = info["xsize"]
    ysize = info["ysize"]
    count = page_data_size // 4
    if count <= 0:
        return None
    handle.seek(data_offset, 0)
    dtype = np.float32 if info["line_type"] in SM4_FLOAT_LINE_TYPES else np.int32
    raw = np.fromfile(handle, dtype=dtype, count=count)
    expected = xsize * ysize
    if raw.size < expected:
        return None
    if raw.size != expected:
        raw = raw[:expected]
    data = raw.reshape((xsize, ysize))
    if info["xscale"] < 0:
        data = np.flip(data, axis=1)
    if info["yscale"] > 0:
        data = np.flip(data, axis=0)
    data = data.astype(np.float32) * info["zscale"] + info["zoffset"]
    return data


def read_sm4_z_forward(path: Path, verbose: bool = False):
    fallback = None
    with path.open("rb") as handle:
        header_size = int(_read_sm4_value(handle, np.uint16))
        _read_sm4_array(handle, np.uint16, 18)
        _ = _read_sm4_value(handle, np.uint32)  # total page count
        object_list_count = int(_read_sm4_value(handle, np.uint32))
        _ = _read_sm4_value(handle, np.uint32)  # object field size
        _read_sm4_array(handle, np.uint32, 2)

        handle.seek(header_size + 2, 0)
        file_objects = _read_sm4_objects(handle, object_list_count)
        page_index_header_offset = _find_sm4_object_offset(file_objects, SM4_OBJECT_PAGE_INDEX_HEADER)
        if page_index_header_offset is None:
            if verbose:
                print(f"[WARN] {path}: missing SM4 page index header")
            return None

        handle.seek(page_index_header_offset, 0)
        page_count = int(_read_sm4_value(handle, np.uint32))
        page_index_obj_count = int(_read_sm4_value(handle, np.uint32))
        _read_sm4_array(handle, np.uint32, 2)
        page_index_objects = _read_sm4_objects(handle, page_index_obj_count)
        page_index_array_offset = _find_sm4_object_offset(page_index_objects, SM4_OBJECT_PAGE_INDEX_ARRAY)
        if page_index_array_offset is None:
            if verbose:
                print(f"[WARN] {path}: missing SM4 page index array")
            return None

        handle.seek(page_index_array_offset, 0)
        for _ in range(page_count):
            _read_sm4_array(handle, np.uint16, 8)
            page_data_type = int(_read_sm4_value(handle, np.uint32))
            _ = _read_sm4_value(handle, np.uint32)  # page source type
            page_obj_count = int(_read_sm4_value(handle, np.uint32))
            _ = _read_sm4_value(handle, np.uint32)  # minor version
            page_objects = _read_sm4_objects(handle, page_obj_count)
            if page_data_type != SM4_PAGE_DATA_IMAGE:
                continue
            header_offset = _find_sm4_object_offset(page_objects, SM4_OBJECT_PAGE_HEADER)
            data_offset = _find_sm4_object_offset(page_objects, SM4_OBJECT_PAGE_DATA)
            if header_offset is None or data_offset is None:
                continue
            pos = handle.tell()
            info = _read_sm4_page_header(handle, header_offset)
            handle.seek(pos, 0)
            if info is None:
                continue
            if info["page_type"] != SM4_PAGE_TOPOGRAPHIC:
                continue
            if info["scan_type"] == SM4_SCAN_FORWARD:
                data = _read_sm4_image_data(handle, data_offset, info)
                if data is not None:
                    return data
            if fallback is None:
                fallback = (data_offset, info)

        if fallback:
            data_offset, info = fallback
            return _read_sm4_image_data(handle, data_offset, info)

    return None


def has_signal(data: np.ndarray, const_tol: float, zero_tol: float) -> bool:
    if data is None:
        return False
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    if zero_tol > 0 and np.all(np.abs(finite) <= zero_tol):
        return False
    min_val = float(np.min(finite))
    max_val = float(np.max(finite))
    if (max_val - min_val) <= const_tol:
        return False
    return True


def apply_scan_dir(data: np.ndarray, scan_dir: str) -> np.ndarray:
    if scan_dir.lower() != "down":
        return np.flipud(data)
    return data


def line_normalize_z(data: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_median = np.nanmedian(data, axis=1, keepdims=True)
    row_median = np.where(np.isnan(row_median), 0.0, row_median)
    return data - row_median


def normalize_for_display(data: np.ndarray, percentiles):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    vmin, vmax = np.percentile(finite, percentiles)
    if vmin == vmax:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmin == vmax:
        return np.zeros_like(data, dtype=np.float32)
    normed = (data - vmin) / (vmax - vmin)
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


def pick_colormap(canonical: str) -> str:
    if canonical in CMAP_FIXED:
        return CMAP_FIXED[canonical]
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(CMAP_CYCLE)
    return CMAP_CYCLE[idx]


def draw_label(draw, position, text, font):
    try:
        draw.text(
            position,
            text,
            fill=(255, 255, 255),
            font=font,
            stroke_fill=(0, 0, 0),
            stroke_width=1,
        )
    except TypeError:
        x, y = position
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
        draw.text(position, text, fill=(255, 255, 255), font=font)


def get_cmap(name: str):
    try:
        from matplotlib import colormaps

        return colormaps.get_cmap(name)
    except Exception:
        return cm.get_cmap(name)


def render_tile(data: np.ndarray, cmap_name: str, tile_size: int, label: str, percentiles, add_label: bool) -> Image.Image:
    normed = normalize_for_display(data, percentiles)
    cmap = get_cmap(cmap_name)
    rgb = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if tile_size and img.size != (tile_size, tile_size):
        img = img.resize((tile_size, tile_size), resample=Image.BILINEAR)
    if add_label and label:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw_label(draw, (2, 2), label, font)
    return img


def is_backward_channel(name: str, direction: str) -> bool:
    cname = canonical_name(name)
    if "bwd" in cname or cname.endswith("bwd"):
        return True
    if direction:
        d = direction.strip().lower()
        if d.startswith("bwd") or d == "backward":
            return True
    return False


def find_channel_index(channels, targets):
    for ch in channels:
        if canonical_name(ch.name) in targets:
            return ch.index
    return None


class MosaicBuilder:
    def __init__(self, channel_name: str, out_dir: Path, tile_size: int, max_tiles: int, cols: int):
        self.channel_name = channel_name
        self.out_dir = out_dir
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        self.cols = cols
        self.tiles = []
        self.part_index = 1

    def add(self, img: Image.Image):
        self.tiles.append(img)
        if self.max_tiles and len(self.tiles) >= self.max_tiles:
            self.flush()

    def flush(self):
        if not self.tiles:
            return
        count = len(self.tiles)
        cols = self.cols
        if not cols:
            cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        mosaic = Image.new("RGB", (cols * self.tile_size, rows * self.tile_size), color=(0, 0, 0))
        for idx, img in enumerate(self.tiles):
            row = idx // cols
            col = idx % cols
            mosaic.paste(img, (col * self.tile_size, row * self.tile_size))
        if cols > 1 or rows > 1:
            draw = ImageDraw.Draw(mosaic)
            width = cols * self.tile_size
            height = rows * self.tile_size
            for col in range(1, cols):
                x = col * self.tile_size
                draw.line([(x, 0), (x, height)], fill=GRID_COLOR, width=GRID_WIDTH)
            for row in range(1, rows):
                y = row * self.tile_size
                draw.line([(0, y), (width, y)], fill=GRID_COLOR, width=GRID_WIDTH)
        out_name = f"{safe_filename(self.channel_name)}_{self.part_index:02d}.png"
        out_path = self.out_dir / out_name
        mosaic.save(out_path, optimize=True)
        self.tiles = []
        self.part_index += 1

    def finalize(self):
        self.flush()

class Progress:
    def __init__(self, total: int, enabled: bool = True, width: int = 28):
        self.total = total
        self.enabled = enabled and total > 0
        self.width = width
        self.count = 0
        if self.enabled:
            self._print()

    def advance(self, step: int = 1):
        if not self.enabled:
            return
        self.count += step
        if self.count > self.total:
            self.count = self.total
        self._print()
        if self.count >= self.total:
            print()

    def _print(self):
        filled = int(self.width * self.count / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100.0 * self.count / self.total
        print(f"\rProgress {self.count}/{self.total} [{bar}] {percent:5.1f}%", end="", flush=True)


def collect_previews(out_dir: Path, collect_dir: Path, rel_path: Path, folder: Path, patterns=None):
    if collect_dir is None:
        return
    if patterns is None:
        patterns = ("*.png",)
    parts = [safe_filename(part) for part in rel_path.parts if part not in (".", "")]
    if not parts:
        parts = [safe_filename(folder.name)]
    prefix = "_".join(parts)
    images = []
    for pattern in patterns:
        images.extend(out_dir.glob(pattern))
    for image_path in sorted(set(images)):
        new_name = f"{prefix}__{image_path.name}"
        shutil.copy2(image_path, collect_dir / new_name)


def process_sxm_file(path: Path, args, builders, counts, out_dir):
    header = parse_header(path)
    channels = parse_channels(header["data_info"])

    z_index = find_channel_index(channels, Z_CANONICAL)
    freq_index = find_channel_index(channels, FREQ_SHIFT_CANONICAL)
    current_index = find_channel_index(channels, CURRENT_CANONICAL)

    label = extract_tail_id(path.stem, args.label_digits)

    with path.open("rb") as handle:
        z_data = read_channel(handle, header, z_index) if z_index is not None else None
        freq_data = read_channel(handle, header, freq_index) if freq_index is not None else None

        z_has_signal = has_signal(z_data, args.const_tol, args.zero_tol) if z_data is not None else False
        freq_has_signal = has_signal(freq_data, args.const_tol, args.zero_tol) if freq_data is not None else False

        for ch in channels:
            cname = canonical_name(ch.name)
            if is_backward_channel(ch.name, ch.direction):
                continue
            if cname in SKIP_CANONICAL:
                continue

            data = None
            channel_key = cname
            channel_label = ch.name

            if cname in Z_CANONICAL:
                if not z_has_signal:
                    continue
                data = z_data
                data = apply_scan_dir(data, header["scan_dir"])
                tile = render_tile(data, pick_colormap("z"), args.tile_size, label, args.percentiles, not args.no_label)
                builders.setdefault("Z", MosaicBuilder("Z", out_dir, args.tile_size, args.max_tiles, args.cols)).add(tile)
                counts["Z"] = counts.get("Z", 0) + 1

                z_corr = line_normalize_z(z_data)
                z_corr = apply_scan_dir(z_corr, header["scan_dir"])
                tile = render_tile(z_corr, pick_colormap("z_line"), args.tile_size, label, args.percentiles, not args.no_label)
                builders.setdefault("Z_line", MosaicBuilder("Z_line", out_dir, args.tile_size, args.max_tiles, args.cols)).add(tile)
                counts["Z_line"] = counts.get("Z_line", 0) + 1
                continue

            if cname in FREQ_SHIFT_CANONICAL:
                if not freq_has_signal:
                    continue
                data = freq_data
            elif cname in CURRENT_CANONICAL:
                if z_has_signal and not freq_has_signal:
                    continue
                if current_index is None:
                    continue
                data = read_channel(handle, header, current_index)
            else:
                data = read_channel(handle, header, ch.index)

            if data is None:
                continue

            if args.skip_constant and not has_signal(data, args.const_tol, args.zero_tol):
                continue

            data = apply_scan_dir(data, header["scan_dir"])
            cmap_name = pick_colormap(channel_key)

            builder = builders.get(channel_label)
            if builder is None:
                builder = MosaicBuilder(channel_label, out_dir, args.tile_size, args.max_tiles, args.cols)
                builders[channel_label] = builder

            tile = render_tile(data, cmap_name, args.tile_size, label, args.percentiles, not args.no_label)
            builder.add(tile)
            counts[channel_label] = counts.get(channel_label, 0) + 1


def process_sm4_file(path: Path, args, builders, counts, out_dir):
    data = read_sm4_z_forward(path, verbose=args.verbose)
    if data is None:
        return
    if args.skip_constant and not has_signal(data, args.const_tol, args.zero_tol):
        return
    label = extract_tail_id(path.stem, args.label_digits)
    tile = render_tile(data, pick_colormap("z"), args.tile_size, label, args.percentiles, not args.no_label)
    builder = builders.get("Z")
    if builder is None:
        builder = MosaicBuilder("Z", out_dir, args.tile_size, args.max_tiles, args.cols)
        builders["Z"] = builder
    builder.add(tile)
    counts["Z"] = counts.get("Z", 0) + 1

    z_corr = line_normalize_z(data)
    tile = render_tile(z_corr, pick_colormap("z_line"), args.tile_size, label, args.percentiles, not args.no_label)
    builder = builders.get("Z_line")
    if builder is None:
        builder = MosaicBuilder("Z_line", out_dir, args.tile_size, args.max_tiles, args.cols)
        builders["Z_line"] = builder
    builder.add(tile)
    counts["Z_line"] = counts.get("Z_line", 0) + 1


def process_folder(folder: Path, files, args, progress=None, rel_path=None, collect_dir=None, sm4_collect_dir=None):
    if not files:
        return
    out_dir = folder / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {}
    counts = {}
    sm4_seen = False

    for path in files:
        try:
            suffix = path.suffix.lower()
            if suffix == ".sxm":
                process_sxm_file(path, args, builders, counts, out_dir)
            elif suffix == ".sm4":
                sm4_seen = True
                process_sm4_file(path, args, builders, counts, out_dir)
            else:
                if args.verbose:
                    print(f"[WARN] {path}: unsupported extension")
        except Exception as exc:
            if args.verbose:
                print(f"[WARN] {path}: {exc}")
        if progress is not None:
            progress.advance()

    for builder in builders.values():
        builder.finalize()
    if collect_dir is not None and rel_path is not None:
        collect_previews(out_dir, collect_dir, rel_path, folder)
    if sm4_collect_dir is not None and sm4_seen and rel_path is not None:
        collect_previews(out_dir, sm4_collect_dir, rel_path, folder, patterns=SM4_PREVIEW_PATTERNS)

    if args.verbose:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{folder}: {summary}")


def find_day_folders(root: Path):
    day_folders = []
    for year_dir in root.iterdir():
        if not year_dir.is_dir():
            continue
        if year_dir.name == "data23":
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
            for day_dir in month_dir.iterdir():
                if not day_dir.is_dir():
                    continue
                if any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTS for p in day_dir.iterdir()):
                    day_folders.append(day_dir)
    data23_dir = root / "data23"
    if data23_dir.is_dir():
        for day_dir in data23_dir.iterdir():
            if not day_dir.is_dir():
                continue
            if any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTS for p in day_dir.iterdir()):
                day_folders.append(day_dir)
    return day_folders


def collect_folder_files(folders, limit: int):
    folder_files = []
    total = 0
    for folder in folders:
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
        if limit:
            files = files[:limit]
        if not files:
            continue
        folder_files.append((folder, files))
        total += len(files)
    return folder_files, total


def parse_percentiles(value: str):
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("Percentiles must be like '1,99'")
    return (float(parts[0]), float(parts[1]))


def main():
    parser = argparse.ArgumentParser(description="Generate mosaic previews for Nanonis SXM and RHK SM4 files.")
    parser.add_argument("root", help="Folder with .sxm/.sm4 files, or dataset root if --recursive")
    parser.add_argument("--recursive", action="store_true", help="Scan year/month/day folders under root")
    parser.add_argument("--out-name", default="PreviewPy", help="Output folder name inside each day folder")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels (square)")
    parser.add_argument("--max-tiles", type=int, default=25, help="Max tiles per mosaic before splitting")
    parser.add_argument("--cols", type=int, default=5, help="Fixed mosaic columns (0 = auto)")
    parser.add_argument("--percentiles", default="1,99", help="Clip percentiles for display scaling")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files per folder (0 = no limit)")
    parser.add_argument("--label-digits", type=int, default=5, help="Digits for file label (tail) in each tile")
    parser.add_argument("--keep-constant", dest="skip_constant", action="store_false", help="Keep constant/empty channels")
    parser.add_argument("--const-tol", type=float, default=0.0, help="Tolerance for constant detection")
    parser.add_argument("--zero-tol", type=float, default=0.0, help="Tolerance for all-zero detection")
    parser.add_argument("--no-label", action="store_true", help="Disable file number labels")
    parser.add_argument("--collect-dir", default="", help="Copy mosaics into this folder with prefixed names")
    parser.add_argument("--sm4-collect-dir", default=SM4_COLLECT_DEFAULT, help="SM4 previews folder (relative to root)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    args.percentiles = parse_percentiles(args.percentiles)
    args.skip_constant = getattr(args, "skip_constant", True)

    root = Path(args.root)
    if args.recursive:
        folders = sorted(find_day_folders(root))
    else:
        folders = [root]

    folder_files, total_files = collect_folder_files(folders, args.limit)
    if total_files == 0:
        print("No .sxm or .sm4 files found.")
        return

    type_counts = {}
    for _, files in folder_files:
        for path in files:
            ext = path.suffix.lower()
            type_counts[ext] = type_counts.get(ext, 0) + 1

    if args.verbose:
        print(f"Folders: {len(folder_files)}")
    if type_counts:
        summary = ", ".join(f"{ext[1:].upper()}={count}" for ext, count in sorted(type_counts.items()))
        print(f"Total files: {total_files} ({summary})")
    else:
        print(f"Total files: {total_files}")

    collect_dir = None
    if args.collect_dir:
        collect_dir = Path(args.collect_dir)
        if not collect_dir.is_absolute():
            collect_dir = root / collect_dir
        collect_dir.mkdir(parents=True, exist_ok=True)

    sm4_collect_dir = None
    if args.sm4_collect_dir:
        sm4_collect_dir = Path(args.sm4_collect_dir)
        if not sm4_collect_dir.is_absolute():
            sm4_collect_dir = root / sm4_collect_dir
        sm4_collect_dir.mkdir(parents=True, exist_ok=True)

    progress = Progress(total_files, enabled=not args.no_progress)
    for folder, files in folder_files:
        try:
            rel_path = folder.relative_to(root)
        except ValueError:
            rel_path = Path(folder.name)
        process_folder(
            folder,
            files,
            args,
            progress=progress,
            rel_path=rel_path,
            collect_dir=collect_dir,
            sm4_collect_dir=sm4_collect_dir,
        )


if __name__ == "__main__":
    main()
