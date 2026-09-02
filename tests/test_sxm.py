"""Tests for the Nanonis ``.sxm`` reader."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ramp, sxm_header_text, write_sxm
from spmtools.io import SpmFormatError
from spmtools.io.sxm import SxmFile, orient_image, parse_data_info, parse_header_text, read_header


def test_header_fields(simple_sxm):
    header = read_header(simple_sxm)
    assert header.version == "2"
    assert (header.nx, header.ny) == (8, 6)
    assert header.scan_dir == "down"
    assert header.dtype == np.dtype(">f4")
    assert header.scan_range_m == pytest.approx((2.0e-8, 1.5e-8))
    assert [ch.name for ch in header.channels] == ["Z", "Current", "OC_M1_Freq._Shift"]
    assert [ch.direction for ch in header.channels] == ["both", "both", "fwd"]
    assert [ch.unit for ch in header.channels] == ["m", "A", "Hz"]
    assert [ch.image_index for ch in header.channels] == [0, 2, 4]
    assert header.n_images == 5
    assert header.pixel_size_nm == pytest.approx((2.5, 2.5))
    assert "COMMENT" in header.tags


def test_reads_every_image_in_storage_order(simple_sxm):
    sxm = SxmFile(simple_sxm)
    assert np.array_equal(sxm.read("Z"), ramp(6, 8, 0))
    assert np.array_equal(sxm.read("Current"), ramp(6, 8, 2000))
    assert np.array_equal(sxm.read("OC_M1_Freq._Shift"), ramp(6, 8, 4000))
    assert sxm.read("Z").dtype == np.float32


def test_forward_only_channel_before_a_both_channel(tmp_path):
    """Offsets must account for single-direction channels (index * 2 is wrong here)."""
    channels = [("OC_M1_Freq._Shift", "Hz", "fwd"), ("Z", "m", "both")]
    images = {
        ("OC_M1_Freq._Shift", "forward"): ramp(4, 5, 0),
        ("Z", "forward"): ramp(4, 5, 100),
        ("Z", "backward"): ramp(4, 5, 200),
    }
    sxm = SxmFile(write_sxm(tmp_path / "a.sxm", channels, images))
    assert sxm.header.channels[1].image_index == 1
    assert np.array_equal(sxm.read("Z", "forward"), ramp(4, 5, 100))
    assert np.array_equal(sxm.read("Z", "backward", orient=False), ramp(4, 5, 200))


def test_backward_image_is_mirrored_horizontally(simple_sxm):
    sxm = SxmFile(simple_sxm)
    stored = ramp(6, 8, 1000)
    assert np.array_equal(sxm.read("Z", "backward"), stored[:, ::-1])
    assert np.array_equal(sxm.read("Z", "backward", orient=False), stored)


def test_up_scan_is_flipped_vertically(tmp_path):
    channels = [("Z", "m", "both")]
    stored_fwd, stored_bwd = ramp(3, 4, 0), ramp(3, 4, 50)
    path = write_sxm(
        tmp_path / "up.sxm",
        channels,
        {("Z", "forward"): stored_fwd, ("Z", "backward"): stored_bwd},
        scan_dir="up",
    )
    sxm = SxmFile(path)
    assert np.array_equal(sxm.read("Z"), stored_fwd[::-1, :])
    assert np.array_equal(sxm.read("Z", "backward"), stored_bwd[::-1, ::-1])
    assert np.array_equal(sxm.read("Z", orient=False), stored_fwd)


def test_orient_image_is_an_involution_free_of_copies():
    img = ramp(3, 5)
    assert np.array_equal(orient_image(img, "down"), img)
    assert np.array_equal(orient_image(img, "up"), img[::-1])
    assert np.array_equal(orient_image(img, "down", "backward"), img[:, ::-1])
    assert orient_image(img, "up", "backward").flags["C_CONTIGUOUS"]


def test_missing_direction_raises(simple_sxm):
    sxm = SxmFile(simple_sxm)
    with pytest.raises(KeyError, match="no backward image"):
        sxm.read("OC_M1_Freq._Shift", "backward")


def test_find_channel_exact_then_substring(simple_sxm):
    header = read_header(simple_sxm)
    assert header.find_channel("z").name == "Z"
    assert header.find_channel("Freq Shift").name == "OC_M1_Freq._Shift"
    assert header.find_channel("current").name == "Current"
    with pytest.raises(KeyError, match="available: Z, Current"):
        header.find_channel("Bias")


def test_read_all(simple_sxm):
    data = SxmFile(simple_sxm).read_all()
    assert set(data) == {"Z", "Current", "OC_M1_Freq._Shift"}
    assert set(data["Z"]) == {"forward", "backward"}
    assert set(data["OC_M1_Freq._Shift"]) == {"forward"}
    assert np.array_equal(data["Current"]["backward"], ramp(6, 8, 3000)[:, ::-1])


def test_little_endian_files_are_supported(tmp_path):
    path = write_sxm(
        tmp_path / "le.sxm",
        [("Z", "m", "fwd")],
        {("Z", "forward"): ramp(2, 3)},
        byteorder="LSBFIRST",
    )
    sxm = SxmFile(path)
    assert sxm.header.dtype == np.dtype("<f4")
    assert np.array_equal(sxm.read("Z"), ramp(2, 3))


def test_rejects_non_sxm_files(tmp_path):
    path = tmp_path / "junk.sxm"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 100)
    with pytest.raises(SpmFormatError, match="not a Nanonis SXM"):
        read_header(path)


def test_missing_marker_raises(tmp_path):
    path = tmp_path / "nomarker.sxm"
    path.write_bytes(sxm_header_text([("Z", "m", "fwd")], 2, 2).encode("latin-1"))
    with pytest.raises(SpmFormatError, match="marker"):
        read_header(path)


def test_truncated_image_data_raises(tmp_path):
    path = write_sxm(
        tmp_path / "short.sxm",
        [("Z", "m", "both")],
        {("Z", "forward"): ramp(4, 4), ("Z", "backward"): ramp(4, 4)},
        truncate=8,
    )
    sxm = SxmFile(path)
    assert np.array_equal(sxm.read("Z", "forward"), ramp(4, 4))
    with pytest.raises(SpmFormatError, match="unexpected end of file"):
        sxm.read("Z", "backward")


def test_non_float_sample_type_raises(tmp_path):
    text = sxm_header_text([("Z", "m", "fwd")], 2, 2).replace("FLOAT", "INT")
    path = tmp_path / "int.sxm"
    path.write_bytes(text.encode("latin-1") + b"\x1a\x04" + b"\0" * 16)
    with pytest.raises(SpmFormatError, match="SCANIT_TYPE"):
        read_header(path)


def test_missing_scan_range(tmp_path):
    path = write_sxm(
        tmp_path / "norange.sxm",
        [("Z", "m", "fwd")],
        {("Z", "forward"): ramp(2, 2)},
        scan_range=None,
    )
    header = read_header(path)
    assert header.scan_range_m is None
    with pytest.raises(SpmFormatError, match="SCAN_RANGE"):
        _ = header.pixel_size_nm


def test_parse_header_text():
    tags = parse_header_text(":A:\n1\n:B:\nline1\nline2\n\n:C:\n:SCANIT_END:\n")
    assert tags == {"A": ["1"], "B": ["line1", "line2", ""], "C": [], "SCANIT_END": []}


def test_parse_data_info_without_header_row_and_unknown_direction():
    rows = ["\t14\tZ\tm\tboth\t1\t0", "\t0\tCurrent\tA\tbwd\t1\t0"]
    channels = parse_data_info(rows)
    assert [(c.name, c.direction, c.image_index) for c in channels] == [
        ("Z", "both", 0),
        ("Current", "bwd", 2),
    ]
    assert channels[1].image_number("backward") == 2
    with pytest.raises(SpmFormatError, match="unknown scan direction"):
        parse_data_info(["\tChannel\tName\tUnit\tDirection", "\t1\tZ\tm\tsideways"])
    assert parse_data_info([]) == ()
