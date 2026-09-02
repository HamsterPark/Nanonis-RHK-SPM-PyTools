"""Tests for sub-pixel alignment and difference maps."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from conftest import write_sxm
from spmtools import align


def blobs(n: int = 96, seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """Smooth test image: Gaussian bumps well inside the frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n))
    for cy, cx, amp in [(30, 40, 1.0), (60, 70, 0.8), (70, 25, 0.6), (45, 50, 0.9)]:
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0**2))
    return img + noise * rng.standard_normal(img.shape)


def test_prepare_for_registration_cleans_and_windows():
    img = blobs()
    img[5, 5] = np.nan
    img[6, 6] = np.inf
    out = align.prepare_for_registration(img)
    assert np.isfinite(out).all()
    assert out[0, :].max() == 0.0 and out[:, 0].max() == 0.0  # Hann window edges
    centre = (img.shape[0] // 2, img.shape[1] // 2)
    assert out[centre] == pytest.approx(img[centre] - np.nanmean(img[np.isfinite(img)]), rel=1e-3)


def test_refinement_removes_window_bias():
    ref = blobs()
    mov = ndimage.shift(ref, (5.0, 7.0), order=3, mode="nearest")
    single, _ = align.estimate_shift(ref, mov, upsample=100, refine=0)
    refined, _ = align.estimate_shift(ref, mov, upsample=100, refine=2)
    assert abs(single[0] + 5.0) > 0.1  # the one-pass estimate is biased towards zero
    assert refined == pytest.approx((-5.0, -7.0), abs=0.05)


@pytest.mark.parametrize("true_shift", [(2.37, -3.61), (-0.4, 0.25), (5.0, 7.0)])
def test_estimate_shift_recovers_subpixel_translation(true_shift):
    ref = blobs()
    mov = ndimage.shift(ref, true_shift, order=3, mode="nearest")
    shift, error = align.estimate_shift(ref, mov, upsample=100)
    assert shift == pytest.approx((-true_shift[0], -true_shift[1]), abs=0.05)
    assert np.isfinite(error)


def test_apply_shift_valid_mask_is_a_rectangle():
    n = 32
    shifted, valid = align.apply_shift(np.ones((n, n)), np.array([2.5, -3.5]))
    assert shifted.shape == (n, n)
    assert align.overlap_slices(valid) == (slice(3, n), slice(0, n - 4))
    assert valid[3:, : n - 4].all() and not valid[:3].any() and not valid[:, n - 4 :].any()
    assert align.overlap_slices(np.zeros((4, 4), dtype=bool)) is None


def test_align_and_diff_reduces_residual():
    ref = blobs(noise=0.01)
    mov = ndimage.shift(blobs(seed=1, noise=0.01), (1.8, -2.2), order=3, mode="nearest")
    result = align.align_and_diff(ref, mov, upsample=50)
    assert result.shift == pytest.approx((-1.8, 2.2), abs=0.05)
    # what remains is the independent noise of the two frames (sqrt(2) * 0.01)
    assert result.rms_after < 0.02 < 0.5 * result.rms_before
    assert result.improvement_percent > 70
    assert result.a.shape == result.b_aligned.shape == result.diff_after.shape
    assert result.a.shape[0] < ref.shape[0] and result.a.shape[1] < ref.shape[1]
    smoothed = align.align_and_diff(ref, mov, upsample=50, smooth_sigma=1.0)
    assert smoothed.rms_after <= result.rms_after
    with pytest.raises(ValueError, match="same shape"):
        align.align_and_diff(ref, ref[:-1])


def test_rms_and_improvement_guard_degenerate_input():
    assert align.rms(np.array([3.0, np.nan, -4.0])) == pytest.approx(5.0 / np.sqrt(2))
    assert align.rms(np.array([np.nan])) == 0.0
    assert align.improvement_percent(0.0, 0.0) == 0.0
    assert align.improvement_percent(2.0, 0.5) == pytest.approx(75.0)


def test_symmetric_limit_is_always_positive():
    diff = np.array([[-1.0, 0.5], [0.25, np.nan]])
    assert align.symmetric_limit(diff, 100) == pytest.approx(1.0)
    assert align.symmetric_limit(diff, 50) == pytest.approx(0.5)
    assert align.symmetric_limit(np.zeros((3, 3))) == 1.0
    assert align.symmetric_limit(np.full((2, 2), np.nan)) == 1.0


def test_naming_helpers():
    assert align.short_id("C:/data/Ag(111)1098.sxm") == "1098"
    assert align.short_id("synthetic_A.sxm") == "synthetic_A"
    assert align.channel_slug("Z") == "z"
    assert align.channel_slug("OC_M1_Freq._Shift") == "freqshift"
    assert align.channel_slug("LI Demod 1 X") == "lidemod1x"
    assert align.channel_slug("***") == "chan"


@pytest.mark.parametrize(
    "fov, expected", [(20.0, 5.0), (500.0, 100.0), (3.0, 0.5), (1234.0, 200.0), (0.0, 1.0)]
)
def test_nice_scalebar_nm(fov, expected):
    assert align.nice_scalebar_nm(fov) == expected


def _pair(tmp_path, shift=(1.5, -2.25), unit="m"):
    ref = blobs(n=64) * 1e-9
    mov = ndimage.shift(ref, shift, order=3, mode="nearest")
    channels = [("Z", unit, "both")]
    path_a = write_sxm(
        tmp_path / "scan_0001.sxm",
        channels,
        {("Z", "forward"): ref, ("Z", "backward"): ref},
        scan_range=(1.6e-8, 1.6e-8),
    )
    path_b = write_sxm(
        tmp_path / "scan_0002.sxm",
        channels,
        {("Z", "forward"): mov, ("Z", "backward"): mov},
        scan_range=(1.6e-8, 1.6e-8),
    )
    return path_a, path_b


def test_align_sxm_files_writes_all_products(tmp_path):
    path_a, path_b = _pair(tmp_path)
    result, outputs = align.align_sxm_files(path_a, path_b, outdir=tmp_path / "out", upsample=20)
    assert result.shift == pytest.approx((-1.5, 2.25), abs=0.05)
    assert [p.name for p in outputs] == [
        "z_fwd_0001-0002.png",
        "z_fwd_0002-0001.png",
        "z_fwd_alignment_overview_0001_0002.png",
        "z_fwd_0001-0002.npy",
    ]
    assert all(p.is_file() and p.stat().st_size > 1000 for p in outputs)
    saved = np.load(outputs.npy)
    assert saved.shape == result.diff_after.shape
    assert np.allclose(saved, result.diff_after)
    assert result.a.max() < 10  # metres were converted to nanometres


def test_align_sxm_files_smoothing_tag_and_default_outdir(tmp_path):
    path_a, path_b = _pair(tmp_path, shift=(0.0, 0.0))
    result, outputs = align.align_sxm_files(
        path_a, path_b, upsample=10, smooth_sigma=1.5, direction="backward"
    )
    assert outputs.diff_ab == tmp_path / "z_bwd_0001-0002_sigma1.5.png"
    assert outputs.diff_ab.is_file()
    assert result.shift == pytest.approx((0.0, 0.0), abs=0.05)
    assert result.improvement_percent == 0.0 or result.rms_before == pytest.approx(0.0, abs=1e-12)
