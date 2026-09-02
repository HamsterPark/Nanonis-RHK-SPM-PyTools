"""Sub-pixel registration and difference of two scans of (nearly) the same area.

Algorithm
---------
1. Both images are cleaned (NaN/Inf replaced by the mean), de-meaned and multiplied by
   a separable 2-D Hann window to suppress edge leakage in the Fourier domain.
2. The translation is estimated by upsampled cross-correlation (Guizar-Sicairos,
   Thurman & Fienup, Opt. Lett. 33, 156 (2008)) through
   :func:`skimage.registration.phase_cross_correlation` with ``normalization=None``.
   Plain cross-correlation is used on purpose: the spectral whitening of *phase*
   correlation locks onto the line noise and periodic streaks common in SPM images and
   then reports spurious shifts of tens of pixels.
3. The estimate is refined: the moving image is resampled with the current shift, both
   images are cropped to their common area and correlated again.  The residual shift is
   then close to zero, where the bias introduced by the window vanishes.
4. The moving image is resampled onto the reference grid with a cubic spline.  Pixels
   that had to be extrapolated are masked out and everything is cropped to the common
   rectangle, so the difference maps contain real data only.

Requires the ``align`` extra (``scipy`` and ``scikit-image``).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spmtools.io.sxm import Direction, SxmFile

try:
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    ndimage = None
    phase_cross_correlation = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)

QUANTITY_BY_UNIT = {
    "Hz": "frequency shift",
    "nm": "height",
    "m": "height",
    "A": "current",
    "V": "voltage",
    "deg": "phase",
}


def _require_extra() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "spmtools.align needs scipy and scikit-image: "
            "pip install 'nanonis-rhk-spm-pytools[align]'"
        ) from _IMPORT_ERROR


# --------------------------------------------------------------------------- registration


def prepare_for_registration(image: np.ndarray) -> np.ndarray:
    """Fill non-finite pixels with the mean, remove the DC level and apply a Hann window."""
    image = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(image)
    fill = image[finite].mean() if finite.any() else 0.0
    clean = np.where(finite, image, fill)
    clean = clean - clean.mean()
    window = np.outer(np.hanning(clean.shape[0]), np.hanning(clean.shape[1]))
    return clean * window


def _correlate(ref: np.ndarray, mov: np.ndarray, upsample: int) -> tuple[np.ndarray, float]:
    shift, error, _phase = phase_cross_correlation(
        prepare_for_registration(ref),
        prepare_for_registration(mov),
        upsample_factor=upsample,
        normalization=None,
    )
    return np.asarray(shift, dtype=float), float(error)


def estimate_shift(
    ref: np.ndarray, mov: np.ndarray, upsample: int = 100, refine: int = 2
) -> tuple[np.ndarray, float]:
    """Translation ``(rows, cols)`` that registers ``mov`` onto ``ref``, plus the error.

    ``ndimage.shift(mov, shift)`` afterwards overlays ``mov`` on ``ref``.  ``upsample``
    sets the sub-pixel resolution (100 = 1/100 px).

    The window applied before correlating biases a single estimate towards zero by a
    few percent of the shift.  Each of the ``refine`` passes therefore resamples ``mov``
    with the current estimate, crops both images to their common area and correlates
    again; the residual is close to zero, where the bias is negligible.
    """
    _require_extra()
    ref = np.asarray(ref, dtype=np.float64)
    mov = np.asarray(mov, dtype=np.float64)
    shift, error = _correlate(ref, mov, upsample)
    for _ in range(refine):
        moved, valid = apply_shift(mov, shift)
        window = overlap_slices(valid)
        if window is None:
            break
        delta, error = _correlate(ref[window], moved[window], upsample)
        if not np.all(np.abs(delta) < 1.0):  # a refinement must be a small correction
            break
        shift = shift + delta
    return shift, error


def apply_shift(mov: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift ``mov`` by ``shift`` (cubic spline) and return ``(shifted, valid_mask)``.

    ``valid_mask`` is True where the shifted image comes from real data and False where
    the edge had to be extrapolated.
    """
    _require_extra()
    shifted = ndimage.shift(np.asarray(mov, dtype=np.float64), shift=shift, order=3, mode="nearest")
    ones = np.ones(mov.shape, dtype=np.float64)
    valid = ndimage.shift(ones, shift=shift, order=1, mode="constant", cval=0.0) > 0.999
    return shifted, valid


def overlap_slices(valid: np.ndarray) -> tuple[slice, slice] | None:
    """Bounding box of the valid region (a rectangle for pure translations)."""
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def rms(values: np.ndarray) -> float:
    """Root mean square of the finite entries (0.0 if there are none)."""
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0


def improvement_percent(rms_before: float, rms_after: float) -> float:
    """Relative RMS reduction in percent; 0 when the reference RMS is zero."""
    if rms_before <= 0 or not math.isfinite(rms_before):
        return 0.0
    return (1.0 - rms_after / rms_before) * 100.0


def symmetric_limit(diff: np.ndarray, clip: float = 99.0) -> float:
    """Colour-scale limit: the ``clip`` percentile of ``|diff|``; always positive."""
    finite = np.abs(diff[np.isfinite(diff)])
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(finite, clip))
    if vmax <= 0 or not math.isfinite(vmax):
        vmax = float(finite.max())
    return vmax if vmax > 0 else 1.0


@dataclass
class AlignResult:
    """Outcome of :func:`align_and_diff`; all arrays are cropped to the common area."""

    shift: tuple[float, float]
    """Estimated translation ``(rows, cols)`` of B relative to A, in pixels."""
    error: float
    a: np.ndarray
    b: np.ndarray
    b_aligned: np.ndarray
    diff_before: np.ndarray
    diff_after: np.ndarray
    rms_before: float
    rms_after: float
    window: tuple[slice, slice]
    """Slices of the original arrays that were kept."""

    @property
    def improvement_percent(self) -> float:
        return improvement_percent(self.rms_before, self.rms_after)


def align_and_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    upsample: int = 100,
    smooth_sigma: float = 0.0,
) -> AlignResult:
    """Register ``b`` onto ``a`` and compute ``a - b`` before and after alignment.

    ``smooth_sigma > 0`` applies a Gaussian filter (in pixels) to both difference maps.
    """
    _require_extra()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"images must have the same shape, got {a.shape} and {b.shape}")
    shift, error = estimate_shift(a, b, upsample=upsample)
    b_aligned, valid = apply_shift(b, shift)
    window = overlap_slices(valid)
    if window is None:
        raise ValueError(f"shift {tuple(shift)} leaves no overlap between the images")
    a_c, b_c, ba_c = a[window], b[window], b_aligned[window]
    diff_before = a_c - b_c
    diff_after = a_c - ba_c
    if smooth_sigma > 0:
        diff_before = ndimage.gaussian_filter(diff_before, smooth_sigma)
        diff_after = ndimage.gaussian_filter(diff_after, smooth_sigma)
    return AlignResult(
        shift=(float(shift[0]), float(shift[1])),
        error=error,
        a=a_c,
        b=b_c,
        b_aligned=ba_c,
        diff_before=diff_before,
        diff_after=diff_after,
        rms_before=rms(diff_before),
        rms_after=rms(diff_after),
        window=window,
    )


# --------------------------------------------------------------------------- naming


def channel_slug(name: str) -> str:
    """``"Z"`` -> ``z``, ``"OC_M1_Freq._Shift"`` -> ``freqshift``, otherwise alphanumerics."""
    lowered = name.lower()
    if lowered == "z":
        return "z"
    if "freq" in lowered and "shift" in lowered:
        return "freqshift"
    return re.sub(r"[^0-9a-z]+", "", lowered) or "chan"


def short_id(path: str | Path) -> str:
    """Trailing digits of the file name (``Ag(111)1098.sxm`` -> ``1098``), else the stem."""
    stem = Path(path).stem
    digits = re.findall(r"\d+", stem)
    return digits[-1] if digits else stem


def nice_scalebar_nm(field_of_view_nm: float, fraction: float = 0.25) -> float:
    """Largest 1-2-5 x 10^k length (nm) not longer than ``fraction`` of the field of view."""
    target = field_of_view_nm * fraction
    if not math.isfinite(target) or target <= 0:
        return 1.0
    exponent = math.floor(math.log10(target))
    best = 10.0**exponent
    for mantissa in (1.0, 2.0, 5.0):
        candidate = mantissa * 10.0**exponent
        if candidate <= target:
            best = candidate
    return best


# --------------------------------------------------------------------------- figures


def _add_scalebar(ax, pixel_size_nm: float, width_px: int, height_px: int) -> None:
    from matplotlib import patheffects

    bar_nm = nice_scalebar_nm(width_px * pixel_size_nm)
    bar_px = bar_nm / pixel_size_nm
    x0 = width_px * 0.95 - bar_px
    y0 = height_px * 0.92
    outline = [patheffects.withStroke(linewidth=4, foreground="white")]
    ax.plot(
        [x0, x0 + bar_px],
        [y0, y0],
        color="black",
        lw=3,
        solid_capstyle="butt",
        path_effects=outline,
    )
    ax.text(
        x0 + bar_px / 2,
        y0 - height_px * 0.02,
        f"{bar_nm:g} nm",
        color="black",
        ha="center",
        va="bottom",
        fontsize=9,
        path_effects=outline,
    )


def save_difference(
    diff: np.ndarray,
    out_png: Path,
    *,
    vmax: float,
    pixel_size_nm: float,
    unit: str,
    quantity: str,
    title: str,
    cmap: str = "RdBu",
) -> Path:
    """Save one difference map with a symmetric diverging colour scale and a scale bar."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7, 4.2))
    ax = fig.add_subplot()
    im = ax.imshow(diff, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    _add_scalebar(ax, pixel_size_nm, diff.shape[1], diff.shape[0])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(f"Δ {quantity} ({unit})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return Path(out_png)


def save_overview(
    result: AlignResult,
    out_png: Path,
    *,
    vmax: float,
    pixel_size_nm: tuple[float, float],
    unit: str,
    quantity: str,
    name_a: str,
    name_b: str,
    cmap: str = "RdBu",
) -> Path:
    """Save the alignment overview: raw A / raw B / aligned B, differences, histogram."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, 7.5), layout="constrained")
    axes = fig.subplots(2, 3)
    both = np.concatenate([result.a.ravel(), result.b.ravel()])
    both = both[np.isfinite(both)]
    lo, hi = np.percentile(both, [2, 98]) if both.size else (0.0, 1.0)
    panels = [
        (axes[0, 0], result.a, f"A (raw)\n{name_a}"),
        (axes[0, 1], result.b, f"B (raw)\n{name_b}"),
        (axes[0, 2], result.b_aligned, "B aligned to A"),
    ]
    for ax, img, title in panels:
        im = ax.imshow(
            img, cmap="afmhot", vmin=lo, vmax=hi, origin="upper", interpolation="nearest"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=list(axes[0, :]), fraction=0.012, pad=0.01, label=f"{quantity} ({unit})")

    diffs = [
        (
            axes[1, 0],
            result.diff_before,
            f"A − B before alignment\nRMS = {result.rms_before:.4g} {unit}",
        ),
        (
            axes[1, 1],
            result.diff_after,
            f"A − B after alignment\nRMS = {result.rms_after:.4g} {unit}",
        ),
    ]
    for ax, img, title in diffs:
        im2 = ax.imshow(
            img, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper", interpolation="nearest"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        im2, ax=list(axes[1, :2]), fraction=0.012, pad=0.01, label=f"Δ {quantity} ({unit})"
    )

    ax = axes[1, 2]
    before = result.diff_before[np.isfinite(result.diff_before)].ravel()
    after = result.diff_after[np.isfinite(result.diff_after)].ravel()
    hist_range = (-vmax * 1.5, vmax * 1.5)
    ax.hist(before, bins=200, range=hist_range, alpha=0.55, color="gray", label="before")
    ax.hist(after, bins=200, range=hist_range, alpha=0.55, color="#c0392b", label="after")
    ax.set_title("Residual distribution", fontsize=10)
    ax.set_xlabel(f"Δ {quantity} ({unit})")
    ax.set_ylabel("pixels")
    ax.legend(fontsize=9)

    dy_px, dx_px = result.shift
    dx_nm, dy_nm = dx_px * pixel_size_nm[0], dy_px * pixel_size_nm[1]
    fig.suptitle(
        f"Sub-pixel alignment (upsampled cross-correlation): "
        f"Δx = {dx_px:+.3f} px ({dx_nm:+.3f} nm), Δy = {dy_px:+.3f} px ({dy_nm:+.3f} nm)"
        f"   |   residual RMS {result.rms_before:.4g} → {result.rms_after:.4g} {unit} "
        f"({result.improvement_percent:.1f}% lower)",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return Path(out_png)


# --------------------------------------------------------------------------- workflow


@dataclass
class AlignOutputs:
    """Files written by :func:`align_sxm_files`."""

    diff_ab: Path
    diff_ba: Path
    overview: Path
    npy: Path

    def __iter__(self):
        return iter((self.diff_ab, self.diff_ba, self.overview, self.npy))


def load_channel(path: Path, channel: str, direction: Direction) -> tuple[np.ndarray, str, str]:
    """Read one channel of an ``.sxm`` file; returns ``(image, channel name, unit)``."""
    sxm = SxmFile(path)
    ch = sxm.find_channel(channel)
    return sxm.read(ch, direction), ch.name, ch.unit


def align_sxm_files(
    path_a: Path,
    path_b: Path,
    *,
    channel: str = "Z",
    direction: Direction = "forward",
    upsample: int = 100,
    smooth_sigma: float = 0.0,
    clip: float = 99.0,
    cmap: str = "RdBu",
    outdir: Path | None = None,
) -> tuple[AlignResult, AlignOutputs]:
    """Align channel ``channel`` of two ``.sxm`` files and write the difference products.

    Outputs go to ``outdir`` (default: next to ``path_a``) and are named
    ``<channel>_<dir>_<idA>-<idB>.png``, ``..._<idB>-<idA>.png``,
    ``..._alignment_overview_<idA>_<idB>.png`` and ``..._<idA>-<idB>.npy``.
    """
    path_a, path_b = Path(path_a), Path(path_b)
    a, ch_name, unit = load_channel(path_a, channel, direction)
    b, _, _ = load_channel(path_b, channel, direction)
    header = SxmFile(path_a).header
    pixel_size = header.pixel_size_nm
    if unit == "m":
        a, b, unit = a * 1e9, b * 1e9, "nm"
    quantity = QUANTITY_BY_UNIT.get(unit, ch_name)
    logger.info(
        "channel %s (%s), unit %s, %s x %s px, %.4f x %.4f nm/px",
        ch_name,
        direction,
        unit,
        a.shape[1],
        a.shape[0],
        *pixel_size,
    )

    result = align_and_diff(a, b, upsample=upsample, smooth_sigma=smooth_sigma)
    dy_px, dx_px = result.shift
    logger.info(
        "shift (rows, cols) = (%+.4f, %+.4f) px = (dy %+.4f nm, dx %+.4f nm), error %.4g; "
        "common area %d x %d px; RMS %.4g -> %.4g %s (%.1f%% lower)",
        dy_px,
        dx_px,
        dy_px * pixel_size[1],
        dx_px * pixel_size[0],
        result.error,
        result.a.shape[1],
        result.a.shape[0],
        result.rms_before,
        result.rms_after,
        unit,
        result.improvement_percent,
    )

    outdir = Path(outdir) if outdir is not None else path_a.resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    id_a, id_b = short_id(path_a), short_id(path_b)
    prefix = f"{channel_slug(ch_name)}_{'fwd' if direction == 'forward' else 'bwd'}"
    tag = f"_sigma{smooth_sigma:g}" if smooth_sigma > 0 else ""
    outputs = AlignOutputs(
        diff_ab=outdir / f"{prefix}_{id_a}-{id_b}{tag}.png",
        diff_ba=outdir / f"{prefix}_{id_b}-{id_a}{tag}.png",
        overview=outdir / f"{prefix}_alignment_overview_{id_a}_{id_b}{tag}.png",
        npy=outdir / f"{prefix}_{id_a}-{id_b}{tag}.npy",
    )
    vmax = symmetric_limit(result.diff_after, clip)
    common = dict(vmax=vmax, pixel_size_nm=pixel_size[0], unit=unit, quantity=quantity, cmap=cmap)
    save_difference(
        result.diff_after,
        outputs.diff_ab,
        title=f"A − B ({id_a} − {id_b}), aligned {ch_name}",
        **common,
    )
    save_difference(
        -result.diff_after,
        outputs.diff_ba,
        title=f"B − A ({id_b} − {id_a}), aligned {ch_name}",
        **common,
    )
    save_overview(
        result,
        outputs.overview,
        vmax=vmax,
        pixel_size_nm=pixel_size,
        unit=unit,
        quantity=quantity,
        name_a=id_a,
        name_b=id_b,
        cmap=cmap,
    )
    np.save(outputs.npy, result.diff_after)
    return result, outputs


__all__ = [
    "QUANTITY_BY_UNIT",
    "AlignOutputs",
    "AlignResult",
    "align_and_diff",
    "align_sxm_files",
    "apply_shift",
    "channel_slug",
    "estimate_shift",
    "improvement_percent",
    "load_channel",
    "nice_scalebar_nm",
    "overlap_slices",
    "prepare_for_registration",
    "rms",
    "save_difference",
    "save_overview",
    "short_id",
    "symmetric_limit",
]
