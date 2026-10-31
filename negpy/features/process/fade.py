"""Fade Restoration: the restoration matrix a Slide as captured renders with, and the
per-image estimate of the dye survivals."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.system.i18n import mark, tr
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    effective_crosstalk_matrix,
    measure_neutral_axis_from_log,
)
from negpy.features.process.models import ProcessConfig
from negpy.features.process.path import RenderPath, render_path

#: Below this density spread a channel's neutral axis is too flat to be signal.
SPREAD_FLOOR = 0.02
#: Spreads within this fraction of the largest agree: no evidence of differential fade.
AGREEMENT_TOLERANCE = 0.05
#: Bound on an implied ratio, a reciprocal pair; the factor stays well conditioned.
RATIO_BOUNDS = (0.2, 5.0)
#: Bound on `fade_ratio_r`, a fraction of the original density, so it cannot exceed 1.0.
RED_SURVIVAL_BOUNDS = (0.05, 1.0)


def resolve_fade_matrix(ratio_r: float, ratio_g: float, ratio_b: float) -> np.ndarray:
    """Restoration operator for a faded dye set: the inverse of diag(ar, ar*ag, ar*ab).

    Not row-normalized: fade changes each layer's neutral density, and that is the signal.
    `ratio_g` and `ratio_b` are survival ratios to red; `ratio_r` is red's absolute survival.
    """
    ar, ag, ab = float(ratio_r), float(ratio_g), float(ratio_b)
    return np.diag([1.0 / ar, 1.0 / (ar * ag), 1.0 / (ar * ab)])


def fade_matrix(process: ProcessConfig) -> Optional[np.ndarray]:
    """The restoration matrix, or None at unit survival or off the TRANSFER path: a Positive
    file is already a positive, so a dye-density scale has no meaning there."""
    ratios = (process.fade_ratio_r, process.fade_ratio_g, process.fade_ratio_b)
    if render_path(process) is not RenderPath.TRANSFER or all(r == 1.0 for r in ratios):
        return None
    return resolve_fade_matrix(*ratios)


def unmix_with_fade(process: ProcessConfig, process_mode: Optional[str]) -> Optional[np.ndarray]:
    """The log-density matrix a render applies: the crosstalk unmix, then the fade restoration of
    the densities it recovers (`fade @ unmix`). None when neither applies."""
    unmix = effective_crosstalk_matrix(process, process_mode)
    fade = fade_matrix(process)
    if fade is None:
        return unmix
    return fade if unmix is None else fade @ unmix


#: The return shape of `measure_neutral_axis_from_log`: midtone, shadow, highlight or None, confidence.
NeutralAxisRefs = tuple[tuple[float, float, float], tuple[float, float, float], object, float]


def measure_neutral_axis_ratios(grid: ImageBuffer) -> tuple[Optional[NeutralAxisRefs], str]:
    """(refs, reason) from the transparency meter's grid, metered against this frame's own
    bounds: the fixed transfer window finds no grays in a faded slide. `refs` is None, with a
    reason, when no trustworthy neutral axis exists."""
    refs = measure_neutral_axis_from_log(grid, _measured_slide_bounds(grid), None, 0.0)
    if refs is None:
        return None, tr("No reliable grays found on this frame.")
    return refs, ""


def _measured_slide_bounds(grid: ImageBuffer) -> LogNegativeBounds:
    """This frame's own bounds in the slide orientation (floor above ceiling): the luma span
    from the base clip, each channel's offset from its extreme percentiles. The negative
    analyzer does not fit, since its dense end is chroma-gated for an orange mask."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    def _pct(p: float) -> list:
        return [float(np.percentile(grid[:, :, ch], p)) for ch in range(3)]

    base = float(EXPOSURE_CONSTANTS["base_luma_clip"])
    luma_f, luma_c = _pct(100.0 - base), _pct(base)
    color_f, color_c = _pct(100.0 - 0.00001), _pct(0.00001)
    mean_f, mean_c = sum(luma_f) / 3.0, sum(luma_c) / 3.0
    mid_f, mid_c = sorted(color_f)[1], sorted(color_c)[1]
    floors = tuple(mean_f + (color_f[ch] - mid_f) for ch in range(3))
    ceils = tuple(mean_c + (color_c[ch] - mid_c) for ch in range(3))
    return LogNegativeBounds(floors, ceils)


class EstimateMethod(StrEnum):
    """MIDTONE reads the midtone grays' own densities. SPREAD reads each channel's midtone-to-shadow
    spread, which cancels a per-channel density offset but follows a flat, noisy shadow."""

    MIDTONE = "midtone"
    SPREAD = "spread"


@dataclass(frozen=True)
class FadeEstimate:
    """Survivals from one Estimate. red is None when the method leaves Red Survival as it is;
    reason is "" for a real estimate (see fade_ratios_from_neutral_axis)."""

    red: Optional[float]
    green: float
    blue: float
    reason: str


def fade_ratios_from_neutral_axis(refs: Optional[NeutralAxisRefs], method: str = EstimateMethod.SPREAD) -> tuple[float, float, str]:
    """(ratio_g, ratio_b, reason) from neutral references made by `measure_neutral_axis_ratios`.

    Ratios are relative to red, the fade matrix's reference channel. `reason` is "" for a real
    estimate, else the fail-closed condition that fired, with 1.0 returned for both."""
    if refs is None:
        return 1.0, 1.0, tr("No grays to measure.")
    mid, shadow = refs[0], refs[1]
    if method == EstimateMethod.MIDTONE:
        # The refs are log transmittance, so density is the negative.
        spreads = tuple(-float(mid[ch]) for ch in range(3))
    else:
        spreads = tuple(float(mid[ch]) - float(shadow[ch]) for ch in range(3))
    r, g, b = spreads
    abs_spreads = (abs(r), abs(g), abs(b))
    if min(abs_spreads) < SPREAD_FLOOR:
        return 1.0, 1.0, tr("The grays in one channel are too flat to measure.")
    if max(abs_spreads) - min(abs_spreads) <= AGREEMENT_TOLERANCE * max(abs_spreads):
        return 1.0, 1.0, tr("The three layers faded equally, so there is nothing to estimate.")
    ratio_g = g / r
    ratio_b = b / r
    lo, hi = RATIO_BOUNDS
    if not (lo <= ratio_g <= hi) or not (lo <= ratio_b <= hi):
        clamped_g = min(max(ratio_g, lo), hi)
        clamped_b = min(max(ratio_b, lo), hi)
        sentences = [
            tr(template, raw=f"{raw:.1f}", lo=f"{lo:g}", hi=f"{hi:g}", clamped=f"{c:.1f}")
            for template, raw, c in (
                (mark("Green read {raw}, outside {lo} to {hi}, so it is set to {clamped}."), ratio_g, clamped_g),
                (mark("Blue read {raw}, outside {lo} to {hi}, so it is set to {clamped}."), ratio_b, clamped_b),
            )
            if raw != c
        ]
        if ratio_g > hi or ratio_b > hi:
            sentences.append(tr("Red faded more than this range covers. Lower Red Survival by eye."))
        reason = " ".join(sentences)
        return clamped_g, clamped_b, reason
    return ratio_g, ratio_b, ""


def estimate_fade(grid: ImageBuffer) -> dict[EstimateMethod, FadeEstimate]:
    """Both methods' estimates from one measurement of the grays. `grid` is the transparency
    meter's grid with the crosstalk unmix and no fade (`TransparencyBaseProcessor`). MIDTONE
    also sets Red Survival so the best-kept layer lands at 1.0; a ratio above 1 says red faded most."""
    refs, reason = measure_neutral_axis_ratios(grid)
    out = {}
    for method in EstimateMethod:
        if refs is None:
            out[method] = FadeEstimate(None, 1.0, 1.0, reason)
            continue
        g, b, why = fade_ratios_from_neutral_axis(refs, method)
        red = None
        if method == EstimateMethod.MIDTONE and not (why and g == 1.0 and b == 1.0):
            lo, hi = RED_SURVIVAL_BOUNDS
            red = min(max(1.0 / max(g, b, 1.0), lo), hi)
        out[method] = FadeEstimate(red, g, b, why)
    return out
