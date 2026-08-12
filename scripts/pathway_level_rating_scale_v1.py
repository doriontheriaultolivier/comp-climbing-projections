"""Research utilities for interpretable two-anchor pathway rating heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PathwayScaleError(ValueError):
    """Raised when a pathway scale cannot be supported by its reference field."""


@dataclass(frozen=True)
class PathwayAnchors:
    context: str
    semifinal_half_skill: float
    win_half_skill: float
    reference_definition: str

    def validate(self) -> None:
        values = np.asarray(
            [self.semifinal_half_skill, self.win_half_skill], dtype=float
        )
        if not np.isfinite(values).all():
            raise PathwayScaleError("anchor skills must be finite")
        if self.win_half_skill <= self.semifinal_half_skill:
            raise PathwayScaleError(
                "50% win skill must exceed 50% semifinal skill"
            )
        if not self.context.strip() or not self.reference_definition.strip():
            raise PathwayScaleError("context and reference definition are required")


def display_rating(skill: float | np.ndarray, anchors: PathwayAnchors) -> np.ndarray:
    """Map latent skill to 2000=50% semi and 3000=50% win.

    This is an interpretable affine display transform.  It does not calibrate
    probabilities and must only be applied after the underlying joint ranking
    distribution has passed chronological calibration.
    """

    anchors.validate()
    values = np.asarray(skill, dtype=float)
    if not np.isfinite(values).all():
        raise PathwayScaleError("skills must be finite")
    width = anchors.win_half_skill - anchors.semifinal_half_skill
    return 2000.0 + 1000.0 * (
        values - anchors.semifinal_half_skill
    ) / width


def skill_from_display_rating(
    rating: float | np.ndarray, anchors: PathwayAnchors
) -> np.ndarray:
    anchors.validate()
    values = np.asarray(rating, dtype=float)
    if not np.isfinite(values).all():
        raise PathwayScaleError("ratings must be finite")
    width = anchors.win_half_skill - anchors.semifinal_half_skill
    return anchors.semifinal_half_skill + (values - 2000.0) * width / 1000.0


def half_probability_skill(
    skill_grid: np.ndarray,
    mean_probabilities: np.ndarray,
) -> float:
    """Interpolate the latent skill at which a reference outcome reaches 50%."""

    skills = np.asarray(skill_grid, dtype=float)
    probabilities = np.asarray(mean_probabilities, dtype=float)
    if (
        skills.ndim != 1
        or probabilities.ndim != 1
        or len(skills) != len(probabilities)
        or len(skills) < 2
        or not np.isfinite(skills).all()
        or not np.isfinite(probabilities).all()
    ):
        raise PathwayScaleError("skill/probability grids must be aligned finite vectors")
    if not np.all(np.diff(skills) > 0):
        raise PathwayScaleError("skill grid must be strictly increasing")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise PathwayScaleError("probabilities must lie in [0, 1]")
    if np.any(np.diff(probabilities) < -1e-12):
        raise PathwayScaleError("reference probability must be monotone in skill")
    if probabilities[0] > 0.5 or probabilities[-1] < 0.5:
        raise PathwayScaleError("reference grid does not identify a 50% threshold")
    exact = np.flatnonzero(np.isclose(probabilities, 0.5, atol=1e-12))
    if exact.size:
        return float(skills[int(exact[0])])
    upper = int(np.searchsorted(probabilities, 0.5, side="right"))
    lower = upper - 1
    p0, p1 = probabilities[lower], probabilities[upper]
    if p1 <= p0:
        raise PathwayScaleError("flat probability segment cannot identify 50%")
    weight = (0.5 - p0) / (p1 - p0)
    return float(skills[lower] + weight * (skills[upper] - skills[lower]))
