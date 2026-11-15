"""
Analog Similarity Index (ASI) computation.

Implements the weighted Bray-Curtis similarity formula for compositional
mineralogical profiles:

    ASI(E, P) = 1 - sum_i(w_i * |x_E,i - x_P,i|) / sum_i(w_i * (x_E,i + x_P,i))

All inputs are expected to be [0, 1]-normalised fractional profiles.
Returns 0.0 for completely disjoint assemblages and 1.0 for identical profiles.
"""
from __future__ import annotations

import numpy as np

from app.core.constants import MINERAL_CLASSES
from app.schemas.mineralogy import ASIAxisDetail


def compute_mineral_asi(
    earth_profile: dict[str, float],
    planetary_profile: dict[str, float],
    weights: dict[str, float] | None = None,
) -> tuple[float, list[ASIAxisDetail]]:
    """
    Compute mineralogy-only ASI between an Earth tile and a planetary target.

    Parameters
    ----------
    earth_profile : dict[str, float]
        Fractional mineral abundances [0, 1] for the Earth analog site.
    planetary_profile : dict[str, float]
        Fractional mineral abundances [0, 1] for the planetary target.
    weights : dict[str, float] | None, optional
        Per-mineral-class weights (defaults to 1.0 for all classes).

    Returns
    -------
    asi_score : float
        Similarity score in [0, 1].
    axis_details : list[ASIAxisDetail]
        Per-class breakdown for UI reporting.
    """
    if weights is None:
        weights = {k: 1.0 for k in MINERAL_CLASSES}

    details: list[ASIAxisDetail] = []
    weighted_abs_diff = 0.0
    weighted_sum = 0.0

    for cls in MINERAL_CLASSES:
        e_val = float(earth_profile.get(cls, 0.0))
        p_val = float(planetary_profile.get(cls, 0.0))
        w = max(0.0, float(weights.get(cls, 1.0)))

        diff = abs(e_val - p_val)
        weighted_abs_diff += w * diff
        weighted_sum += w * (e_val + p_val)

        details.append(
            ASIAxisDetail(
                mineral_class=cls,
                earth_fraction=round(e_val, 4),
                planetary_fraction=round(p_val, 4),
                abs_delta=round(diff, 4),
                weight=w,
            )
        )

    if weighted_sum <= 0.0:
        asi = 0.0
    else:
        asi = 1.0 - (weighted_abs_diff / weighted_sum)

    return float(np.clip(asi, 0.0, 1.0)), details