"""
Analog Similarity Index (ASI) computation.

Implements the weighted normalised Euclidean distance formula:

    ASI(E, P) = 1 - sqrt(sum_i w_i (x_E,i - x_P,i)^2) / sqrt(sum_i w_i)

All inputs are expected to be [0, 1]-normalised fractional profiles.
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

    Returns
    -------
    asi_score : float in [0, 1]
    axis_details : per-class breakdown
    """
    if weights is None:
        weights = {k: 1.0 for k in MINERAL_CLASSES}

    details: list[ASIAxisDetail] = []
    weighted_sq_diff = 0.0
    weight_sum = 0.0

    for cls in MINERAL_CLASSES:
        e_val = earth_profile.get(cls, 0.0)
        p_val = planetary_profile.get(cls, 0.0)
        w = weights.get(cls, 1.0)
        diff = abs(e_val - p_val)
        weighted_sq_diff += w * (diff ** 2)
        weight_sum += w
        details.append(
            ASIAxisDetail(
                mineral_class=cls,
                earth_fraction=round(e_val, 4),
                planetary_fraction=round(p_val, 4),
                abs_delta=round(diff, 4),
                weight=w,
            )
        )

    asi = 1.0 - (np.sqrt(weighted_sq_diff) / np.sqrt(weight_sum))
    return float(np.clip(asi, 0.0, 1.0)), details
