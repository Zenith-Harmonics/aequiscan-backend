"""
Unit tests for the ASI similarity engine (no I/O, no credentials).
"""
from __future__ import annotations

import random

import pytest

from app.core.constants import MINERAL_CLASSES
from app.services.similarity import compute_mineral_asi


def _uniform(val: float) -> dict[str, float]:
    return {cls: val for cls in MINERAL_CLASSES}


def test_identical_profiles_score_1():
    profile = {
        "mafic_olivine_pyroxene": 0.5,
        "plagioclase_anorthosite": 0.1,
        "phyllosilicates_clays": 0.2,
        "sulfates_evaporites": 0.1,
        "carbonates": 0.05,
        "iron_oxides_hydroxides": 0.05,
    }
    score, details = compute_mineral_asi(profile, profile)
    assert score == pytest.approx(1.0)
    for d in details:
        assert d.abs_delta == pytest.approx(0.0, abs=1e-6)


def test_maximally_different_profiles():
    earth = _uniform(0.0)
    planet = _uniform(1.0)
    score, _ = compute_mineral_asi(earth, planet)
    assert score == pytest.approx(0.0)


def test_custom_weights_affect_score():
    earth = {
        "mafic_olivine_pyroxene": 1.0,
        "plagioclase_anorthosite": 0.5,
        **{c: 0.0 for c in MINERAL_CLASSES[2:]},
    }
    planet = {
        "mafic_olivine_pyroxene": 0.0,
        "plagioclase_anorthosite": 0.5,
        **{c: 0.0 for c in MINERAL_CLASSES[2:]},
    }

    # Equal weights
    s1, _ = compute_mineral_asi(earth, planet, weights={c: 1.0 for c in MINERAL_CLASSES})
    # Heavy weight on the differing axis
    heavy = {c: 1.0 for c in MINERAL_CLASSES}
    heavy["mafic_olivine_pyroxene"] = 10.0
    s2, _ = compute_mineral_asi(earth, planet, weights=heavy)

    assert s2 < s1  # heavier weight on the divergent axis -> lower score


def test_score_bounded_0_1():
    random.seed(42)
    for _ in range(50):
        e = {c: random.random() for c in MINERAL_CLASSES}
        p = {c: random.random() for c in MINERAL_CLASSES}
        score, _ = compute_mineral_asi(e, p)
        assert 0.0 <= score <= 1.0


def test_partial_profile_defaults_missing_to_zero():
    earth = {"mafic_olivine_pyroxene": 0.8}
    planet = {"mafic_olivine_pyroxene": 0.8}
    score, details = compute_mineral_asi(earth, planet)
    assert score == pytest.approx(1.0)
    # All non-specified classes should default to 0.0
    for d in details:
        if d.mineral_class != "mafic_olivine_pyroxene":
            assert d.earth_fraction == pytest.approx(0.0)
