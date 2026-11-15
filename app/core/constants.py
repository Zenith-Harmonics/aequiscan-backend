"""
Domain constants: mineral taxonomy, keyword map, preloaded targets.

These are pure data – no I/O, no side-effects.
"""
from __future__ import annotations

# ── Harmonized cross-planetary mineral classes (6 diagnostic axes) ────
MINERAL_CLASSES: list[str] = [
    "mafic_olivine_pyroxene",
    "plagioclase_anorthosite",
    "phyllosilicates_clays",
    "sulfates_evaporites",
    "carbonates",
    "iron_oxides_hydroxides",
]

# ── EMIT USGS Spectral Library 7 keyword → harmonised class ─────────
EMIT_KEYWORD_MAP: dict[str, str] = {
    # Mafic
    "olivine": "mafic_olivine_pyroxene",
    "pyroxene": "mafic_olivine_pyroxene",
    "augite": "mafic_olivine_pyroxene",
    "diopside": "mafic_olivine_pyroxene",
    "enstatite": "mafic_olivine_pyroxene",
    "hornblende": "mafic_olivine_pyroxene",
    "actinolite": "mafic_olivine_pyroxene",
    # Plagioclase
    "plagioclase": "plagioclase_anorthosite",
    "anorthite": "plagioclase_anorthosite",
    "albite": "plagioclase_anorthosite",
    "oligoclase": "plagioclase_anorthosite",
    "labradorite": "plagioclase_anorthosite",
    "bytownite": "plagioclase_anorthosite",
    # Phyllosilicates
    "montmorillonite": "phyllosilicates_clays",
    "kaolinite": "phyllosilicates_clays",
    "illite": "phyllosilicates_clays",
    "chlorite": "phyllosilicates_clays",
    "smectite": "phyllosilicates_clays",
    "nontronite": "phyllosilicates_clays",
    "muscovite": "phyllosilicates_clays",
    "sericite": "phyllosilicates_clays",
    "dickite": "phyllosilicates_clays",
    # Sulfates / evaporites
    "gypsum": "sulfates_evaporites",
    "alunite": "sulfates_evaporites",
    "jarosite": "sulfates_evaporites",
    "epsomite": "sulfates_evaporites",
    "anhydrite": "sulfates_evaporites",
    "bassanite": "sulfates_evaporites",
    "copiapite": "sulfates_evaporites",
    # Carbonates
    "calcite": "carbonates",
    "dolomite": "carbonates",
    "magnesite": "carbonates",
    "siderite": "carbonates",
    # Iron oxides
    "hematite": "iron_oxides_hydroxides",
    "goethite": "iron_oxides_hydroxides",
    "ferrihydrite": "iron_oxides_hydroxides",
    "lepidocrocite": "iron_oxides_hydroxides",
}

# ── Preloaded planetary reference targets ────────────────────────────
PRELOADED_TARGETS: dict[str, dict] = {
    "earth": {
        "qaidam_dalangtan": {
            "name": "Qaidam Basin (Dalangtan Playa)",
            "lat": 38.4500,
            "lon": 91.4200,
            "radius_m": 2000.0,
            "body": "earth",
        },
    },
    "mars": {
        "gale_crater": {
            "name": "Gale Crater (Glen Torridon / Clay-Sulfate Unit)",
            "lat": -4.6500,
            "lon": 137.4000,
            "radius_m": 2000.0,
            "body": "mars",
        },
        "jezero_mons": {
            "name": "Jezero Mons",
            "lat": 16.5300,
            "lon": 77.4100,
            "radius_m": 2000.0,
            "body": "mars",
        },
    },
    "moon": {
        "mons_mouton_plateau": {
            "name": "Mons Mouton Plateau (Artemis Candidate)",
            "lat": -84.6000,
            "lon": 31.0000,
            "radius_m": 2000.0,
            "body": "moon",
        },
        "peak_near_cabeus_b": {
            "name": "Peak near Cabeus B",
            "lat": -82.48,
            "lon": -37.44,
            "radius_m": 2000.0,
            "body": "moon",
        },
        "haworth": {
            "name": "Haworth",
            "lat": -87.45,
            "lon": -5.10,
            "radius_m": 2000.0,
            "body": "moon",
        },
        "malapert_massif": {
            "name": "Malapert Massif",
            "lat": -85.99,
            "lon": 0.44,
            "radius_m": 2000.0,
            "body": "moon",
        },
        "mons_mouton": {
            "name": "Mons Mouton",
            "lat": -84.60,
            "lon": 24.50,
            "radius_m": 2000.0,
            "body": "moon",
        },
        "nobile_rim_1": {
            "name": "Nobile Rim 1",
            "lat": -85.32,
            "lon": 36.00,
            "radius_m": 2000.0,
            "body": "moon",
        },
        "nobile_rim_2": {
            "name": "Nobile Rim 2",
            "lat": -85.25,
            "lon": 52.50,
            "radius_m": 2000.0,
            "body": "moon",
        },
        "de_gerlache_rim_2": {
            "name": "de Gerlache Rim 2",
            "lat": -88.70,
            "lon": -73.10,
            "radius_m": 2000.0,
            "body": "moon",
        },
        "slater_plain": {
            "name": "Slater Plain",
            "lat": -88.18,
            "lon": 3.10,
            "radius_m": 2000.0,
            "body": "moon",
        },
    },
}

# ── Artemis III 9 candidate region IDs (quick lookup) ────────────────
ARTEMIS_III_REGIONS: list[str] = [
    "peak_near_cabeus_b",
    "haworth",
    "malapert_massif",
    "mons_mouton_plateau",
    "mons_mouton",
    "nobile_rim_1",
    "nobile_rim_2",
    "de_gerlache_rim_2",
    "slater_plain",
]
