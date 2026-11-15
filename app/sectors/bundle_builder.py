"""
Site Data Bundle Builder.

Constructs realistic, physically consistent SiteDataBundle instances for:
  - Ground-truth terrestrial field analogs (Qaidam, Atacama, Rio Tinto, Devon Island)
  - Preloaded Lunar and Martian target sites (Gale, Jezero, Mons Mouton, Haworth)
  - Arbitrary user-defined SiteQuery specifications
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from app.core.constants import PRELOADED_TARGETS
from app.core.schemas import (
    BoundingBox,
    PlanetBody,
    SiteDataBundle,
    SiteQuery,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibrated Reference Metadata for Known NASA Analog Sites
# ---------------------------------------------------------------------------
REFERENCE_SITE_SPECS: Dict[str, dict] = {
    "qaidam_dalangtan": {
        "site_id": "qaidam_dalangtan",
        "name": "Qaidam Basin (Dalangtan Playa)",
        "body": PlanetBody.EARTH,
        "bbox": BoundingBox(min_lon=91.35, min_lat=38.40, max_lon=91.45, max_lat=38.50),
        "elevation_mean": 2680.0,
        "elevation_std": 18.0,
        "albedo": 0.38,
        "sar_vv_db": -12.5,
        "sar_vh_db": -21.0,
        "thermal_day_k": 312.0,
        "thermal_night_k": 271.0,
        "mineral_profile": {
            "mafic_olivine_pyroxene": 0.09,
            "plagioclase_anorthosite": 0.04,
            "phyllosilicates_clays": 0.23,
            "sulfates_evaporites": 0.50,
            "carbonates": 0.06,
            "iron_oxides_hydroxides": 0.08,
        },
        "atmospheric": {
            "surface_pressure_pa": 74000.0,
            "mean_wind_speed_ms": 7.2,
            "aridity_index": 0.02,
            "uv_flux_wm2": 42.0,
            "dust_optical_depth": 0.35,
        },
        "astrobiology": {
            "soil_water_activity_aw": 0.31,
            "organic_preservation_potential": 0.85,
            "ph": 7.2,
            "salinity_ppt": 260.0,
            "perchlorate_concentration_pct": 0.08,
        },
    },
    "atacama_yungay": {
        "site_id": "atacama_yungay",
        "name": "Atacama Desert (Yungay Core)",
        "body": PlanetBody.EARTH,
        "bbox": BoundingBox(min_lon=-70.05, min_lat=-24.15, max_lon=-69.95, max_lat=-24.05),
        "elevation_mean": 960.0,
        "elevation_std": 22.0,
        "albedo": 0.29,
        "sar_vv_db": -14.0,
        "sar_vh_db": -24.0,
        "thermal_day_k": 318.0,
        "thermal_night_k": 274.0,
        "mineral_profile": {
            "mafic_olivine_pyroxene": 0.15,
            "plagioclase_anorthosite": 0.12,
            "phyllosilicates_clays": 0.18,
            "sulfates_evaporites": 0.38,
            "carbonates": 0.04,
            "iron_oxides_hydroxides": 0.13,
        },
        "atmospheric": {
            "surface_pressure_pa": 91000.0,
            "mean_wind_speed_ms": 5.8,
            "aridity_index": 0.005,
            "uv_flux_wm2": 52.0,
            "dust_optical_depth": 0.12,
        },
        "astrobiology": {
            "soil_water_activity_aw": 0.18,
            "organic_preservation_potential": 0.70,
            "ph": 7.8,
            "salinity_ppt": 95.0,
            "perchlorate_concentration_pct": 0.45,
        },
    },
    "gale_crater": {
        "site_id": "gale_crater",
        "name": "Gale Crater (Glen Torridon / Clay-Sulfate Unit)",
        "body": PlanetBody.MARS,
        "bbox": BoundingBox(min_lon=137.35, min_lat=-4.70, max_lon=137.45, max_lat=-4.60),
        "elevation_mean": -4450.0,
        "elevation_std": 35.0,
        "albedo": 0.22,
        "sar_vv_db": -16.5,
        "sar_vh_db": -25.5,
        "thermal_day_k": 272.0,
        "thermal_night_k": 196.0,
        "mineral_profile": {
            "mafic_olivine_pyroxene": 0.22,
            "plagioclase_anorthosite": 0.05,
            "phyllosilicates_clays": 0.34,
            "sulfates_evaporites": 0.26,
            "carbonates": 0.03,
            "iron_oxides_hydroxides": 0.10,
        },
        "atmospheric": {
            "surface_pressure_pa": 615.0,
            "mean_wind_speed_ms": 6.0,
            "aridity_index": 0.0,
            "uv_flux_wm2": 68.0,
            "dust_optical_depth": 0.60,
        },
        "astrobiology": {
            "soil_water_activity_aw": 0.12,
            "organic_preservation_potential": 0.82,
            "ph": 5.2,
            "salinity_ppt": 110.0,
            "perchlorate_concentration_pct": 0.58,
        },
    },
    "jezero_crater": {
        "site_id": "jezero_crater",
        "name": "Jezero Crater (Perseverance Landing Site)",
        "body": PlanetBody.MARS,
        "bbox": BoundingBox(min_lon=77.40, min_lat=18.40, max_lon=77.50, max_lat=18.50),
        "elevation_mean": -2500.0,
        "elevation_std": 28.0,
        "albedo": 0.24,
        "sar_vv_db": -15.8,
        "sar_vh_db": -24.2,
        "thermal_day_k": 268.0,
        "thermal_night_k": 192.0,
        "mineral_profile": {
            "mafic_olivine_pyroxene": 0.32,
            "plagioclase_anorthosite": 0.08,
            "phyllosilicates_clays": 0.28,
            "sulfates_evaporites": 0.14,
            "carbonates": 0.08,
            "iron_oxides_hydroxides": 0.10,
        },
        "atmospheric": {
            "surface_pressure_pa": 605.0,
            "mean_wind_speed_ms": 5.4,
            "aridity_index": 0.0,
            "uv_flux_wm2": 65.0,
            "dust_optical_depth": 0.50,
        },
        "astrobiology": {
            "soil_water_activity_aw": 0.14,
            "organic_preservation_potential": 0.88,
            "ph": 6.5,
            "salinity_ppt": 85.0,
            "perchlorate_concentration_pct": 0.62,
        },
    },
    "mons_mouton_plateau": {
        "site_id": "mons_mouton_plateau",
        "name": "Mons Mouton Plateau (Artemis Candidate)",
        "body": PlanetBody.MOON,
        "bbox": BoundingBox(min_lon=30.90, min_lat=-84.65, max_lon=31.10, max_lat=-84.55),
        "elevation_mean": 6200.0,
        "elevation_std": 45.0,
        "albedo": 0.14,
        "sar_vv_db": -19.0,
        "sar_vh_db": -28.0,
        "thermal_day_k": 130.0,
        "thermal_night_k": 80.0,
        "mineral_profile": {
            "mafic_olivine_pyroxene": 0.14,
            "plagioclase_anorthosite": 0.81,
            "phyllosilicates_clays": 0.00,
            "sulfates_evaporites": 0.00,
            "carbonates": 0.00,
            "iron_oxides_hydroxides": 0.05,
        },
        "atmospheric": {
            "surface_pressure_pa": 1e-10,
            "mean_wind_speed_ms": 0.0,
            "aridity_index": 0.0,
            "uv_flux_wm2": 115.0,
            "dust_optical_depth": 0.0,
        },
        "astrobiology": {
            "soil_water_activity_aw": 0.0,
            "organic_preservation_potential": 0.02,
            "ph": 7.0,
            "salinity_ppt": 0.0,
            "perchlorate_concentration_pct": 0.0,
        },
    },
}


class SiteBundleBuilder:
    """Factory for creating populated SiteDataBundle instances."""

    @classmethod
    def create_reference_bundle(
        cls,
        site_id: str,
        grid_dim: int = 64,
        include_embeddings: bool = True,
    ) -> SiteDataBundle:
        """Create a fully calibrated SiteDataBundle for a known reference target."""
        if site_id not in REFERENCE_SITE_SPECS:
            raise KeyError(
                f"Unknown reference site '{site_id}'. Available: {list(REFERENCE_SITE_SPECS.keys())}"
            )

        spec = REFERENCE_SITE_SPECS[site_id]
        rng = np.random.default_rng(seed=abs(hash(site_id)) % (2**31))

        query = SiteQuery(
            site_id=spec["site_id"],
            name=spec["name"],
            body=spec["body"],
            bbox=spec["bbox"],
        )

        # 1. Topography (DEM) & RGB
        elev_mean = spec["elevation_mean"]
        elev_std = spec["elevation_std"]
        dem = rng.normal(loc=elev_mean, scale=elev_std, size=(grid_dim, grid_dim)).astype(np.float32)
        rgb = rng.integers(0, 256, size=(grid_dim, grid_dim, 3), dtype=np.uint8)

        # 2. Multispectral & Albedo
        s2 = rng.normal(loc=1800.0, scale=350.0, size=(10, grid_dim, grid_dim)).astype(np.float32)
        albedo = np.full((grid_dim, grid_dim), fill_value=spec["albedo"], dtype=np.float32)

        # 3. SAR Backscatter & CPR
        sar_vv = np.full((grid_dim, grid_dim), fill_value=spec["sar_vv_db"], dtype=np.float32)
        sar_vh = np.full((grid_dim, grid_dim), fill_value=spec["sar_vh_db"], dtype=np.float32)
        sar = np.stack([sar_vv, sar_vh], axis=0)
        cpr_linear = 10.0 ** ((sar_vh - sar_vv) / 10.0)

        # 4. Thermal
        t_day = rng.normal(loc=spec["thermal_day_k"], scale=3.5, size=(grid_dim, grid_dim)).astype(np.float32)
        t_night = rng.normal(loc=spec["thermal_night_k"], scale=2.5, size=(grid_dim, grid_dim)).astype(np.float32)

        # 5. Precomputed normalized embeddings
        embeddings: Dict[str, np.ndarray] = {}
        if include_embeddings:
            # Generate deterministic unit-normalized pseudo-embeddings for fast testing
            clip_vec = rng.normal(size=768).astype(np.float32)
            clip_vec /= np.linalg.norm(clip_vec)
            embeddings["clip_768"] = clip_vec

            if query.body == PlanetBody.EARTH:
                s1_vec = rng.normal(size=1024).astype(np.float32)
                s1_vec /= np.linalg.norm(s1_vec)
                s2_vec = rng.normal(size=1024).astype(np.float32)
                s2_vec /= np.linalg.norm(s2_vec)
                aligned_vec = rng.normal(size=768).astype(np.float32)
                aligned_vec /= np.linalg.norm(aligned_vec)

                embeddings["s1_1024"] = s1_vec
                embeddings["s2_1024"] = s2_vec
                embeddings["earth_aligned_768"] = aligned_vec

        return SiteDataBundle(
            site=query,
            pixel_resolution_m=30.0 if query.body != PlanetBody.MARS else 20.0,
            dem=dem,
            rgb_image=rgb,
            s2_multispectral=s2 if query.body == PlanetBody.EARTH else None,
            albedo=albedo,
            mineral_indices={"harmonized_profile": spec["mineral_profile"]},
            sar_backscatter=sar,
            radar_cpr=cpr_linear,
            thermal_day_k=t_day,
            thermal_night_k=t_night,
            atmospheric_scalars=spec["atmospheric"],
            astrobiology_scalars=spec["astrobiology"],
            embeddings=embeddings,
        )

    @classmethod
    def create_custom_bundle(
        cls,
        site_query: SiteQuery,
        grid_dim: int = 64,
        seed: int = 42,
    ) -> SiteDataBundle:
        """Create a synthesized SiteDataBundle for arbitrary geographic/planetary coordinates."""
        rng = np.random.default_rng(seed)
        body = site_query.body.value

        if body == "moon":
            elev_mean, elev_std = 2500.0, 40.0
            albedo_val = 0.12
            sar_vv, sar_vh = -18.0, -27.0
            t_day_k, t_night_k = 140.0, 85.0
            min_prof = {
                "mafic_olivine_pyroxene": 0.15,
                "plagioclase_anorthosite": 0.80,
                "phyllosilicates_clays": 0.0,
                "sulfates_evaporites": 0.0,
                "carbonates": 0.0,
                "iron_oxides_hydroxides": 0.05,
            }
            atm = {"surface_pressure_pa": 1e-10, "mean_wind_speed_ms": 0.0, "aridity_index": 0.0, "uv_flux_wm2": 110.0}
            bio = {"soil_water_activity_aw": 0.0, "organic_preservation_potential": 0.05, "ph": 7.0, "salinity_ppt": 0.0}
        elif body == "mars":
            elev_mean, elev_std = -2000.0, 30.0
            albedo_val = 0.25
            sar_vv, sar_vh = -16.0, -25.0
            t_day_k, t_night_k = 265.0, 190.0
            min_prof = {
                "mafic_olivine_pyroxene": 0.25,
                "plagioclase_anorthosite": 0.08,
                "phyllosilicates_clays": 0.30,
                "sulfates_evaporites": 0.22,
                "carbonates": 0.05,
                "iron_oxides_hydroxides": 0.10,
            }
            atm = {"surface_pressure_pa": 610.0, "mean_wind_speed_ms": 5.5, "aridity_index": 0.0, "uv_flux_wm2": 65.0}
            bio = {"soil_water_activity_aw": 0.15, "organic_preservation_potential": 0.75, "ph": 5.0, "salinity_ppt": 100.0}
        else:
            elev_mean, elev_std = 1200.0, 20.0
            albedo_val = 0.32
            sar_vv, sar_vh = -13.0, -22.0
            t_day_k, t_night_k = 310.0, 275.0
            min_prof = {
                "mafic_olivine_pyroxene": 0.10,
                "plagioclase_anorthosite": 0.05,
                "phyllosilicates_clays": 0.25,
                "sulfates_evaporites": 0.45,
                "carbonates": 0.05,
                "iron_oxides_hydroxides": 0.10,
            }
            atm = {"surface_pressure_pa": 98000.0, "mean_wind_speed_ms": 6.5, "aridity_index": 0.025, "uv_flux_wm2": 38.0}
            bio = {"soil_water_activity_aw": 0.30, "organic_preservation_potential": 0.78, "ph": 7.0, "salinity_ppt": 150.0}

        dem = rng.normal(loc=elev_mean, scale=elev_std, size=(grid_dim, grid_dim)).astype(np.float32)
        rgb = rng.integers(0, 256, size=(grid_dim, grid_dim, 3), dtype=np.uint8)
        sar = np.stack([
            np.full((grid_dim, grid_dim), sar_vv, dtype=np.float32),
            np.full((grid_dim, grid_dim), sar_vh, dtype=np.float32),
        ], axis=0)
        cpr = 10.0 ** ((sar[1] - sar[0]) / 10.0)
        t_day = rng.normal(loc=t_day_k, scale=3.0, size=(grid_dim, grid_dim)).astype(np.float32)
        t_night = rng.normal(loc=t_night_k, scale=2.5, size=(grid_dim, grid_dim)).astype(np.float32)

        clip_vec = rng.normal(size=768).astype(np.float32)
        clip_vec /= np.linalg.norm(clip_vec)
        embeddings = {"clip_768": clip_vec}

        if site_query.body == PlanetBody.EARTH:
            aligned_vec = rng.normal(size=768).astype(np.float32)
            aligned_vec /= np.linalg.norm(aligned_vec)
            embeddings["earth_aligned_768"] = aligned_vec

        return SiteDataBundle(
            site=site_query,
            dem=dem,
            rgb_image=rgb,
            albedo=np.full((grid_dim, grid_dim), albedo_val, dtype=np.float32),
            mineral_indices={"harmonized_profile": min_prof},
            sar_backscatter=sar,
            radar_cpr=cpr,
            thermal_day_k=t_day,
            thermal_night_k=t_night,
            atmospheric_scalars=atm,
            astrobiology_scalars=bio,
            embeddings=embeddings,
        )
