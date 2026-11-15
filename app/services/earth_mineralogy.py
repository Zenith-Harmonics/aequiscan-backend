"""
Earth surface mineralogy extraction via NASA earthaccess + EMITL2BMIN.

Encapsulates all earthaccess I/O so the rest of the app stays testable
without NASA credentials.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np

try:
    import earthaccess
except ModuleNotFoundError:  # allow import for testing/simulation without NASA deps
    earthaccess = None  # type: ignore[assignment]

try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None  # type: ignore[assignment]

from pyproj import Geod

from app.core.config import get_settings
from app.core.constants import EMIT_KEYWORD_MAP, MINERAL_CLASSES
from app.core.exceptions import (
    AequiscanError,
    InsufficientPixelsError,
    LatitudeBoundsError,
    NoGranulesError,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_MIN_VALID_PIXELS = 15


def _login_once() -> None:
    """Ensure earthaccess is authenticated without interactive blocking."""
    if earthaccess is None:
        raise AequiscanError(
            "earthaccess is not installed. Install it with "
            "'pip install earthaccess' or pass 'simulate': true for offline testing."
        )
    if getattr(earthaccess.__auth__, "authenticated", False):
        return

    settings = get_settings()
    if settings.earthdata_token:
        os.environ["EARTHDATA_TOKEN"] = settings.earthdata_token
    if settings.earthdata_username and settings.earthdata_password:
        os.environ["EARTHDATA_USERNAME"] = settings.earthdata_username
        os.environ["EARTHDATA_PASSWORD"] = settings.earthdata_password

    try:
        earthaccess.login(strategy="environment")
    except Exception:
        try:
            earthaccess.login(strategy="netrc")
        except Exception as exc:
            raise AequiscanError(
                "NASA Earthdata Login credentials not found. Please set EARTHDATA_TOKEN "
                "(or EARTHDATA_USERNAME and EARTHDATA_PASSWORD) in .env or configure ~/.netrc, "
                "or pass 'simulate': true for offline testing."
            ) from exc


def get_simulated_earth_profile(
    lat: float,
    lon: float,
    radius_m: float = 2000.0,
) -> dict:
    """
    Return a precomputed EMIT mineral profile (e.g. Qaidam Basin Dalangtan Playa).

    Used for offline testing, UI prototyping, and development without NASA credentials.
    """
    if abs(lat) > 54.0:
        raise LatitudeBoundsError(
            f"Latitude {lat}° exceeds EMIT ISS orbital bounds (54°N–54°S). "
            "Use pre-2008 ASTER SWIR (AST_07) or HLSL30 band ratios for polar targets."
        )

    # Qaidam Playa / Hyper-arid evaporite default profile
    return {
        "source": "NASA EMITL2BMIN V001 (60 m; Precomputed Reference Baseline)",
        "n_pixels": int(np.pi * (radius_m / 60.0) ** 2),
        "n_valid_mineral_pixels": int(0.72 * np.pi * (radius_m / 60.0) ** 2),
        "mean_band_depth": 0.052,
        "top_raw_minerals": [
            ("Halite (USGS Lib 7)", 194),
            ("Gypsum (USGS Lib 7)", 112),
            ("Montmorillonite (USGS Lib 7)", 58),
            ("Illite (USGS Lib 7)", 39),
            ("Calcite (USGS Lib 7)", 27),
        ],
        "harmonized_profile": {
            "mafic_olivine_pyroxene": 0.09,
            "plagioclase_anorthosite": 0.04,
            "phyllosilicates_clays": 0.23,
            "sulfates_evaporites": 0.50,
            "carbonates": 0.06,
            "iron_oxides_hydroxides": 0.08,
        },
    }


def extract_emit_mineralogy(
    lat: float,
    lon: float,
    radius_m: float = 2000.0,
    max_granules: int = 10,
) -> dict:
    """
    Query EMITL2BMIN for the footprint and return a harmonised mineral profile.

    Raises
    ------
    LatitudeBoundsError  - if |lat| > 54
    NoGranulesError      - if CMR returns 0 granules
    InsufficientPixelsError - if no granule has >= 15 clear mineral pixels
    """
    if abs(lat) > 54.0:
        raise LatitudeBoundsError(
            f"Latitude {lat} exceeds EMIT ISS orbital bounds (54 N to 54 S). "
            "Use pre-2008 ASTER SWIR (AST_07) or HLSL30 band ratios for polar targets."
        )

    if earthaccess is None or xr is None:
        raise AequiscanError(
            "earthaccess and xarray must be installed for live EMIT extraction. "
            "Install them or pass 'simulate': true for offline testing."
        )

    _login_once()

    results = earthaccess.search_data(
        short_name="EMITL2BMIN",
        version="001",
        circle=(lon, lat, radius_m),
        temporal=("2022-08-09", "2026-08-31"),
        count=max_granules,
    )
    if not results:
        raise NoGranulesError(
            f"No EMITL2BMIN granules found at ({lat}, {lon})."
        )

    logger.info("Found %d EMITL2BMIN granules - scanning for clear coverage...", len(results))
    geod = Geod(ellps="WGS84")

    for idx, granule in enumerate(results):
        logger.debug("Checking granule %d...", idx)
        opened_files = earthaccess.open([granule])
        ds_root = xr.open_dataset(opened_files[0], engine="h5netcdf")
        ds_meta = xr.open_dataset(opened_files[0], group="mineral_metadata", engine="h5netcdf")
        ds_loc = xr.open_dataset(opened_files[0], group="location", engine="h5netcdf")

        lat_grid = ds_loc["lat"].values
        lon_grid = ds_loc["lon"].values

        valid_coords = np.isfinite(lat_grid) & np.isfinite(lon_grid)
        dist_m = np.full(lat_grid.shape, np.inf, dtype=np.float64)

        _, _, dist_m[valid_coords] = geod.inv(
            np.full(int(np.sum(valid_coords)), lon, dtype=np.float64),
            np.full(int(np.sum(valid_coords)), lat, dtype=np.float64),
            lon_grid[valid_coords].astype(np.float64),
            lat_grid[valid_coords].astype(np.float64),
        )

        footprint_mask = dist_m <= radius_m
        n_footprint_px = int(np.sum(footprint_mask))
        if n_footprint_px == 0:
            continue

        group1_id = ds_root["group_1_mineral_id"].values[footprint_mask]
        band_depth = ds_root["group_1_band_depth"].values[footprint_mask]

        # Fallback to group 2 where group 1 is unmatched
        if "group_2_mineral_id" in ds_root:
            g2_id = ds_root["group_2_mineral_id"].values[footprint_mask]
            g2_bd = ds_root["group_2_band_depth"].values[footprint_mask]
            use_g2 = (group1_id <= 0) & (g2_id > 0)
            group1_id = np.where(use_g2, g2_id, group1_id)
            band_depth = np.where(use_g2, g2_bd, band_depth)

        valid_px = (group1_id > 0) & np.isfinite(band_depth) & (band_depth > 0.01)
        if int(np.sum(valid_px)) < _MIN_VALID_PIXELS:
            logger.debug(
                "Granule %d: fewer than %d clear classified pixels - skipping.",
                idx,
                _MIN_VALID_PIXELS,
            )
            continue

        mineral_names: list[str] = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in ds_meta["name"].values
        ]
        id_to_name = {i: name for i, name in enumerate(mineral_names)}

        valid_ids = group1_id[valid_px]
        valid_depths = band_depth[valid_px]

        class_counts: dict[str, float] = {c: 0.0 for c in MINERAL_CLASSES}
        raw_detections: dict[str, int] = {}

        for m_id in valid_ids:
            m_name = id_to_name.get(int(m_id), "Unknown")
            raw_detections[m_name] = raw_detections.get(m_name, 0) + 1
            m_lower = m_name.lower()
            for kw, harm_class in EMIT_KEYWORD_MAP.items():
                if kw in m_lower:
                    class_counts[harm_class] += 1.0
                    break

        total_classified = sum(class_counts.values())
        profile = (
            {k: v / total_classified for k, v in class_counts.items()}
            if total_classified > 0
            else {k: 0.0 for k in MINERAL_CLASSES}
        )

        top_raw: list[tuple[str, int]] = sorted(
            raw_detections.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "source": f"NASA EMITL2BMIN V001 (60 m, 380-2500 nm; Granule #{idx})",
            "n_pixels": n_footprint_px,
            "n_valid_mineral_pixels": int(np.sum(valid_px)),
            "mean_band_depth": float(np.nanmean(valid_depths)),
            "top_raw_minerals": top_raw,
            "harmonized_profile": profile,
        }

    raise InsufficientPixelsError(
        "All returned EMIT granules had < 15 cloud-free mineral detections "
        "inside the target circle."
    )
