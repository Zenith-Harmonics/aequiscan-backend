"""
Earth Sentinel-1 C-band Synthetic Aperture Radar (SAR) Ingestion & Processing.

Streams real Radiometrically Terrain-Corrected (RTC) VV/VH backscatter rasters
from Microsoft Planetary Computer STAC, applies a Lee speckle filter, converts
linear backscatter to decibels for ConvNeXtV2-S1 inference, and queries ASF DAAC
for official granule provenance.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import httpx
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.ndimage import uniform_filter

from app.core.cache import DiskCacheManager
from app.core.config import get_settings
from app.core.schemas import BoundingBox, SiteQuery

logger = logging.getLogger(__name__)

PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def lee_speckle_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a Minimum Mean Square Error (MMSE) Lee speckle filter in linear power domain
    while preserving finite NaN masks.
    """
    arr = np.asarray(img, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0.0)
    if not np.any(valid):
        return arr.astype(np.float32)

    fill_val = float(np.nanmedian(arr[valid]))
    filled = np.where(valid, arr, fill_val)

    mean = uniform_filter(filled, size=window_size)
    sq_mean = uniform_filter(filled ** 2, size=window_size)
    variance = np.maximum(sq_mean - mean ** 2, 0.0)

    overall_variance = float(np.var(filled[valid]))
    weights = variance / (variance + max(overall_variance, 1e-12))
    filtered = mean + weights * (filled - mean)
    return np.where(valid, filtered, np.nan).astype(np.float32)


def _utm_crs_from_bbox(bbox: BoundingBox) -> CRS:
    center_lon, center_lat = bbox.center()
    zone = int(math.floor((center_lon + 180.0) / 6.0)) + 1
    epsg = (32600 + zone) if center_lat >= 0.0 else (32700 + zone)
    return CRS.from_epsg(epsg)


def fetch_sentinel1_rtc_raster(
    site: SiteQuery,
    cache: Optional[DiskCacheManager] = None,
    grid_size: int = 120,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Download, Lee-filter, and reproject Sentinel-1 RTC [VV_dB, VH_dB] onto a (2, H, W) UTM grid.

    Returns
    -------
    sar_backscatter_db : (2, grid_size, grid_size) float32 array [VV_dB, VH_dB] or None
    cpr_linear : (grid_size, grid_size) float32 cross-polarization ratio (VH / VV) or None
    telemetry : dictionary of real scene statistics and STAC item metadata
    """
    cache_mgr = cache if cache is not None else DiskCacheManager()
    cache_key = site.cache_key(f"s1_rtc_{grid_size}")

    if cache_mgr.has_arrays(cache_key) and cache_mgr.has_json(cache_key):
        cached_arrs = cache_mgr.load_arrays(cache_key)
        cached_meta = cache_mgr.load_json(cache_key)
        if cached_arrs is not None and cached_meta is not None:
            return cached_arrs.get("sar_backscatter"), cached_arrs.get("radar_cpr"), cached_meta

    bbox_tuple = site.bbox.to_tuple()
    dst_crs = _utm_crs_from_bbox(site.bbox)
    utm_bounds = transform_bounds("EPSG:4326", dst_crs, *bbox_tuple)
    dst_transform = from_bounds(*utm_bounds, width=grid_size, height=grid_size)

    try:
        client = pystac_client.Client.open(
            PLANETARY_COMPUTER_STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
        search = client.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox_tuple,
            datetime=f"{site.start_date}/{site.end_date}",
            max_items=5,
        )
        items = [it for it in search.items() if "vv" in it.assets and "vh" in it.assets]
        if not items:
            logger.info("Planetary Computer STAC: 0 Sentinel-1 RTC scenes for %s", site.site_id)
            return None, None, {}

        top_item = items[0]

        def _read_pol(asset_key: str) -> np.ndarray:
            dst = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
            with rasterio.open(top_item.assets[asset_key].href) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
            return lee_speckle_filter(dst, window_size=5)

        vv_lin = _read_pol("vv")
        vh_lin = _read_pol("vh")

        valid_mask = np.isfinite(vv_lin) & (vv_lin > 1e-6) & np.isfinite(vh_lin) & (vh_lin > 1e-6)
        if not np.any(valid_mask):
            return None, None, {}

        vv_db = np.where(valid_mask, 10.0 * np.log10(np.maximum(vv_lin, 1e-6)), np.nan).astype(np.float32)
        vh_db = np.where(valid_mask, 10.0 * np.log10(np.maximum(vh_lin, 1e-6)), np.nan).astype(np.float32)
        cpr = np.where(valid_mask, np.clip(vh_lin / np.maximum(vv_lin, 1e-6), 0.0, 2.0), np.nan).astype(np.float32)

        # Fill isolated boundary NaNs with scene median so ConvNeXtV2 receives finite inputs
        vv_clean = np.where(np.isfinite(vv_db), vv_db, float(np.nanmedian(vv_db)))
        vh_clean = np.where(np.isfinite(vh_db), vh_db, float(np.nanmedian(vh_db)))
        sar_stack = np.stack([vv_clean, vh_clean], axis=0).astype(np.float32)

        telemetry: Dict[str, Any] = {
            "source": "Planetary Computer Sentinel-1 RTC (Lee-filtered C-band)",
            "stac_item_id": top_item.id,
            "datetime": top_item.datetime.isoformat() if top_item.datetime else None,
            "orbit_state": top_item.properties.get("sat:orbit_state"),
            "sar_vv_db": round(float(np.nanmean(vv_db)), 3),
            "sar_vh_db": round(float(np.nanmean(vh_db)), 3),
            "cpr_linear": round(float(np.nanmean(cpr)), 4),
        }

        cache_mgr.save_arrays(cache_key, sar_backscatter=sar_stack, radar_cpr=cpr)
        cache_mgr.save_json(cache_key, telemetry)
        return sar_stack, cpr, telemetry

    except Exception as exc:
        logger.warning("Sentinel-1 RTC raster ingestion failed for %s: %s", site.site_id, exc)
        return None, None, {}


async def fetch_earth_sentinel1_metadata(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """
    Query Alaska Satellite Facility (ASF) DAAC REST API for live Sentinel-1 GRD granule metadata.
    """
    settings = get_settings()
    bbox_str = f"{min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}"
    params = {
        "platform": "SENTINEL-1",
        "bbox": bbox_str,
        "processingLevel": "GRD_HD",
        "output": "json",
        "maxResults": 3,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(settings.asf_daac_base_url, params=params)
            resp.raise_for_status()

        data = resp.json()
        results_list = (
            data[0]
            if (isinstance(data, list) and len(data) > 0 and isinstance(data[0], list))
            else []
        )
        if not results_list:
            return {"source": "ASF DAAC Sentinel-1 GRD", "granules_matched": 0, "granule_id": None}

        top = results_list[0]
        return {
            "source": "Alaska Satellite Facility (ASF) DAAC",
            "granules_matched": len(results_list),
            "granule_id": top.get("granuleName"),
            "polarization": top.get("polarization"),
            "beam_mode": top.get("beamModeType"),
            "flight_direction": top.get("flightDirection"),
            "scene_date": top.get("startTime"),
        }
    except Exception as exc:
        logger.warning("ASF DAAC metadata query failed for bbox %s: %s", bbox_str, exc)
        return {"source": "ASF DAAC", "granules_matched": 0, "granule_id": None}