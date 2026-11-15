"""
Earth Optical, DEM, and Thermal COG Ingestion via Microsoft Planetary Computer STAC.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from app.core.cache import DiskCacheManager
from app.core.config import get_settings
from app.core.schemas import BoundingBox, SiteQuery

logger = logging.getLogger(__name__)

# BigEarthNet v2.0 10-band Sentinel-2 L2A order
S2_BANDS_10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def _utm_crs_from_bbox(bbox: BoundingBox) -> CRS:
    center_lon, center_lat = bbox.center()
    zone = int(math.floor((center_lon + 180.0) / 6.0)) + 1
    epsg = (32600 + zone) if center_lat >= 0.0 else (32700 + zone)
    return CRS.from_epsg(epsg)


class PlanetaryDataOrchestrator:
    """Streams and caches co-registered Earth STAC rasters onto a uniform UTM grid."""

    def __init__(
        self,
        cache: Optional[DiskCacheManager] = None,
        grid_size: int = 120,
        enable_live_stac: bool = True,
    ) -> None:
        self.cache = cache if cache is not None else DiskCacheManager()
        self.grid_size = grid_size
        self.enable_live_stac = enable_live_stac
        self.stac_url = get_settings().planetary_computer_stac_url

    def _reproject_asset(
        self,
        href: str,
        dst_transform: Any,
        dst_crs: CRS,
        resampling: Resampling = Resampling.bilinear,
    ) -> np.ndarray:
        dst = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)
        with rasterio.open(href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
        return dst

    def _fetch_earth_stac_rasters(self, site: SiteQuery) -> Dict[str, Any]:
        """Fetch DEM, Sentinel-2 10-band + RGB, and Landsat thermal rasters."""
        if not self.enable_live_stac:
            return {}

        cache_key = site.cache_key(f"earth_stac_{self.grid_size}")
        if self.cache.has_arrays(cache_key):
            cached = self.cache.load_arrays(cache_key)
            if cached is not None:
                return dict(cached)

        bbox_tuple = site.bbox.to_tuple()
        dst_crs = _utm_crs_from_bbox(site.bbox)
        utm_bounds = transform_bounds("EPSG:4326", dst_crs, *bbox_tuple)
        dst_transform = from_bounds(*utm_bounds, width=self.grid_size, height=self.grid_size)
        dt_range = f"{site.start_date}/{site.end_date}"

        out: Dict[str, np.ndarray] = {}

        try:
            client = pystac_client.Client.open(
                self.stac_url,
                modifier=planetary_computer.sign_inplace,
            )

            # 1. Copernicus DEM 30m
            dem_items = list(
                client.search(collections=["cop-dem-glo-30"], bbox=bbox_tuple, max_items=1).items()
            )
            if dem_items and "data" in dem_items[0].assets:
                dem = self._reproject_asset(dem_items[0].assets["data"].href, dst_transform, dst_crs)
                if np.any(np.isfinite(dem)):
                    med = float(np.nanmedian(dem))
                    out["dem"] = np.where(np.isfinite(dem), dem, med).astype(np.float32)

            # 2. Sentinel-2 L2A (least cloudy scene in window)
            s2_search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox_tuple,
                datetime=dt_range,
                query={"eo:cloud_cover": {"lt": 25}},
                max_items=5,
            )
            s2_items = sorted(
                s2_search.items(),
                key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)),
            )
            if s2_items:
                top_s2 = s2_items[0]
                if all(b in top_s2.assets for b in S2_BANDS_10):
                    bands = []
                    for b in S2_BANDS_10:
                        arr = self._reproject_asset(top_s2.assets[b].href, dst_transform, dst_crs)
                        med = float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else 0.0
                        bands.append(np.where(np.isfinite(arr), arr, med))
                    s2_stack = np.stack(bands, axis=0).astype(np.float32)
                    out["s2_multispectral"] = s2_stack

                    # RGB true-color uint8 from [B04, B03, B02]
                    rgb_f = np.stack([s2_stack[2], s2_stack[1], s2_stack[0]], axis=-1) / 3000.0
                    out["rgb_image"] = np.clip(rgb_f * 255.0, 0.0, 255.0).astype(np.uint8)

                    # Broadband shortwave albedo approximation (Liang 2001 visible/NIR/SWIR weights)
                    vis_swir = (
                        0.356 * s2_stack[0]
                        + 0.130 * s2_stack[2]
                        + 0.373 * s2_stack[6]
                        + 0.085 * s2_stack[8]
                        + 0.072 * s2_stack[9]
                    ) / 10000.0
                    out["albedo"] = np.clip(vis_swir, 0.01, 0.95).astype(np.float32)

            # 3. Landsat C2 L2 Surface Temperature (lwir11 -> Kelvin)
            ls_search = client.search(
                collections=["landsat-c2-l2"],
                bbox=bbox_tuple,
                datetime=dt_range,
                query={"eo:cloud_cover": {"lt": 25}},
                max_items=5,
            )
            ls_items = sorted(
                ls_search.items(),
                key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)),
            )
            if ls_items and "lwir11" in ls_items[0].assets:
                dn = self._reproject_asset(
                    ls_items[0].assets["lwir11"].href,
                    dst_transform,
                    dst_crs,
                    resampling=Resampling.nearest,
                )
                valid_dn = np.isfinite(dn) & (dn > 0.0)
                if np.any(valid_dn):
                    # Landsat Collection 2 ST_B10 scale=0.00341802, offset=149.0 (Kelvin)
                    temp_k = np.where(valid_dn, dn * 0.00341802 + 149.0, np.nan).astype(np.float32)
                    med_k = float(np.nanmedian(temp_k))
                    out["thermal_day_k"] = np.where(np.isfinite(temp_k), temp_k, med_k).astype(np.float32)

            if out:
                self.cache.save_arrays(cache_key, **out)
            return out

        except Exception as exc:
            logger.warning("Earth STAC raster fetch failed for %s: %s", site.site_id, exc)
            return out