"""
Unified Live Planetary Data Provider.

Populates SiteDataBundle instances exclusively from real satellite and orbital APIs
(Microsoft Planetary Computer STAC, NASA POWER, ISRIC SoilGrids, NASA EMIT, NASA PDS ODE)
backed by atomic disk caching.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from app.core.cache import DiskCacheManager
from app.core.schemas import PlanetBody, SiteDataBundle, SiteQuery
from app.models.feature_extractor import MultiModalFeatureExtractor
from app.models.space_aligner import LatentSpaceAligner
from app.models.weights_manager import ModelWeightsManager
from app.services.climate import fetch_earth_climate_and_atmosphere
from app.services.earth_mineralogy import extract_emit_mineralogy
from app.services.earth_stac import PlanetaryDataOrchestrator
from app.services.pds_planetary import fetch_mars_orbital_telemetry, fetch_moon_orbital_telemetry
from app.services.sentinel1_sar import fetch_sentinel1_rtc_raster

logger = logging.getLogger(__name__)


class UnifiedPlanetaryDataProvider:
    """
    Resolves Earth, Mars, and Moon SiteQuery requests into SiteDataBundle containers
    using live remote-sensing APIs, local model inference, and atomic disk caching.
    """

    def __init__(
        self,
        cache: Optional[DiskCacheManager] = None,
        grid_size: int = 120,
        aligner: Optional[LatentSpaceAligner] = None,
    ) -> None:
        self.cache = cache if cache is not None else DiskCacheManager()
        self.grid_size = grid_size
        self.stac_orchestrator = PlanetaryDataOrchestrator(
            cache=self.cache,
            grid_size=grid_size,
            enable_live_stac=True,
        )
        self.weights_manager = ModelWeightsManager()
        self.feature_extractor = MultiModalFeatureExtractor(auto_fetch=False)
        self.aligner = aligner

    def _populate_available_embeddings(self, bundle: SiteDataBundle) -> None:
        """Run neural feature extraction only for models already present on disk."""
        try:
            if bundle.rgb_image is not None and self.weights_manager.is_downloaded("mars_remoteclip"):
                bundle.embeddings["clip_768"] = self.feature_extractor.extract_optical_clip(
                    bundle.rgb_image
                )[0]

            if bundle.site.body == PlanetBody.EARTH:
                if bundle.sar_backscatter is not None and self.weights_manager.is_downloaded("sentinel_1"):
                    bundle.embeddings["s1_1024"] = self.feature_extractor.extract_sentinel1(
                        bundle.sar_backscatter
                    )[0]

                if bundle.s2_multispectral is not None and self.weights_manager.is_downloaded("sentinel_2"):
                    bundle.embeddings["s2_1024"] = self.feature_extractor.extract_sentinel2(
                        bundle.s2_multispectral
                    )[0]

                if self.aligner is not None and self.aligner.input_dim is not None:
                    if (
                        self.aligner.input_dim == 2048
                        and "s1_1024" in bundle.embeddings
                        and "s2_1024" in bundle.embeddings
                    ):
                        fused = np.concatenate([bundle.embeddings["s1_1024"], bundle.embeddings["s2_1024"]])
                        bundle.embeddings["earth_aligned_768"] = self.aligner.project_earth_to_target_space(
                            fused
                        )[0]
                    elif self.aligner.input_dim == 1024 and "s2_1024" in bundle.embeddings:
                        bundle.embeddings["earth_aligned_768"] = self.aligner.project_earth_to_target_space(
                            bundle.embeddings["s2_1024"]
                        )[0]
        except Exception as exc:
            logger.warning("Embedding extraction skipped for %s: %s", bundle.site.site_id, exc)

    async def load_bundle_async(self, site: SiteQuery) -> SiteDataBundle:
        """Asynchronously populate a SiteDataBundle from live orbital/satellite services."""
        center_lon, center_lat = site.bbox.center()

        if site.body == PlanetBody.EARTH:
            bundle = SiteDataBundle(site=site, pixel_resolution_m=30.0)

            stac_task = asyncio.to_thread(self.stac_orchestrator._fetch_earth_stac_rasters, site)
            s1_task = asyncio.to_thread(fetch_sentinel1_rtc_raster, site, self.cache, self.grid_size)
            climate_task = fetch_earth_climate_and_atmosphere(
                lat=center_lat,
                lon=center_lon,
                start_date=site.start_date,
                end_date=site.end_date,
            )
            emit_task = asyncio.to_thread(extract_emit_mineralogy, center_lat, center_lon)

            stac_res, s1_res, climate_res, emit_res = await asyncio.gather(
                stac_task, s1_task, climate_task, emit_task, return_exceptions=True
            )

            if isinstance(stac_res, dict):
                bundle.dem = stac_res.get("dem")
                bundle.s2_multispectral = stac_res.get("s2_multispectral")
                bundle.rgb_image = stac_res.get("rgb_image")
                bundle.albedo = stac_res.get("albedo")
                bundle.thermal_day_k = stac_res.get("thermal_day_k")

            if isinstance(s1_res, tuple):
                sar_stack, radar_cpr, _ = s1_res
                bundle.sar_backscatter = sar_stack
                bundle.radar_cpr = radar_cpr

            if isinstance(climate_res, dict) and climate_res:
                atm = climate_res.get("atmospheric", {})
                astro = climate_res.get("astrobiology", {})
                therm = climate_res.get("thermal", {})

                for key in (
                    "aridity_index",
                    "surface_pressure_pa",
                    "uv_flux_wm2",
                    "mean_wind_speed_ms",
                    "dust_optical_depth",
                ):
                    val = atm.get(key)
                    if val is not None:
                        bundle.atmospheric_scalars[key] = float(val)

                for key in (
                    "soil_water_activity_aw",
                    "organic_preservation_potential",
                    "ph",
                    "soil_organic_carbon_g_kg",
                ):
                    val = astro.get(key)
                    if val is not None:
                        bundle.astrobiology_scalars[key] = float(val)

                if bundle.thermal_day_k is None and therm.get("t_max_k") is not None:
                    bundle.thermal_day_k = np.full(
                        (self.grid_size, self.grid_size), float(therm["t_max_k"]), dtype=np.float32
                    )
                if bundle.thermal_night_k is None and therm.get("t_min_k") is not None:
                    bundle.thermal_night_k = np.full(
                        (self.grid_size, self.grid_size), float(therm["t_min_k"]), dtype=np.float32
                    )

            if isinstance(emit_res, dict) and isinstance(emit_res.get("harmonized_profile"), dict):
                bundle.mineral_indices["harmonized_profile"] = emit_res["harmonized_profile"]  # type: ignore[assignment]

            await asyncio.to_thread(self._populate_available_embeddings, bundle)
            return bundle

        if site.body == PlanetBody.MARS:
            bundle = SiteDataBundle(site=site, pixel_resolution_m=18.0)
            mars_data = await fetch_mars_orbital_telemetry(lat=center_lat, lon=center_lon)

            bundle.rgb_image = mars_data.get("rgb_image")
            if isinstance(mars_data.get("harmonized_profile"), dict):
                bundle.mineral_indices["harmonized_profile"] = mars_data["harmonized_profile"]  # type: ignore[assignment]

            therm = mars_data.get("thermal", {})
            if therm.get("t_max_k") is not None:
                bundle.thermal_day_k = np.full(
                    (self.grid_size, self.grid_size), float(therm["t_max_k"]), dtype=np.float32
                )
            if therm.get("t_min_k") is not None:
                bundle.thermal_night_k = np.full(
                    (self.grid_size, self.grid_size), float(therm["t_min_k"]), dtype=np.float32
                )

            atm = mars_data.get("atmospheric", {})
            if atm.get("surface_pressure_pa") is not None:
                bundle.atmospheric_scalars["surface_pressure_pa"] = float(atm["surface_pressure_pa"])
            if atm.get("dust_optical_depth") is not None:
                bundle.atmospheric_scalars["dust_optical_depth"] = float(atm["dust_optical_depth"])

            await asyncio.to_thread(self._populate_available_embeddings, bundle)
            return bundle

        # PlanetBody.MOON
        bundle = SiteDataBundle(site=site, pixel_resolution_m=20.0)
        moon_data = await fetch_moon_orbital_telemetry(lat=center_lat, lon=center_lon)

        bundle.rgb_image = moon_data.get("rgb_image")
        if moon_data.get("albedo") is not None:
            bundle.albedo = np.full(
                (self.grid_size, self.grid_size), float(moon_data["albedo"]), dtype=np.float32
            )
        if isinstance(moon_data.get("harmonized_profile"), dict):
            bundle.mineral_indices["harmonized_profile"] = moon_data["harmonized_profile"]  # type: ignore[assignment]

        therm = moon_data.get("thermal", {})
        if therm.get("t_max_k") is not None:
            bundle.thermal_day_k = np.full(
                (self.grid_size, self.grid_size), float(therm["t_max_k"]), dtype=np.float32
            )
        if therm.get("t_min_k") is not None:
            bundle.thermal_night_k = np.full(
                (self.grid_size, self.grid_size), float(therm["t_min_k"]), dtype=np.float32
            )

        if moon_data.get("cpr_mean") is not None:
            bundle.radar_cpr = np.full(
                (self.grid_size, self.grid_size), float(moon_data["cpr_mean"]), dtype=np.float32
            )

        await asyncio.to_thread(self._populate_available_embeddings, bundle)
        return bundle