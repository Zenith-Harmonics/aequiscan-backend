"""
Earth Climate, Atmosphere, Soil, and Surface Thermal Service.

Fetches live satellite-derived meteorological, radiative, and soil moisture data
from NASA POWER API and pedological properties from ISRIC SoilGrids v2.0.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SOILGRIDS_QUERY_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Conversion factors to W/m^2 for NASA POWER radiative flux parameters
_POWER_FLUX_TO_WM2: Dict[str, float] = {
    "W/m^2": 1.0,
    "W/m2": 1.0,
    "MJ/m^2/day": 1_000_000.0 / 86_400.0,  # ~11.574074 W/m^2
    "MJ/m2/day": 1_000_000.0 / 86_400.0,
    "kWh/m^2/day": 1_000.0 / 24.0,         # ~41.666667 W/m^2
    "kWh/m2/day": 1_000.0 / 24.0,
}


def _compute_unesco_aridity_index(
    t_mean_c: float,
    t_max_c: float,
    t_min_c: float,
    precip_mm_day: float,
    sw_kwh_m2_day: float,
) -> float:
    """
    Estimate UNESCO Aridity Index (P / PET) using the Hargreaves-Samani
    radiation-temperature Potential Evapotranspiration (PET) formulation.
    """
    td = max(0.5, t_max_c - t_min_c)
    # 1 kWh/m^2/day = 3.6 MJ/m^2/day; latent heat factor 0.408 -> 1.4688 mm/day
    ra_mm_day = max(0.5, sw_kwh_m2_day * 1.4688)
    pet_mm_day = max(0.1, 0.0023 * (t_mean_c + 17.8) * math.sqrt(td) * ra_mm_day)
    return max(0.0001, precip_mm_day / pet_mm_day)


def _convert_power_flux_to_wm2(value: Optional[float], unit_str: Optional[str]) -> Optional[float]:
    """Convert NASA POWER irradiance values to W/m^2 using header unit metadata."""
    if value is None:
        return None
    if unit_str:
        clean_unit = unit_str.strip()
        if clean_unit in _POWER_FLUX_TO_WM2:
            return value * _POWER_FLUX_TO_WM2[clean_unit]
    # NASA POWER daily/climatology ALLSKY_SFC_UVA & UVB default to MJ/m^2/day
    return value * _POWER_FLUX_TO_WM2["MJ/m^2/day"]


async def fetch_soilgrids_properties(
    lat: float,
    lon: float,
    client: Optional[httpx.AsyncClient] = None,
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """
    Query ISRIC SoilGrids v2.0 REST API for topsoil (0-5 cm) pH (H2O)
    and Soil Organic Carbon (SOC in g/kg). Returns None values over barren rock/ice.
    """
    params = [
        ("lon", round(lon, 4)),
        ("lat", round(lat, 4)),
        ("property", "phh2o"),
        ("property", "soc"),
        ("depth", "0-5cm"),
        ("value", "mean"),
    ]

    async def _do_request(http_client: httpx.AsyncClient) -> Dict[str, Any]:
        resp = await http_client.get(SOILGRIDS_QUERY_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()

        layers = payload.get("properties", {}).get("layers", [])
        extracted: Dict[str, Any] = {
            "ph": None,
            "soc_g_kg": None,
            "soilgrids_status": "ok",
        }
        for layer in layers:
            name = layer.get("name")
            depths = layer.get("depths", [])
            if not depths:
                continue
            mean_val = depths[0].get("values", {}).get("mean")
            if mean_val is None:
                continue
            if name == "phh2o":
                # SoilGrids stores pH * 10
                extracted["ph"] = round(float(mean_val) / 10.0, 2)
            elif name == "soc":
                # SoilGrids stores SOC in dg/kg (divide by 10 for g/kg)
                extracted["soc_g_kg"] = round(float(mean_val) / 10.0, 2)

        if extracted["ph"] is None and extracted["soc_g_kg"] is None:
            extracted["soilgrids_status"] = "no_soil_cover_or_masked"
        return extracted

    try:
        if client is not None:
            return await _do_request(client)
        async with httpx.AsyncClient(timeout=timeout_s) as new_client:
            return await _do_request(new_client)
    except httpx.TimeoutException:
        logger.warning("SoilGrids timed out for (%.4f, %.4f)", lat, lon)
        return {"ph": None, "soc_g_kg": None, "soilgrids_status": "timeout"}
    except Exception as exc:
        logger.warning("SoilGrids unavailable for (%.4f, %.4f): %s", lat, lon, exc)
        return {"ph": None, "soc_g_kg": None, "soilgrids_status": "error"}


async def fetch_earth_climate_and_atmosphere(
    lat: float,
    lon: float,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    timeout_s: float = 15.0,
) -> Dict[str, Any]:
    """
    Query NASA POWER API and ISRIC SoilGrids concurrently for live thermal,
    atmospheric, radiative, and astrobiological substrate parameters.
    """
    settings = get_settings()
    s_date = start_date.replace("-", "")
    e_date = end_date.replace("-", "")

    power_params = {
        "parameters": (
            "T2M,T2M_MAX,T2M_MIN,TS,PS,WS10M,RH2M,GWETTOP,"
            "PRECTOTCORR,AOD_55,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_UVA,ALLSKY_SFC_UVB"
        ),
        "community": "RE",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": s_date,
        "end": e_date,
        "format": "JSON",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            power_task = client.get(settings.nasa_power_base_url, params=power_params)
            soil_task = fetch_soilgrids_properties(lat=lat, lon=lon, client=client, timeout_s=timeout_s)
            power_resp, soil_props = await asyncio.gather(power_task, soil_task)

        power_resp.raise_for_status()
        data = power_resp.json()

        props = data.get("properties", {})
        param_dict = props.get("parameter", {})
        param_meta = data.get("header", {}).get("parameter", {})
        coords = data.get("geometry", {}).get("coordinates", [lon, lat, 0.0])
        elevation_m = float(coords[2]) if len(coords) >= 3 else 0.0

        def _mean_val(key: str) -> Optional[float]:
            series = param_dict.get(key, {})
            vals = [float(v) for v in series.values() if v is not None and float(v) > -900.0]
            return float(sum(vals) / len(vals)) if vals else None

        def _unit_for(key: str) -> Optional[str]:
            meta = param_meta.get(key)
            return meta.get("units") if isinstance(meta, dict) else None

        t_mean_c = _mean_val("T2M")
        t_skin_c = _mean_val("TS")
        t_max_c = _mean_val("T2M_MAX")
        t_min_c = _mean_val("T2M_MIN")
        ps_kpa = _mean_val("PS")
        ws_ms = _mean_val("WS10M")
        rh_pct = _mean_val("RH2M")
        gwettop = _mean_val("GWETTOP")  # Topsoil wetness [0..1]
        precip_mm_day = _mean_val("PRECTOTCORR")
        aod_55 = _mean_val("AOD_55")
        sw_raw = _mean_val("ALLSKY_SFC_SW_DWN")
        uva_raw = _mean_val("ALLSKY_SFC_UVA")
        uvb_raw = _mean_val("ALLSKY_SFC_UVB")

        if t_mean_c is None or t_max_c is None or t_min_c is None:
            logger.warning("NASA POWER returned incomplete temperature series for (%.4f, %.4f)", lat, lon)
            return {}

        # Keep T2M, T2M_MAX, T2M_MIN on the same 2m physical reference; report TS separately
        t_mean_k = t_mean_c + 273.15
        t_max_k = max(t_mean_k, t_max_c + 273.15)
        t_min_k = min(t_mean_k, t_min_c + 273.15)
        t_skin_k = (t_skin_c + 273.15) if t_skin_c is not None else t_mean_k
        diurnal_range_k = max(1.0, t_max_k - t_min_k)

        # Normalize shortwave irradiance to kWh/m^2/day for Hargreaves-Samani PET
        sw_wm2 = _convert_power_flux_to_wm2(sw_raw, _unit_for("ALLSKY_SFC_SW_DWN"))
        sw_kwh = (sw_wm2 * 24.0 / 1000.0) if sw_wm2 is not None else 5.0

        aridity_index = _compute_unesco_aridity_index(
            t_mean_c=t_mean_c,
            t_max_c=t_max_c,
            t_min_c=t_min_c,
            precip_mm_day=precip_mm_day if precip_mm_day is not None else 0.0,
            sw_kwh_m2_day=sw_kwh,
        )

        # Convert UVA + UVB to W/m^2 using NASA POWER header units (fallback: 4.5% of SW flux)
        uva_wm2 = _convert_power_flux_to_wm2(uva_raw, _unit_for("ALLSKY_SFC_UVA"))
        uvb_wm2 = _convert_power_flux_to_wm2(uvb_raw, _unit_for("ALLSKY_SFC_UVB"))
        if uva_wm2 is not None and uvb_wm2 is not None:
            uv_flux_wm2 = uva_wm2 + uvb_wm2
        elif sw_wm2 is not None:
            uv_flux_wm2 = sw_wm2 * 0.045
        else:
            uv_flux_wm2 = None

        # Topsoil wetness (GWETTOP) is distinct from atmospheric RH; conservative fallback if missing
        soil_wetness = float(min(1.0, max(0.0, gwettop))) if gwettop is not None else None
        if soil_wetness is not None:
            water_activity = soil_wetness
        elif rh_pct is not None:
            water_activity = float(min(1.0, max(0.0, (rh_pct / 100.0) * 0.5)))
        else:
            water_activity = None

        # Physical biosignature preservation model: cold temperatures + low water activity
        cold_factor = min(1.0, max(0.0, (305.0 - t_mean_k) / 50.0))
        arid_factor = min(1.0, max(0.0, 1.0 - (water_activity if water_activity is not None else 0.3)))
        preservation_potential = round(0.5 * cold_factor + 0.5 * arid_factor, 4)

        return {
            "source": "NASA POWER API & ISRIC SoilGrids v2.0",
            "elevation_m": round(elevation_m, 1),
            "thermal": {
                "t_mean_k": round(t_mean_k, 2),
                "t_skin_k": round(t_skin_k, 2),
                "t_max_k": round(t_max_k, 2),
                "t_min_k": round(t_min_k, 2),
                "day_temp_k": round(t_max_k, 2),
                "night_temp_k": round(t_min_k, 2),
                "diurnal_range_k": round(diurnal_range_k, 2),
            },
            "atmospheric": {
                "surface_pressure_pa": round(ps_kpa * 1000.0, 1) if ps_kpa is not None else None,
                "mean_wind_speed_ms": round(ws_ms, 2) if ws_ms is not None else None,
                "aridity_index": round(aridity_index, 5),
                "uv_flux_wm2": round(uv_flux_wm2, 2) if uv_flux_wm2 is not None else None,
                "dust_optical_depth": round(aod_55, 4) if aod_55 is not None else None,
            },
            "astrobiology": {
                "surface_soil_wetness": round(soil_wetness, 4) if soil_wetness is not None else None,
                "soil_water_activity_aw": round(water_activity, 4) if water_activity is not None else None,
                "relative_humidity_pct": round(rh_pct, 1) if rh_pct is not None else None,
                "organic_preservation_potential": preservation_potential,
                "ph": soil_props["ph"],
                "soil_organic_carbon_g_kg": soil_props["soc_g_kg"],
                "soilgrids_status": soil_props["soilgrids_status"],
            },
        }
    except Exception as exc:
        logger.warning("NASA POWER API query failed for (%.4f, %.4f): %s", lat, lon, exc)
        return {}