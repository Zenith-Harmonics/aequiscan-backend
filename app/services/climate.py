"""
Earth Climate, Atmosphere, Soil, and Surface Thermal Service.

Fetches live satellite-derived meteorological, radiative, and soil moisture data
from NASA POWER API and pedological properties from ISRIC SoilGrids v2.0.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SOILGRIDS_QUERY_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"


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
    # Convert shortwave irradiance (kWh/m^2/day) to equivalent evaporation (mm/day): 1 kWh/m^2 = 0.408 * 3.6 mm
    ra_mm_day = max(0.5, sw_kwh_m2_day * 1.4688)
    pet_mm_day = max(0.1, 0.0023 * (t_mean_c + 17.8) * math.sqrt(td) * ra_mm_day)
    return max(0.0001, precip_mm_day / pet_mm_day)


async def fetch_soilgrids_properties(
    lat: float,
    lon: float,
    timeout_s: float = 10.0,
) -> Dict[str, Optional[float]]:
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
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(SOILGRIDS_QUERY_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()

        layers = payload.get("properties", {}).get("layers", [])
        extracted: Dict[str, Optional[float]] = {"ph": None, "soc_g_kg": None}
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
        return extracted
    except Exception as exc:
        logger.debug("SoilGrids unavailable for (%.4f, %.4f): %s", lat, lon, exc)
        return {"ph": None, "soc_g_kg": None}


async def fetch_earth_climate_and_atmosphere(
    lat: float,
    lon: float,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    timeout_s: float = 15.0,
) -> Dict[str, Any]:
    """
    Query NASA POWER API and ISRIC SoilGrids for live thermal, atmospheric,
    radiative, and astrobiological substrate parameters.
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
            resp = await client.get(settings.nasa_power_base_url, params=power_params)
            resp.raise_for_status()
            data = resp.json()

        soil_props = await fetch_soilgrids_properties(lat=lat, lon=lon)

        props = data.get("properties", {})
        param_dict = props.get("parameter", {})
        coords = data.get("geometry", {}).get("coordinates", [lon, lat, 0.0])
        elevation_m = float(coords[2]) if len(coords) >= 3 else 0.0

        def _mean_val(key: str) -> Optional[float]:
            series = param_dict.get(key, {})
            vals = [float(v) for v in series.values() if v is not None and float(v) > -900.0]
            return float(sum(vals) / len(vals)) if vals else None

        t_mean_c = _mean_val("T2M")
        t_skin_c = _mean_val("TS") or t_mean_c
        t_max_c = _mean_val("T2M_MAX")
        t_min_c = _mean_val("T2M_MIN")
        ps_kpa = _mean_val("PS")
        ws_ms = _mean_val("WS10M")
        rh_pct = _mean_val("RH2M")
        gwettop = _mean_val("GWETTOP")  # Satellite surface soil wetness [0..1]
        precip_mm_day = _mean_val("PRECTOTCORR")
        aod_55 = _mean_val("AOD_55")
        sw_kwh = _mean_val("ALLSKY_SFC_SW_DWN")
        uva_wm2 = _mean_val("ALLSKY_SFC_UVA")
        uvb_wm2 = _mean_val("ALLSKY_SFC_UVB")

        if t_mean_c is None or t_max_c is None or t_min_c is None:
            return {}

        t_mean_k = (t_skin_c if t_skin_c is not None else t_mean_c) + 273.15
        t_max_k = t_max_c + 273.15
        t_min_k = t_min_c + 273.15
        diurnal_range_k = max(1.0, t_max_k - t_min_k)

        aridity_index = _compute_unesco_aridity_index(
            t_mean_c=t_mean_c,
            t_max_c=t_max_c,
            t_min_c=t_min_c,
            precip_mm_day=precip_mm_day if precip_mm_day is not None else 0.0,
            sw_kwh_m2_day=sw_kwh if sw_kwh is not None else 5.0,
        )

        # Direct UVA + UVB irradiance if available, otherwise 4.5% of total shortwave flux
        if uva_wm2 is not None and uvb_wm2 is not None:
            uv_flux_wm2 = uva_wm2 + uvb_wm2
        elif sw_kwh is not None:
            uv_flux_wm2 = (sw_kwh * 1000.0 / 24.0) * 0.045
        else:
            uv_flux_wm2 = None

        # Substrate water activity from SMAP-assimilated GWETTOP (fallback to RH/100)
        water_activity = (
            float(min(1.0, max(0.0, gwettop)))
            if gwettop is not None
            else (float(min(1.0, max(0.0, rh_pct / 100.0))) if rh_pct is not None else None)
        )

        # Physical biosignature preservation model: cold temperatures + low water activity + low SOC oxidation
        cold_factor = min(1.0, max(0.0, (305.0 - t_mean_k) / 50.0))
        arid_factor = min(1.0, max(0.0, 1.0 - (water_activity if water_activity is not None else 0.3)))
        preservation_potential = round(0.5 * cold_factor + 0.5 * arid_factor, 4)

        return {
            "source": "NASA POWER API & ISRIC SoilGrids v2.0",
            "elevation_m": round(elevation_m, 1),
            "thermal": {
                "t_mean_k": round(t_mean_k, 2),
                "t_max_k": round(t_max_k, 2),
                "t_min_k": round(t_min_k, 2),
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
                "soil_water_activity_aw": round(water_activity, 4) if water_activity is not None else None,
                "relative_humidity_pct": round(rh_pct, 1) if rh_pct is not None else None,
                "organic_preservation_potential": preservation_potential,
                "ph": soil_props["ph"],
                "soil_organic_carbon_g_kg": soil_props["soc_g_kg"],
            },
        }
    except Exception as exc:
        logger.warning("NASA POWER API query failed for (%.4f, %.4f): %s", lat, lon, exc)
        return {}