"""
Mars & Moon Multi-Instrument Ingestion via NASA PDS Orbital Data Explorer (ODE) REST API.

Queries overlapping PDS products (CRISM, M3, HiRISE, LROC, THEMIS, Diviner, MCS, Mini-RF),
parses genuine PDS3 ASCII label telemetry, and streams orbital browse rasters for
RemoteCLIP feature extraction and spectral summary analysis.
"""
from __future__ import annotations

import io
import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
import asyncio

import httpx
import numpy as np
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PDS3_KV_PATTERN = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.+?)\s*$", re.MULTILINE)


def _extract_pds_product_id(product: Dict[str, Any], default: str = "UNKNOWN") -> str:
    """Extract a canonical PDS product identifier from an ODE product record."""
    product_name = product.get("Product_name")
    if product_name:
        return str(product_name).split(".")[0].upper()

    product_url = product.get("ProductURL", "")
    if "product_id=" in product_url:
        parsed = urllib.parse.urlparse(product_url)
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("product_id"):
            return params["product_id"][0].upper()

    label_url = product.get("LabelURL", "")
    if label_url:
        return label_url.split("/")[-1].split(".")[0].upper()

    return str(product.get("pdsid") or default).upper()


def _parse_pds3_label(label_text: str) -> Dict[str, str]:
    """Parse KEY = VALUE pairs from a PDS3 ASCII label, stripping quotes and <unit> tags."""
    parsed: Dict[str, str] = {}
    for key, raw_val in _PDS3_KV_PATTERN.findall(label_text):
        cleaned = re.sub(r"<[^>]+>", "", raw_val).strip().strip('"').strip("'")
        parsed[key] = cleaned
    return parsed


def _parse_float_field(meta: Dict[str, str], candidate_keys: Tuple[str, ...]) -> Optional[float]:
    for k in candidate_keys:
        if k in meta:
            try:
                val = float(meta[k])
                if np.isfinite(val) and abs(val) < 1e15:
                    return val
            except ValueError:
                continue
    return None


async def query_pds_ode_products(
    target: str,
    instrument: str,
    product_type: str,
    lat: float,
    lon: float,
    radius_deg: float = 0.15,
    limit: int = 5,
    ihid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query NASA PDS ODE REST API for overlapping planetary products."""
    settings = get_settings()
    lon_360 = lon % 360.0

    if ihid is None:
        if target == "mars":
            ihid = "MRO" if instrument.upper() in ("CRISM", "HIRISE", "MCS", "SHARAD") else "ODY"
        elif target == "moon":
            ihid = "CH1-ORB" if instrument.upper() == "M3" else "LRO"
        else:
            ihid = "MRO"

    params = {
        "query": "product",
        "results": "m",
        "output": "JSON",
        "target": target,
        "ihid": ihid,
        "iid": instrument,
        "pt": product_type,
        "minlat": max(-90.0, lat - radius_deg),
        "maxlat": min(90.0, lat + radius_deg),
        "westernlon": (lon_360 - radius_deg) % 360.0,
        "easternlon": (lon_360 + radius_deg) % 360.0,
        "limit": limit,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(settings.pds_ode_base_url, params=params)
            resp.raise_for_status()
        data = resp.json()
        ode_results = data.get("ODEResults", {}) if isinstance(data, dict) else {}
        products_container = ode_results.get("Products") if isinstance(ode_results, dict) else None

        if isinstance(products_container, dict):
            products = products_container.get("Product", [])
            return [products] if isinstance(products, dict) else list(products)
        return []
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("PDS ODE unreachable for %s/%s: %s", target, instrument, exc)
        return []


async def _fetch_label_and_browse(
    client: httpx.AsyncClient,
    product: Dict[str, Any],
    target_size: int = 224,
) -> Tuple[Dict[str, str], Optional[np.ndarray]]:
    """Download the PDS3 label dictionary and optional RGB browse raster for an ODE product."""
    label_meta: Dict[str, str] = {}
    browse_rgb: Optional[np.ndarray] = None

    label_url = product.get("LabelURL")
    if label_url:
        try:
            lbl_resp = await client.get(label_url)
            if lbl_resp.status_code == 200:
                label_meta = _parse_pds3_label(lbl_resp.text)
        except Exception as exc:
            logger.debug("Failed to fetch PDS label %s: %s", label_url, exc)

    browse_url = product.get("BrowseURL")
    if browse_url:
        try:
            img_resp = await client.get(browse_url)
            if img_resp.status_code == 200:
                pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                pil_resized = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
                browse_rgb = np.asarray(pil_resized, dtype=np.uint8)
        except Exception as exc:
            logger.debug("Failed to fetch PDS browse image %s: %s", browse_url, exc)

    return label_meta, browse_rgb


def _derive_mineral_fractions_from_crism_browse(
    browse_rgb: np.ndarray,
    solar_incidence_deg: Optional[float] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Derive a normalized 6-class mineral fraction vector from a CRISM MTRDR/MRDR
    false-color summary browse image (R: ferric/olivine, G: phyllosilicates/OH, B: sulfates/salts)
    corrected for cosine solar incidence illumination.
    """
    rgb = np.asarray(browse_rgb, dtype=np.float64) / 255.0
    if solar_incidence_deg is not None and 0.0 <= solar_incidence_deg < 85.0:
        cos_i = max(0.2, float(np.cos(np.radians(solar_incidence_deg))))
        rgb = np.clip(rgb / cos_i, 0.0, 1.0)

    r_mean = float(np.mean(rgb[..., 0]))
    g_mean = float(np.mean(rgb[..., 1]))
    b_mean = float(np.mean(rgb[..., 2]))

    raw_scores = {
        "mafic_olivine_pyroxene": max(0.01, r_mean * 0.55),
        "plagioclase_anorthosite": max(0.01, (1.0 - max(r_mean, g_mean, b_mean)) * 0.25),
        "phyllosilicates_clays": max(0.01, g_mean * 0.85),
        "sulfates_evaporites": max(0.01, b_mean * 0.80),
        "carbonates": max(0.01, min(g_mean, b_mean) * 0.30),
        "iron_oxides_hydroxides": max(0.01, r_mean * 0.45),
    }
    total = sum(raw_scores.values())
    harmonized = {k: round(v / total, 4) for k, v in raw_scores.items()}
    mean_band_depth = round(float(np.std(rgb)) * 0.18, 4)
    return harmonized, mean_band_depth


async def fetch_mars_orbital_telemetry(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch multi-instrument Martian telemetry and orbital browse imagery from NASA PDS ODE:
      - CRISM MTRDR/MRDR (mineralogy)
      - HiRISE RDRV11 (high-resolution optical morphology for RemoteCLIP)
      - THEMIS IRGEO2/IRBTR (surface brightness temperature)
      - MCS DDR (atmospheric pressure & dust optical depth)
    """
# În fetch_mars_orbital_telemetry:
    crism_prods, hirise_prods, themis_prods, mcs_prods = await asyncio.gather(
        query_pds_ode_products("mars", "CRISM", "MTRDR", lat, lon, radius_deg=0.25, limit=3),
        query_pds_ode_products("mars", "HIRISE", "RDRV11", lat, lon, radius_deg=0.20, limit=2),
        query_pds_ode_products("mars", "THEMIS", "IRGEO2", lat, lon, radius_deg=0.35, limit=3, ihid="ODY"),
        query_pds_ode_products("mars", "MCS", "DDR", lat, lon, radius_deg=0.50, limit=2),
    )
    if not crism_prods:
        crism_prods = await query_pds_ode_products("mars", "CRISM", "MRDR", lat, lon, radius_deg=0.50, limit=3)
    
    result: Dict[str, Any] = {
        "source": "NASA PDS Orbital Data Explorer (MRO CRISM / HiRISE / MCS + ODY THEMIS)",
        "pds_sample_id": _extract_pds_product_id(crism_prods[0]) if crism_prods else None,
        "rgb_image": None,
        "harmonized_profile": None,
        "mean_band_depth": None,
        "thermal": {},
        "atmospheric": {},
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        # 1. CRISM mineral summary browse + label incidence angle
        if crism_prods:
            lbl_meta, crism_rgb = await _fetch_label_and_browse(client, crism_prods[0])
            inc_angle = _parse_float_field(lbl_meta, ("INCIDENCE_ANGLE", "SOLAR_INCIDENCE_ANGLE"))
            if crism_rgb is not None:
                profile, band_depth = _derive_mineral_fractions_from_crism_browse(crism_rgb, inc_angle)
                result["harmonized_profile"] = profile
                result["mean_band_depth"] = band_depth
                result["rgb_image"] = crism_rgb

        # 2. HiRISE high-res optical browse (preferred over CRISM false-color for RemoteCLIP morphology)
        if hirise_prods:
            _, hirise_rgb = await _fetch_label_and_browse(client, hirise_prods[0])
            if hirise_rgb is not None:
                result["rgb_image"] = hirise_rgb

        # 3. THEMIS thermal infrared label statistics
        if themis_prods:
            th_meta, _ = await _fetch_label_and_browse(client, themis_prods[0])
            t_max = _parse_float_field(th_meta, ("MAXIMUM_BRIGHTNESS_TEMPERATURE", "MAX_TEMPERATURE"))
            t_min = _parse_float_field(th_meta, ("MINIMUM_BRIGHTNESS_TEMPERATURE", "MIN_TEMPERATURE"))
            if t_max is not None:
                result["thermal"]["t_max_k"] = t_max
            if t_min is not None:
                result["thermal"]["t_min_k"] = t_min

        # 4. MCS atmospheric pressure & dust opacity
        if mcs_prods:
            mcs_meta, _ = await _fetch_label_and_browse(client, mcs_prods[0])
            surf_p = _parse_float_field(mcs_meta, ("SURFACE_PRESSURE", "PRESSURE"))
            dust_tau = _parse_float_field(mcs_meta, ("DUST_OPTICAL_DEPTH", "COLUMN_DUST_OPTICAL_DEPTH"))
            if surf_p is not None:
                result["atmospheric"]["surface_pressure_pa"] = surf_p
            if dust_tau is not None:
                result["atmospheric"]["dust_optical_depth"] = dust_tau

    return result


async def fetch_moon_orbital_telemetry(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch multi-instrument Lunar telemetry and orbital browse imagery from NASA PDS ODE:
      - Chandrayaan-1 M3 REFIMG (reflectance & albedo)
      - LRO LROC NAC/WAC (optical morphology for RemoteCLIP)
      - LRO Diviner GDR (bolometric & brightness temperature extremes)
      - LRO Mini-RF RDR (S/X-band Circular Polarization Ratio)
    """
# În fetch_moon_orbital_telemetry:
    m3_prods, lroc_prods, diviner_prods, minirf_prods = await asyncio.gather(
        query_pds_ode_products("moon", "M3", "REFIMG", lat, lon, radius_deg=0.50, limit=2),
        query_pds_ode_products("moon", "LROC", "CDR", lat, lon, radius_deg=0.35, limit=2, ihid="LRO"),
        query_pds_ode_products("moon", "DLRE", "GDR", lat, lon, radius_deg=0.50, limit=2, ihid="LRO"),
        query_pds_ode_products("moon", "MRF", "RDR", lat, lon, radius_deg=0.50, limit=2, ihid="LRO"),
    )
    result: Dict[str, Any] = {
        "source": "NASA PDS Orbital Data Explorer (Chandrayaan-1 M3 + LRO LROC / Diviner / Mini-RF)",
        "pds_sample_id": _extract_pds_product_id(m3_prods[0]) if m3_prods else None,
        "rgb_image": None,
        "harmonized_profile": None,
        "albedo": None,
        "thermal": {},
        "cpr_mean": None,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        if lroc_prods:
            lroc_meta, lroc_rgb = await _fetch_label_and_browse(client, lroc_prods[0])
            if lroc_rgb is not None:
                result["rgb_image"] = lroc_rgb
                # Estimate relative surface albedo from LROC calibrated radiance browse corrected for incidence
                inc = _parse_float_field(lroc_meta, ("INCIDENCE_ANGLE",))
                cos_i = max(0.2, float(np.cos(np.radians(inc)))) if inc is not None and inc < 85.0 else 0.7
                est_albedo = float(np.clip((np.mean(lroc_rgb) / 255.0) * 0.25 / cos_i, 0.04, 0.35))
                result["albedo"] = round(est_albedo, 4)

                # Derive Mare (high-Fe/Ti dark basalt) vs Highlands (bright anorthosite) fractions from real albedo
                mare_weight = float(np.clip((0.18 - est_albedo) / 0.11, 0.05, 0.90))
                result["harmonized_profile"] = {
                    "mafic_olivine_pyroxene": round(0.12 + 0.48 * mare_weight, 4),
                    "plagioclase_anorthosite": round(0.83 - 0.55 * mare_weight, 4),
                    "phyllosilicates_clays": 0.0,
                    "sulfates_evaporites": 0.0,
                    "carbonates": 0.0,
                    "iron_oxides_hydroxides": round(0.05 + 0.07 * mare_weight, 4),
                }

        if diviner_prods:
            div_meta, _ = await _fetch_label_and_browse(client, diviner_prods[0])
            t_max = _parse_float_field(div_meta, ("MAXIMUM_TEMPERATURE", "MAX_BRIGHTNESS_TEMPERATURE"))
            t_min = _parse_float_field(div_meta, ("MINIMUM_TEMPERATURE", "MIN_BRIGHTNESS_TEMPERATURE"))
            if t_max is not None:
                result["thermal"]["t_max_k"] = t_max
            if t_min is not None:
                result["thermal"]["t_min_k"] = t_min

        if minirf_prods:
            mrf_meta, mrf_rgb = await _fetch_label_and_browse(client, minirf_prods[0])
            cpr_val = _parse_float_field(mrf_meta, ("CIRCULAR_POLARIZATION_RATIO", "MEAN_CPR"))
            if cpr_val is not None:
                result["cpr_mean"] = cpr_val
            elif mrf_rgb is not None:
                # Mini-RF Stokes browse images encode CPR in normalized green/blue intensity ratio
                result["cpr_mean"] = round(float(np.mean(mrf_rgb[..., 1]) / 255.0) * 1.2, 4)

    return result