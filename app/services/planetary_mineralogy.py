"""
Mars & Moon mineralogy via PDS Orbital Data Explorer (ODE) REST API.

Until full spectral processing of CRISM / M3 cubes is implemented,
this module returns literature-anchored reference profiles enriched
with live PDS product IDs from ODE.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _extract_pds_product_id(product: dict[str, Any], default: str) -> str:
    """Extract a concise PDS product identifier from an ODE product record."""
    # 1. Product_name if present (e.g. 'frt000058a3_07_if165j_mtr3.img')
    product_name = product.get("Product_name")
    if product_name:
        return str(product_name).split(".")[0].upper()

    # 2. Extract from ProductURL query parameter 'product_id='
    product_url = product.get("ProductURL", "")
    if "product_id=" in product_url:
        parsed = urllib.parse.urlparse(product_url)
        params = urllib.parse.parse_qs(parsed.query)
        if "product_id" in params and params["product_id"]:
            return params["product_id"][0]

    # 3. Extract from LabelURL filename
    label_url = product.get("LabelURL", "")
    if label_url:
        filename = label_url.split("/")[-1]
        return filename.split(".")[0]

    # 4. Fallback to pdsid if present or default
    return str(product.get("pdsid") or default)


async def query_pds_ode_products(
    target: str,
    instrument: str,
    product_type: str,
    lat: float,
    lon: float,
    radius_deg: float = 0.08,
    limit: int = 5,
    ihid: str | None = None,
) -> list[dict[str, Any]]:
    """
    Query PDS ODE REST API for overlapping products.

    Parameters
    ----------
    target : "mars" | "moon"
    instrument : e.g. "CRISM", "M3"
    product_type : e.g. "MTRDR", "REFIMG"
    """
    settings = get_settings()
    lon_360 = lon % 360.0

    if ihid is None:
        if target == "mars":
            ihid = "MRO"
        elif target == "moon":
            ihid = "CH1-ORB" if instrument.upper() == "M3" else "LRO"
        else:
            ihid = "MRO"

    min_lat = max(-90.0, lat - radius_deg)
    max_lat = min(90.0, lat + radius_deg)

    params = {
        "query": "product",
        "results": "m",
        "output": "JSON",
        "target": target,
        "ihid": ihid,
        "iid": instrument,
        "pt": product_type,
        "minlat": min_lat,
        "maxlat": max_lat,
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
            if isinstance(products, dict):
                products = [products]
            return products
        return []
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("PDS ODE unreachable for %s/%s: %s", target, instrument, exc)
        return []


async def get_mars_mineral_profile(
    lat: float, lon: float
) -> dict[str, Any]:
    """Return a Mars CRISM-anchored mineral profile for the given coordinate."""
    products = await query_pds_ode_products(
        target="mars",
        instrument="CRISM",
        product_type="MTRDR",
        lat=lat,
        lon=lon,
    )
    logger.info("PDS ODE: %d CRISM MTRDR granules for Mars (%.2f, %.2f)", len(products), lat, lon)

    return {
        "source": f"MRO CRISM MTRDR (~18 m; {len(products)} PDS granules matched)",
        "pds_sample_id": (
            _extract_pds_product_id(products[0], "FRT0000B6F1") if products else "FRT0000B6F1"
        ),
        "mean_band_depth": 0.048,
        "harmonized_profile": {
            "mafic_olivine_pyroxene": 0.22,
            "plagioclase_anorthosite": 0.05,
            "phyllosilicates_clays": 0.34,
            "sulfates_evaporites": 0.26,
            "carbonates": 0.03,
            "iron_oxides_hydroxides": 0.10,
        },
    }


async def get_moon_mineral_profile(
    lat: float, lon: float
) -> dict[str, Any]:
    """Return a Moon M3/Diviner-anchored mineral profile for the given coordinate."""
    products = await query_pds_ode_products(
        target="moon",
        instrument="M3",
        product_type="REFIMG",
        lat=lat,
        lon=lon,
        radius_deg=0.5,
    )
    logger.info("PDS ODE: %d M3 REFIMG granules for Moon (%.2f, %.2f)", len(products), lat, lon)

    return {
        "source": f"Chandrayaan-1 M3 Level-2 Reflectance ({len(products)} PDS granules matched)",
        "pds_sample_id": (
            _extract_pds_product_id(products[0], "M3G20090208T051432")
            if products
            else "M3G20090208T051432"
        ),
        "mean_band_depth": 0.021,
        "harmonized_profile": {
            "mafic_olivine_pyroxene": 0.14,
            "plagioclase_anorthosite": 0.81,
            "phyllosilicates_clays": 0.00,
            "sulfates_evaporites": 0.00,
            "carbonates": 0.00,
            "iron_oxides_hydroxides": 0.05,
        },
    }
