"""
Mineralogy analytical axis - surface composition comparison.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.core.constants import MINERAL_CLASSES, PRELOADED_TARGETS
from app.core.exceptions import (
    AequiscanError,
    InsufficientPixelsError,
    LatitudeBoundsError,
    NoGranulesError,
)
from app.schemas.mineralogy import (
    ASIResult,
    MineralogyCompareRequest,
    MineralogyCompareResponse,
    MineralProfile,
)
from app.services.earth_mineralogy import (
    extract_emit_mineralogy,
    get_simulated_earth_profile,
)
from app.services.planetary_mineralogy import (
    get_mars_mineral_profile,
    get_moon_mineral_profile,
)
from app.services.similarity import compute_mineral_asi

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mineralogy", tags=["mineralogy"])


def _find_planetary_target(target_id: str) -> dict:
    """Look up a target by ID across moon and mars catalogues."""
    for body in ("mars", "moon"):
        targets = PRELOADED_TARGETS.get(body, {})
        if target_id in targets:
            return targets[target_id]
    raise HTTPException(
        404,
        f"Planetary target '{target_id}' not found. "
        f"Available: {list(PRELOADED_TARGETS['mars']) + list(PRELOADED_TARGETS['moon'])}",
    )


@router.get(
    "/mineral-classes",
    response_model=list[str],
    summary="List the 6 harmonised mineral classes",
)
async def list_mineral_classes():
    return MINERAL_CLASSES


@router.post(
    "/compare",
    response_model=MineralogyCompareResponse,
    summary="Compare Earth site mineralogy against a planetary target",
    responses={
        400: {"description": "Coordinate outside sensor coverage"},
        404: {"description": "No granules / target not found"},
        422: {"description": "Validation error"},
        502: {"description": "Upstream NASA/PDS service error"},
    },
)
async def compare_mineralogy(req: MineralogyCompareRequest):
    """
    Run a full Earth <-> planetary mineralogy comparison:

    1. Extract Earth EMITL2BMIN profile via earthaccess
    2. Fetch planetary profile from PDS ODE
    3. Compute weighted ASI score
    """
    # -- Resolve planetary target (preloaded ID or custom coordinates) --
    if req.planetary_target_id:
        target = _find_planetary_target(req.planetary_target_id)
        body = target["body"]
        target_lat = target["lat"]
        target_lon = target["lon"]
        target_name = target["name"]
    else:
        body = req.planetary_body.lower()  # type: ignore[union-attr]
        target_lat = req.planetary_lat     # type: ignore[assignment]
        target_lon = req.planetary_lon     # type: ignore[assignment]
        target_name = f"Custom {body.capitalize()} ({target_lat:.4f}°N, {target_lon:.4f}°E)"

    # -- Earth extraction (CPU-bound -> run in threadpool) ---------
    try:
        if req.simulate:
            earth_data = get_simulated_earth_profile(
                lat=req.earth_lat,
                lon=req.earth_lon,
                radius_m=req.earth_radius_m,
            )
        else:
            earth_data = await asyncio.to_thread(
                extract_emit_mineralogy,
                lat=req.earth_lat,
                lon=req.earth_lon,
                radius_m=req.earth_radius_m,
            )
    except LatitudeBoundsError as exc:
        raise HTTPException(400, str(exc))
    except NoGranulesError as exc:
        raise HTTPException(404, str(exc))
    except InsufficientPixelsError as exc:
        raise HTTPException(404, str(exc))
    except AequiscanError as exc:
        raise HTTPException(502, str(exc))

    # -- Planetary profile ----------------------------------------
    if body == "mars":
        planet_data = await get_mars_mineral_profile(target_lat, target_lon)
    elif body == "moon":
        planet_data = await get_moon_mineral_profile(target_lat, target_lon)
    else:
        raise HTTPException(400, f"Unsupported body: {body}")

    # -- ASI computation ------------------------------------------
    weights = req.weights.as_dict() if req.weights else None
    asi_score, axis_details = compute_mineral_asi(
        earth_data["harmonized_profile"],
        planet_data["harmonized_profile"],
        weights=weights,
    )

    earth_profile = MineralProfile(
        source=earth_data["source"],
        n_pixels=earth_data.get("n_pixels"),
        n_valid_mineral_pixels=earth_data.get("n_valid_mineral_pixels"),
        mean_band_depth=earth_data.get("mean_band_depth"),
        top_raw_minerals=earth_data.get("top_raw_minerals"),
        harmonized_profile=earth_data["harmonized_profile"],
    )
    planetary_profile = MineralProfile(
        source=planet_data["source"],
        pds_sample_id=planet_data.get("pds_sample_id"),
        mean_band_depth=planet_data.get("mean_band_depth"),
        harmonized_profile=planet_data["harmonized_profile"],
    )

    return MineralogyCompareResponse(
        earth_site_name=f"Custom ({req.earth_lat:.4f}°N, {req.earth_lon:.4f}°E)",
        planetary_site_name=target_name,
        planetary_body=body,
        asi=ASIResult(
            asi_score=asi_score,
            axis_details=axis_details,
            earth_profile=earth_profile,
            planetary_profile=planetary_profile,
        ),
    )
