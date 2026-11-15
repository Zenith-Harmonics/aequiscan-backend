"""
Preloaded planetary target catalogue.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.constants import ARTEMIS_III_REGIONS, PRELOADED_TARGETS
from app.schemas.common import TargetSite

router = APIRouter(prefix="/targets", tags=["targets"])


@router.get(
    "/",
    response_model=dict[str, dict[str, TargetSite]],
    summary="List all preloaded targets grouped by body",
)
async def list_all_targets():
    """Return every preloaded Earth, Moon, and Mars reference target."""
    return PRELOADED_TARGETS


@router.get(
    "/moon/artemis-iii",
    response_model=list[TargetSite],
    summary="The 9 Artemis III South Pole candidate regions",
)
async def artemis_iii_regions():
    moon_targets = PRELOADED_TARGETS["moon"]
    return [
        TargetSite(**moon_targets[rid])
        for rid in ARTEMIS_III_REGIONS
        if rid in moon_targets
    ]


@router.get(
    "/{body}",
    response_model=dict[str, TargetSite],
    summary="List preloaded targets for a single body",
)
async def list_body_targets(body: str):
    body = body.lower()
    if body not in PRELOADED_TARGETS:
        raise HTTPException(404, f"Unknown body '{body}'. Use earth, moon, or mars.")
    return PRELOADED_TARGETS[body]
