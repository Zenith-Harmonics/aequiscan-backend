"""
Multi-Sector Planetary Analog Comparison Router (6 Scientific Pillars).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.core.schemas import AnalogComparisonReport, AnalogComparisonRequest
from app.sectors.pipeline import PlanetaryAnalogPipeline
from app.services.unified_provider import UnifiedPlanetaryDataProvider

router = APIRouter(prefix="/analogs", tags=["analogs"])

_provider = UnifiedPlanetaryDataProvider()
_pipeline = PlanetaryAnalogPipeline()


@router.post(
    "/compare",
    response_model=AnalogComparisonReport,
    summary="Run 6-pillar cross-planetary analog comparison using live satellite & orbital APIs",
)
async def compare_planetary_analogs(req: AnalogComparisonRequest) -> AnalogComparisonReport:
    earth_bundle, target_bundle = await asyncio.gather(
        _provider.load_bundle_async(req.earth_site),
        _provider.load_bundle_async(req.target_site),
    )
    return _pipeline.compare(
        earth_bundle=earth_bundle,
        target_bundle=target_bundle,
        mission_profile=req.mission_profile,
        custom_weights=req.custom_weights,
    )