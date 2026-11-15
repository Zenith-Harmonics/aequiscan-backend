"""
Shared Pydantic models used across multiple routers.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Coordinates(BaseModel):
    """Geodetic coordinates with an exploration footprint radius."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude (deg N)")
    lon: float = Field(..., ge=-180, le=360, description="Longitude (deg E)")
    radius_m: float = Field(
        default=2000.0,
        ge=10,
        le=6_000_000,
        description="Footprint radius in metres (CMR constraint: 10-6 000 000 m)",
    )


class TargetSite(Coordinates):
    """A named site on Earth, Moon, or Mars."""
    name: str
    body: str = Field(
        ...,
        pattern="^(earth|moon|mars)$",
        description="Celestial body: earth | moon | mars",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None
