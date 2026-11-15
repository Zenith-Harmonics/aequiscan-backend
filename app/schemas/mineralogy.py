"""
Request / response models for the mineralogy analytical axis.
"""
from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator

from app.core.constants import MINERAL_CLASSES


class MineralProfile(BaseModel):
    """Harmonised 6-class mineral fractional profile."""
    source: str = Field(..., description="Sensor + archive provenance string")
    pds_sample_id: str | None = Field(
        None, description="PDS product ID (Mars / Moon only)"
    )
    n_pixels: int | None = Field(
        None, ge=0, description="Total footprint pixels examined"
    )
    n_valid_mineral_pixels: int | None = Field(
        None, ge=0, description="Cloud-free classified pixels"
    )
    mean_band_depth: float | None = Field(
        None, description="Mean spectral band depth of detections"
    )
    top_raw_minerals: list[tuple[str, int]] | None = Field(
        None, description="Top 5 USGS Spectral Library 7 detections [(name, count)]"
    )
    harmonized_profile: dict[str, float] = Field(
        ...,
        description="Fractional abundance per harmonised mineral class",
        json_schema_extra={
            "example": {cls: 0.0 for cls in MINERAL_CLASSES},
        },
    )


class ASIWeights(BaseModel):
    """User-adjustable per-class weights for the ASI formula."""
    mafic_olivine_pyroxene: float = Field(default=1.0, ge=0)
    plagioclase_anorthosite: float = Field(default=1.0, ge=0)
    phyllosilicates_clays: float = Field(default=1.0, ge=0)
    sulfates_evaporites: float = Field(default=1.0, ge=0)
    carbonates: float = Field(default=1.0, ge=0)
    iron_oxides_hydroxides: float = Field(default=1.0, ge=0)

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class ASIAxisDetail(BaseModel):
    mineral_class: str
    earth_fraction: float
    planetary_fraction: float
    abs_delta: float
    weight: float


class ASIResult(BaseModel):
    """Analog Similarity Index result for one Earth-Planetary pair."""
    asi_score: float = Field(
        ..., ge=0, le=1, description="Overall ASI (1 = identical)"
    )
    axis_details: list[ASIAxisDetail]
    earth_profile: MineralProfile
    planetary_profile: MineralProfile


class MineralogyCompareRequest(BaseModel):
    """POST body for /mineralogy/compare."""
    earth_lat: float = Field(..., ge=-54, le=54, description="Earth latitude (EMIT: 54 N-54 S)")
    earth_lon: float = Field(..., ge=-180, le=360)
    earth_radius_m: float = Field(default=2000.0, ge=10, le=6_000_000)

    # 1. Option A: Preloaded planetary target ID (e.g. 'gale_crater', 'mons_mouton_plateau')
    planetary_target_id: str | None = Field(
        default=None,
        description="Key into PRELOADED_TARGETS[moon|mars], e.g. 'gale_crater'",
    )

    # 2. Option B: Custom arbitrary coordinates on Moon or Mars
    planetary_body: str | None = Field(
        default=None,
        pattern="^(moon|mars)$",
        description="Planetary target body ('moon' or 'mars') when using custom coordinates",
    )
    planetary_lat: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        description="Custom planetary latitude (deg N)",
    )
    planetary_lon: float | None = Field(
        default=None,
        ge=-180,
        le=360,
        description="Custom planetary longitude (East-positive deg E, 0-360 or -180 to 180)",
    )

    weights: ASIWeights | None = None
    simulate: bool = Field(
        default=False,
        description="If True, returns precomputed EMIT profile (useful for offline testing or without NASA credentials)",
    )

    @model_validator(mode="after")
    def validate_target_specification(self) -> Self:
        has_id = bool(self.planetary_target_id)
        has_coords = (
            self.planetary_body is not None
            and self.planetary_lat is not None
            and self.planetary_lon is not None
        )
        if not (has_id or has_coords):
            raise ValueError(
                "Must provide either 'planetary_target_id' or all of "
                "('planetary_body', 'planetary_lat', 'planetary_lon')."
            )
        return self


class MineralogyCompareResponse(BaseModel):
    """Full comparison result returned by /mineralogy/compare."""
    earth_site_name: str
    planetary_site_name: str
    planetary_body: str
    asi: ASIResult
