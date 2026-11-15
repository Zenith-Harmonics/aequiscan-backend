from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import json
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class PlanetBody(str, Enum):
    EARTH = "earth"
    MARS = "mars"
    MOON = "moon"


class SectorID(IntEnum):
    GEOMORPHOLOGY = 1   # Geomorphology & Topography
    MINERALOGY = 2      # Mineralogy & Geochemistry
    SUBSURFACE = 3      # Subsurface & Geophysics
    THERMAL = 4         # Thermal Environment
    ATMOSPHERIC = 5     # Atmospheric & Aeolian Processes
    ASTROBIOLOGY = 6    # Astrobiology & Habitability


SECTOR_NAMES: Dict[SectorID, str] = {
    SectorID.GEOMORPHOLOGY: "Geomorphology & Topography",
    SectorID.MINERALOGY: "Mineralogy & Geochemistry",
    SectorID.SUBSURFACE: "Subsurface & Geophysics",
    SectorID.THERMAL: "Thermal Environment",
    SectorID.ATMOSPHERIC: "Atmospheric & Aeolian Processes",
    SectorID.ASTROBIOLOGY: "Astrobiology & Habitability",
}


class MissionProfile(str, Enum):
    BALANCED = "balanced"
    TRAVERSABILITY = "traversability"
    ISRU_MINING = "isru_mining"
    HABITAT_SUBSURFACE = "habitat_subsurface"
    ASTROBIOLOGY = "astrobiology"


MISSION_WEIGHTS: Dict[MissionProfile, Dict[SectorID, float]] = {
    MissionProfile.BALANCED: {
        SectorID.GEOMORPHOLOGY: 1 / 6,
        SectorID.MINERALOGY: 1 / 6,
        SectorID.SUBSURFACE: 1 / 6,
        SectorID.THERMAL: 1 / 6,
        SectorID.ATMOSPHERIC: 1 / 6,
        SectorID.ASTROBIOLOGY: 1 / 6,
    },
    MissionProfile.TRAVERSABILITY: {
        SectorID.GEOMORPHOLOGY: 0.40,
        SectorID.MINERALOGY: 0.10,
        SectorID.SUBSURFACE: 0.15,
        SectorID.THERMAL: 0.15,
        SectorID.ATMOSPHERIC: 0.15,
        SectorID.ASTROBIOLOGY: 0.05,
    },
    MissionProfile.ISRU_MINING: {
        SectorID.GEOMORPHOLOGY: 0.15,
        SectorID.MINERALOGY: 0.45,
        SectorID.SUBSURFACE: 0.20,
        SectorID.THERMAL: 0.10,
        SectorID.ATMOSPHERIC: 0.05,
        SectorID.ASTROBIOLOGY: 0.05,
    },
    MissionProfile.HABITAT_SUBSURFACE: {
        SectorID.GEOMORPHOLOGY: 0.15,
        SectorID.MINERALOGY: 0.10,
        SectorID.SUBSURFACE: 0.45,
        SectorID.THERMAL: 0.20,
        SectorID.ATMOSPHERIC: 0.05,
        SectorID.ASTROBIOLOGY: 0.05,
    },
    MissionProfile.ASTROBIOLOGY: {
        SectorID.GEOMORPHOLOGY: 0.10,
        SectorID.MINERALOGY: 0.25,
        SectorID.SUBSURFACE: 0.10,
        SectorID.THERMAL: 0.10,
        SectorID.ATMOSPHERIC: 0.10,
        SectorID.ASTROBIOLOGY: 0.35,
    },
}


class BoundingBox(BaseModel):
    """WGS84 / Planetocentric bounding box in decimal degrees."""

    min_lon: float = Field(..., ge=-180.0, le=360.0)
    min_lat: float = Field(..., ge=-90.0, le=90.0)
    max_lon: float = Field(..., ge=-180.0, le=360.0)
    max_lat: float = Field(..., ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        if self.min_lon >= self.max_lon:
            raise ValueError(f"min_lon ({self.min_lon}) must be strictly less than max_lon ({self.max_lon})")
        if self.min_lat >= self.max_lat:
            raise ValueError(f"min_lat ({self.min_lat}) must be strictly less than max_lat ({self.max_lat})")
        return self

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def center(self) -> Tuple[float, float]:
        return ((self.min_lon + self.max_lon) / 2.0, (self.min_lat + self.max_lat) / 2.0)

    def deterministic_hash(self) -> str:
        raw = f"{self.min_lon:.6f}_{self.min_lat:.6f}_{self.max_lon:.6f}_{self.max_lat:.6f}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class SiteQuery(BaseModel):
    """Input specification for an Earth analog site or a Lunar/Martian target site."""

    site_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=128)
    body: PlanetBody
    bbox: BoundingBox
    start_date: str = Field(default="2023-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(default="2024-12-31", pattern=r"^\d{4}-\d{2}-\d{2}$")

    def cache_key(self, namespace: str) -> str:
        payload = {
            "ns": namespace,
            "body": self.body.value,
            "bbox": self.bbox.to_tuple(),
            "start": self.start_date,
            "end": self.end_date,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return f"{self.body.value}_{namespace}_{digest}"


class MetricComparison(BaseModel):
    """Single physical or spectral metric comparison between Earth and target site."""

    name: str
    earth_value: Optional[Union[float, str]] = None
    target_value: Optional[Union[float, str]] = None
    unit: str = ""
    similarity: float = Field(..., ge=0.0, le=1.0)


class SectorResult(BaseModel):
    """Standardized output contract returned by every scientific sector evaluator."""

    sector: SectorID
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    weighted_contribution: float = Field(default=0.0, ge=0.0, le=1.0)
    metrics: List[MetricComparison] = Field(default_factory=list)
    explanation: str = ""

    @field_validator("score", "confidence", "weighted_contribution")
    @classmethod
    def round_floats(cls, v: float) -> float:
        return round(float(v), 4)


class AnalogComparisonRequest(BaseModel):
    """API request schema for comparing an Earth candidate against a planetary target."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "earth_site": {
                    "site_id": "qaidam_dalangtan",
                    "name": "Qaidam Basin (Dalangtan Playa)",
                    "body": "earth",
                    "bbox": {
                        "min_lon": 91.35,
                        "min_lat": 38.40,
                        "max_lon": 91.45,
                        "max_lat": 38.50,
                    },
                    "start_date": "2024-05-01",
                    "end_date": "2024-09-30",
                },
                "target_site": {
                    "site_id": "gale_crater",
                    "name": "Gale Crater",
                    "body": "mars",
                    "bbox": {
                        "min_lon": 137.35,
                        "min_lat": -4.70,
                        "max_lon": 137.45,
                        "max_lat": -4.60,
                    },
                },
                "mission_profile": "astrobiology",
            }
        }
    }

    earth_site: SiteQuery
    target_site: SiteQuery
    mission_profile: MissionProfile = MissionProfile.BALANCED
    custom_weights: Optional[Dict[SectorID, float]] = None

    @model_validator(mode="after")
    def validate_comparison_inputs(self) -> "AnalogComparisonRequest":
        if self.earth_site.body != PlanetBody.EARTH:
            raise ValueError(
                f"earth_site.body must be 'earth', got '{self.earth_site.body.value}'."
            )
        if self.target_site.body == PlanetBody.EARTH:
            raise ValueError("target_site.body must be 'mars' or 'moon'.")
        if self.custom_weights is not None:
            if any(w < 0.0 for w in self.custom_weights.values()):
                raise ValueError("custom_weights values must be non-negative.")
            if sum(self.custom_weights.values()) <= 0.0:
                raise ValueError("custom_weights must sum to a strictly positive number.")
        return self

class AnalogComparisonReport(BaseModel):
    """API response schema containing full multi-sector breakdown and explainability."""

    earth_site: SiteQuery
    target_site: SiteQuery
    mission_profile: MissionProfile
    overall_similarity_score: float = Field(..., ge=0.0, le=1.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    sector_results: List[SectorResult]
    key_strengths: List[str] = Field(default_factory=list)
    key_divergences: List[str] = Field(default_factory=list)


@dataclass
class SiteDataBundle:
    """
    Internal in-memory container holding co-registered rasters, physical scalars,
    and pre-extracted neural embeddings for a single site.
    """

    site: SiteQuery
    pixel_resolution_m: float = 30.0

    # Sector 1: Geomorphology & Topography (H, W) elevation in meters & RGB optical (H, W, 3) uint8
    dem: Optional[np.ndarray] = None
    rgb_image: Optional[np.ndarray] = None

    # Sector 2: Mineralogy & Geochemistry (10, H, W) Sentinel-2 L2A bands & (H, W) bolometric albedo
    s2_multispectral: Optional[np.ndarray] = None
    albedo: Optional[np.ndarray] = None
    mineral_indices: Dict[str, np.ndarray] = field(default_factory=dict)

    # Sector 3: Subsurface & Geophysics (2, H, W) SAR backscatter [VV_dB, VH_dB] or dual-pol radar
    sar_backscatter: Optional[np.ndarray] = None
    radar_cpr: Optional[np.ndarray] = None  # Circular/Cross-Polarization Ratio

    # Sector 4: Thermal Environment (H, W) surface temperature in Kelvin
    thermal_day_k: Optional[np.ndarray] = None
    thermal_night_k: Optional[np.ndarray] = None

    # Sector 5: Atmospheric & Aeolian Processes (MERRA-2 / MCS / REMS scalars)
    atmospheric_scalars: Dict[str, float] = field(default_factory=dict)

    # Sector 6: Astrobiology & Habitability (SMAP / Extremophile envelope scalars)
    astrobiology_scalars: Dict[str, float] = field(default_factory=dict)

    # Pre-computed AI embeddings: keys e.g. 'clip_768', 's1_1024', 's2_1024', 'earth_aligned_768'
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)