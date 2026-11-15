"""
Core domain logic, models, schemas, and configurations for AequiScan.
"""
from app.core.base_sector import BaseSectorEvaluator
from app.core.cache import DiskCacheManager
from app.core.config import Settings, get_settings
from app.core.constants import (
    ARTEMIS_III_REGIONS,
    EMIT_KEYWORD_MAP,
    MINERAL_CLASSES,
    PRELOADED_TARGETS,
)
from app.core.exceptions import (
    AequiscanError,
    InsufficientPixelsError,
    LatitudeBoundsError,
    NoGranulesError,
)
from app.core.schemas import (
    MISSION_WEIGHTS,
    SECTOR_NAMES,
    AnalogComparisonReport,
    AnalogComparisonRequest,
    BoundingBox,
    MetricComparison,
    MissionProfile,
    PlanetBody,
    SectorID,
    SectorResult,
    SiteDataBundle,
    SiteQuery,
)

__all__ = [
    "BaseSectorEvaluator",
    "DiskCacheManager",
    "Settings",
    "get_settings",
    "ARTEMIS_III_REGIONS",
    "EMIT_KEYWORD_MAP",
    "MINERAL_CLASSES",
    "PRELOADED_TARGETS",
    "AequiscanError",
    "InsufficientPixelsError",
    "LatitudeBoundsError",
    "NoGranulesError",
    "MISSION_WEIGHTS",
    "SECTOR_NAMES",
    "AnalogComparisonReport",
    "AnalogComparisonRequest",
    "BoundingBox",
    "MetricComparison",
    "MissionProfile",
    "PlanetBody",
    "SectorID",
    "SectorResult",
    "SiteDataBundle",
    "SiteQuery",
]
