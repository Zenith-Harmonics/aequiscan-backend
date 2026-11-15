"""
Custom exceptions for clean error propagation.
"""
from __future__ import annotations


class AequiscanError(Exception):
    """Base exception for all AequiScan errors."""


class LatitudeBoundsError(AequiscanError):
    """Coordinate outside sensor coverage (e.g. EMIT 54° N–54° S)."""


class NoGranulesError(AequiscanError):
    """CMR returned zero matching granules."""


class InsufficientPixelsError(AequiscanError):
    """Granule has too few valid pixels inside the footprint."""


class PlanetaryDataError(AequiscanError):
    """Failed to retrieve or parse planetary archive data."""
