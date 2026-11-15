"""
Shared pytest fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.cache import DiskCacheManager
from app.core.schemas import (
    BoundingBox,
    PlanetBody,
    SiteDataBundle,
    SiteQuery,
)
from app.main import app


@pytest.fixture
async def client():
    """Async test client that talks directly to the ASGI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def tmp_cache(tmp_path) -> DiskCacheManager:
    """Provide an isolated temporary disk cache for each test."""
    return DiskCacheManager(cache_dir=tmp_path / "test_cache")


@pytest.fixture
def earth_site() -> SiteQuery:
    """Sample terrestrial analog query (Rio Tinto / Atacama style coordinates)."""
    return SiteQuery(
        site_id="earth_atacama_01",
        name="Yungay Core Atacama",
        body=PlanetBody.EARTH,
        bbox=BoundingBox(min_lon=-70.05, min_lat=-24.15, max_lon=-69.95, max_lat=-24.05),
    )


@pytest.fixture
def mars_site() -> SiteQuery:
    """Sample Martian target query (Jezero Crater delta region)."""
    return SiteQuery(
        site_id="mars_jezero_01",
        name="Jezero Western Delta",
        body=PlanetBody.MARS,
        bbox=BoundingBox(min_lon=77.35, min_lat=18.40, max_lon=77.45, max_lat=18.50),
    )


@pytest.fixture
def synthetic_earth_bundle(earth_site: SiteQuery) -> SiteDataBundle:
    """Create a deterministic synthetic Earth SiteDataBundle with all sensor modalities."""
    rng = np.random.default_rng(seed=101)
    return SiteDataBundle(
        site=earth_site,
        pixel_resolution_m=30.0,
        dem=rng.normal(loc=1000.0, scale=15.0, size=(120, 120)).astype(np.float32),
        rgb_image=rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        s2_multispectral=rng.normal(loc=1500.0, scale=400.0, size=(10, 120, 120)).astype(np.float32),
        sar_backscatter=rng.normal(loc=-15.0, scale=3.0, size=(2, 120, 120)).astype(np.float32),
        thermal_day_k=rng.normal(loc=305.0, scale=4.0, size=(120, 120)).astype(np.float32),
        thermal_night_k=rng.normal(loc=272.0, scale=3.0, size=(120, 120)).astype(np.float32),
    )


@pytest.fixture
def synthetic_mars_bundle(mars_site: SiteQuery) -> SiteDataBundle:
    """Create a deterministic synthetic Mars SiteDataBundle."""
    rng = np.random.default_rng(seed=202)
    return SiteDataBundle(
        site=mars_site,
        pixel_resolution_m=20.0,
        dem=rng.normal(loc=-2500.0, scale=16.0, size=(120, 120)).astype(np.float32),
        rgb_image=rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        thermal_day_k=rng.normal(loc=265.0, scale=5.0, size=(120, 120)).astype(np.float32),
        thermal_night_k=rng.normal(loc=185.0, scale=4.0, size=(120, 120)).astype(np.float32),
    )
