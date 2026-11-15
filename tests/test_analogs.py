"""
Integration tests for mineralogy comparison endpoint.

These require NASA Earthdata credentials and network access.
Mark with @pytest.mark.integration and skip in CI by default.
"""
from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compare_qaidam_vs_gale(client):
    """Full Earth <-> Mars mineralogy comparison (needs EDL + network)."""
    resp = await client.post(
        "/api/v1/mineralogy/compare",
        json={
            "earth_lat": 38.45,
            "earth_lon": 91.42,
            "earth_radius_m": 2000.0,
            "planetary_target_id": "gale_crater",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["asi"]["asi_score"] <= 1.0
    assert data["planetary_body"] == "mars"
