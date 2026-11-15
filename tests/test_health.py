"""
Smoke tests for health and targets endpoints (no NASA credentials needed).
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


@pytest.mark.asyncio
async def test_list_targets(client):
    resp = await client.get("/api/v1/targets/")
    assert resp.status_code == 200
    data = resp.json()
    assert "earth" in data
    assert "moon" in data
    assert "mars" in data


@pytest.mark.asyncio
async def test_artemis_regions(client):
    resp = await client.get("/api/v1/targets/moon/artemis-iii")
    assert resp.status_code == 200
    regions = resp.json()
    assert len(regions) == 9
    names = {r["name"] for r in regions}
    assert "Haworth" in names
    assert "Slater Plain" in names


@pytest.mark.asyncio
async def test_mineral_classes(client):
    resp = await client.get("/api/v1/mineralogy/mineral-classes")
    assert resp.status_code == 200
    classes = resp.json()
    assert len(classes) == 6
    assert "mafic_olivine_pyroxene" in classes


@pytest.mark.asyncio
async def test_unknown_body_404(client):
    resp = await client.get("/api/v1/targets/pluto")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_mars_targets(client):
    resp = await client.get("/api/v1/targets/mars")
    assert resp.status_code == 200
    data = resp.json()
    assert "gale_crater" in data
    assert "jezero_mons" in data
