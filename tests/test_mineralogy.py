"""
Unit tests for the /mineralogy API router with mocked external services.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_compare_latitude_out_of_bounds(client):
    """Latitudes exceeding 54° should be rejected per ISS EMIT bounds."""
    resp = await client.post(
        "/api/v1/mineralogy/compare",
        json={
            "earth_lat": 65.0,  # exceeds 54°
            "earth_lon": 10.0,
            "earth_radius_m": 2000.0,
            "planetary_target_id": "gale_crater",
        },
    )
    assert resp.status_code == 422  # Pydantic schema validation ge=-54, le=54


@pytest.mark.asyncio
async def test_compare_target_not_found(client):
    resp = await client.post(
        "/api/v1/mineralogy/compare",
        json={
            "earth_lat": 38.45,
            "earth_lon": 91.42,
            "earth_radius_m": 2000.0,
            "planetary_target_id": "non_existent_crater",
        },
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_compare_successful_with_mocks(client):
    mock_earth = {
        "source": "NASA EMITL2BMIN V001 (Mocked)",
        "n_pixels": 3480,
        "n_valid_mineral_pixels": 450,
        "mean_band_depth": 0.052,
        "top_raw_minerals": [("Halite", 120), ("Gypsum", 95)],
        "harmonized_profile": {
            "mafic_olivine_pyroxene": 0.10,
            "plagioclase_anorthosite": 0.05,
            "phyllosilicates_clays": 0.20,
            "sulfates_evaporites": 0.50,
            "carbonates": 0.05,
            "iron_oxides_hydroxides": 0.10,
        },
    }

    with (
        patch("app.api.v1.mineralogy.extract_emit_mineralogy", return_value=mock_earth),
        patch("app.services.planetary_mineralogy.query_pds_ode_products", return_value=[{"pdsid": "FRT0000B6F1"}]),
    ):
        resp = await client.post(
            "/api/v1/mineralogy/compare",
            json={
                "earth_lat": 38.45,
                "earth_lon": 91.42,
                "earth_radius_m": 2000.0,
                "planetary_target_id": "gale_crater",
                "weights": {
                    "mafic_olivine_pyroxene": 1.0,
                    "plagioclase_anorthosite": 1.0,
                    "phyllosilicates_clays": 1.0,
                    "sulfates_evaporites": 2.0,
                    "carbonates": 1.0,
                    "iron_oxides_hydroxides": 1.0,
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["planetary_body"] == "mars"
        assert data["planetary_site_name"] == "Gale Crater (Glen Torridon / Clay-Sulfate Unit)"
        assert 0.0 <= data["asi"]["asi_score"] <= 1.0
        assert len(data["asi"]["axis_details"]) == 6
        assert data["asi"]["earth_profile"]["n_valid_mineral_pixels"] == 450


@pytest.mark.asyncio
async def test_compare_with_simulate_flag(client):
    """Test simulated EMIT extraction without mocks or credentials."""
    resp = await client.post(
        "/api/v1/mineralogy/compare",
        json={
            "earth_lat": 38.45,
            "earth_lon": 91.42,
            "earth_radius_m": 2000.0,
            "planetary_target_id": "gale_crater",
            "simulate": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["planetary_body"] == "mars"
    assert 0.0 <= data["asi"]["asi_score"] <= 1.0
    assert len(data["asi"]["axis_details"]) == 6
    assert data["asi"]["earth_profile"]["harmonized_profile"]["sulfates_evaporites"] == 0.50
    assert data["asi"]["planetary_profile"]["harmonized_profile"]["sulfates_evaporites"] == 0.26


@pytest.mark.asyncio
async def test_compare_with_custom_coordinates(client):
    """Test using custom arbitrary Mars coordinates instead of preloaded ID."""
    resp = await client.post(
        "/api/v1/mineralogy/compare",
        json={
            "earth_lat": 38.45,
            "earth_lon": 91.42,
            "earth_radius_m": 2000.0,
            "planetary_body": "mars",
            "planetary_lat": 18.65,
            "planetary_lon": 226.2,  # Olympus Mons area
            "simulate": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["planetary_body"] == "mars"
    assert "Custom Mars" in data["planetary_site_name"]
    assert 0.0 <= data["asi"]["asi_score"] <= 1.0


@pytest.mark.asyncio
async def test_missing_both_id_and_coords_fails_422(client):
    """Omitting both planetary_target_id and custom coordinates must fail validation."""
    resp = await client.post(
        "/api/v1/mineralogy/compare",
        json={
            "earth_lat": 38.45,
            "earth_lon": 91.42,
            "simulate": True,
        },
    )
    assert resp.status_code == 422
    assert "planetary_target_id" in resp.text or "planetary_body" in resp.text
