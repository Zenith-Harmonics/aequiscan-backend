"""
Comprehensive unit tests for the 6 Scientific Pillars and PlanetaryAnalogPipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.schemas import (
    MISSION_WEIGHTS,
    BoundingBox,
    MissionProfile,
    PlanetBody,
    SectorID,
    SiteDataBundle,
    SiteQuery,
)
from app.sectors.astrobiology import AstrobiologyEvaluator
from app.sectors.atmospheric import AtmosphericEvaluator
from app.sectors.bundle_builder import SiteBundleBuilder
from app.sectors.geomorphology import GeomorphologyEvaluator
from app.sectors.mineralogy import MineralogyEvaluator
from app.sectors.pipeline import PlanetaryAnalogPipeline
from app.sectors.subsurface import SubsurfaceEvaluator
from app.sectors.thermal import ThermalEvaluator


@pytest.fixture
def qaidam_bundle() -> SiteDataBundle:
    return SiteBundleBuilder.create_reference_bundle("qaidam_dalangtan")


@pytest.fixture
def gale_bundle() -> SiteDataBundle:
    return SiteBundleBuilder.create_reference_bundle("gale_crater")


@pytest.fixture
def moon_bundle() -> SiteDataBundle:
    return SiteBundleBuilder.create_reference_bundle("mons_mouton_plateau")


# ===========================================================================
# 1. Sector 1: Geomorphology & Topography Tests
# ===========================================================================


def test_geomorphology_slope_and_roughness_math():
    """Verify continuous 2D slope angle calculation and RMS roughness."""
    dem = np.zeros((10, 10), dtype=np.float32)
    # 45 degree uniform slope in x-direction with pixel_res = 10m -> dz = 10m per pixel
    for col in range(10):
        dem[:, col] = col * 10.0

    slopes = GeomorphologyEvaluator.compute_slope_map(dem, pixel_res_m=10.0)
    assert slopes.shape == (10, 10)
    # Interior pixels should be 45 degrees
    assert pytest.approx(45.0, abs=1.0) == slopes[5, 5]

    roughness = GeomorphologyEvaluator.compute_rms_roughness(dem)
    assert roughness > 0.0


def test_geomorphology_evaluator(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle):
    """Test full evaluation of Geomorphology & Topography pillar."""
    evaluator = GeomorphologyEvaluator()
    result = evaluator.evaluate(qaidam_bundle, gale_bundle)

    assert result.sector == SectorID.GEOMORPHOLOGY
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.metrics) >= 3
    metric_names = [m.name for m in result.metrics]
    assert "Slope Angle Distribution" in metric_names
    assert "Surface RMS Roughness" in metric_names
    assert len(result.explanation) > 0


# ===========================================================================
# 2. Sector 2: Mineralogy & Geochemistry Tests
# ===========================================================================


def test_mineralogy_evaluator(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle, moon_bundle: SiteDataBundle):
    """Test full evaluation of Mineralogy & Geochemistry pillar."""
    evaluator = MineralogyEvaluator()

    # Earth (Qaidam evaporite) vs Mars (Gale clay/sulfate)
    mars_res = evaluator.evaluate(qaidam_bundle, gale_bundle)
    assert mars_res.sector == SectorID.MINERALOGY
    assert 0.0 <= mars_res.score <= 1.0
    assert len(mars_res.metrics) >= 2

    # Earth (Qaidam) vs Moon (Mons Mouton anorthosite)
    moon_res = evaluator.evaluate(qaidam_bundle, moon_bundle)
    assert moon_res.sector == SectorID.MINERALOGY
    # Anorthosite on Moon vs Evaporite on Earth should score lower in mineral parity
    assert moon_res.score < mars_res.score


# ===========================================================================
# 3. Sector 3: Subsurface & Geophysics Tests
# ===========================================================================


def test_subsurface_cpr_math():
    """Verify linear Cross-Polarization Ratio (CPR) extraction from SAR backscatter."""
    # VV = -10 dB, VH = -20 dB -> CPR = 10^((-20 - (-10))/10) = 10^(-1) = 0.1
    sar = np.array([
        np.full((5, 5), -10.0, dtype=np.float32),
        np.full((5, 5), -20.0, dtype=np.float32),
    ])
    cpr = SubsurfaceEvaluator.compute_cpr_from_sar(sar)
    assert cpr.shape == (5, 5)
    np.testing.assert_allclose(cpr, 0.1, atol=1e-4)


def test_subsurface_evaluator(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle):
    """Test full evaluation of Subsurface & Geophysics pillar."""
    evaluator = SubsurfaceEvaluator()
    result = evaluator.evaluate(qaidam_bundle, gale_bundle)

    assert result.sector == SectorID.SUBSURFACE
    assert 0.0 <= result.score <= 1.0
    metric_names = [m.name for m in result.metrics]
    assert "Cross-Polarization Ratio (CPR)" in metric_names
    assert "Radar Backscatter Intensity" in metric_names


# ===========================================================================
# 4. Sector 4: Thermal Environment Tests
# ===========================================================================


def test_thermal_evaluator(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle, moon_bundle: SiteDataBundle):
    """Test full evaluation of Thermal Environment pillar."""
    evaluator = ThermalEvaluator()

    res_mars = evaluator.evaluate(qaidam_bundle, gale_bundle)
    assert res_mars.sector == SectorID.THERMAL
    assert 0.0 <= res_mars.score <= 1.0
    metric_names = [m.name for m in res_mars.metrics]
    assert "Diurnal Temperature Range" in metric_names
    assert "Extreme Minimum Temperature" in metric_names

    # Lunar cryogenic environment should show thermal divergence compared to terrestrial desert
    res_moon = evaluator.evaluate(qaidam_bundle, moon_bundle)
    assert res_moon.score < res_mars.score


# ===========================================================================
# 5. Sector 5: Atmospheric & Aeolian Processes Tests
# ===========================================================================


def test_atmospheric_evaluator(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle, moon_bundle: SiteDataBundle):
    """Test full evaluation of Atmospheric & Aeolian Processes pillar."""
    evaluator = AtmosphericEvaluator()

    # Earth vs Mars: wind and hyper-aridity match, pressure differs by 2 orders of magnitude
    res_mars = evaluator.evaluate(qaidam_bundle, gale_bundle)
    assert res_mars.sector == SectorID.ATMOSPHERIC
    assert 0.0 <= res_mars.score <= 1.0
    metric_names = [m.name for m in res_mars.metrics]
    assert "Aridity & Desiccation Index" in metric_names
    assert "Atmospheric Pressure Regime" in metric_names

    # Earth vs Moon: Moon has no atmosphere at all (vacuum, 0 wind), so score should be lower
    res_moon = evaluator.evaluate(qaidam_bundle, moon_bundle)
    assert res_moon.score < res_mars.score


# ===========================================================================
# 6. Sector 6: Astrobiology & Habitability Tests
# ===========================================================================


def test_astrobiology_evaluator(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle, moon_bundle: SiteDataBundle):
    """Test full evaluation of Astrobiology & Habitability pillar."""
    evaluator = AstrobiologyEvaluator()

    # Earth vs Mars: High biosignature preservation in clay/evaporites and perchlorates
    res_mars = evaluator.evaluate(qaidam_bundle, gale_bundle)
    assert res_mars.sector == SectorID.ASTROBIOLOGY
    assert res_mars.score >= 0.70
    metric_names = [m.name for m in res_mars.metrics]
    assert "Water Activity (a_w) Envelope" in metric_names
    assert "Biosignature Preservation Potential" in metric_names

    # Earth vs Moon: Moon has 0 water activity and 0 organic preservation
    res_moon = evaluator.evaluate(qaidam_bundle, moon_bundle)
    assert res_moon.score < res_mars.score


# ===========================================================================
# 7. SiteBundleBuilder Tests
# ===========================================================================


def test_bundle_builder_custom():
    """Verify custom bundle creation for arbitrary coordinate query."""
    custom_query = SiteQuery(
        site_id="custom_antarctica",
        name="McMurdo Dry Valleys",
        body=PlanetBody.EARTH,
        bbox=BoundingBox(min_lon=160.0, min_lat=-78.0, max_lon=161.0, max_lat=-77.0),
    )
    bundle = SiteBundleBuilder.create_custom_bundle(custom_query)
    assert bundle.site.name == "McMurdo Dry Valleys"
    assert bundle.dem is not None
    assert bundle.sar_backscatter is not None
    assert bundle.thermal_day_k is not None


# ===========================================================================
# 8. PlanetaryAnalogPipeline Multi-Sector Integration Tests
# ===========================================================================


def test_pipeline_balanced_comparison(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle):
    """Test multi-sector pipeline execution under BALANCED mission profile."""
    pipeline = PlanetaryAnalogPipeline()
    report = pipeline.compare(
        earth_bundle=qaidam_bundle,
        target_bundle=gale_bundle,
        mission_profile=MissionProfile.BALANCED,
    )

    assert report.earth_site.site_id == qaidam_bundle.site.site_id
    assert report.target_site.site_id == gale_bundle.site.site_id
    assert report.mission_profile == MissionProfile.BALANCED
    assert 0.0 <= report.overall_similarity_score <= 1.0
    assert 0.0 <= report.overall_confidence <= 1.0

    # Must contain all 6 scientific sectors
    assert len(report.sector_results) == 6
    sector_ids = {r.sector for r in report.sector_results}
    assert sector_ids == set(SectorID)

    # Weighted contributions must sum to overall score
    total_contrib = sum(r.weighted_contribution for r in report.sector_results)
    assert total_contrib == pytest.approx(report.overall_similarity_score, abs=1e-3)

    # Explainability fields
    assert len(report.key_strengths) > 0
    assert len(report.key_divergences) > 0


def test_pipeline_all_mission_profiles(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle):
    """Verify pipeline executes cleanly across all 5 NASA mission profiles."""
    pipeline = PlanetaryAnalogPipeline()

    scores = {}
    for profile in MissionProfile:
        report = pipeline.compare(
            earth_bundle=qaidam_bundle,
            target_bundle=gale_bundle,
            mission_profile=profile,
        )
        assert 0.0 <= report.overall_similarity_score <= 1.0
        scores[profile] = report.overall_similarity_score

    # Profiles emphasize different sectors; scores should reflect mission priorities
    assert len(scores) == 5


def test_pipeline_custom_weights(qaidam_bundle: SiteDataBundle, gale_bundle: SiteDataBundle):
    """Verify custom sector weights properly dictate overall score."""
    pipeline = PlanetaryAnalogPipeline()

    # 100% weight on Mineralogy
    report_min = pipeline.compare(
        earth_bundle=qaidam_bundle,
        target_bundle=gale_bundle,
        custom_weights={
            SectorID.GEOMORPHOLOGY: 0.0,
            SectorID.MINERALOGY: 1.0,
            SectorID.SUBSURFACE: 0.0,
            SectorID.THERMAL: 0.0,
            SectorID.ATMOSPHERIC: 0.0,
            SectorID.ASTROBIOLOGY: 0.0,
        },
    )
    min_sector_result = next(r for r in report_min.sector_results if r.sector == SectorID.MINERALOGY)
    assert report_min.overall_similarity_score == pytest.approx(min_sector_result.score, abs=1e-4)


def test_pipeline_missing_data_resilience():
    """Verify pipeline handles incomplete bundles without exceptions."""
    minimal_query = SiteQuery(
        site_id="empty_site",
        name="Empty Site",
        body=PlanetBody.EARTH,
        bbox=BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=1.0, max_lat=1.0),
    )
    empty_bundle = SiteDataBundle(site=minimal_query)
    target_bundle = SiteBundleBuilder.create_reference_bundle("gale_crater")

    pipeline = PlanetaryAnalogPipeline()
    report = pipeline.compare(earth_bundle=empty_bundle, target_bundle=target_bundle)

    assert 0.0 <= report.overall_similarity_score <= 1.0
    assert len(report.sector_results) == 6
