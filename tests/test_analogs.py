"""
Unit and integration tests for the Planetary Analog Framework core.

Tests cover: schema validation, cache roundtrips, similarity math primitives,
latent space alignment, and (optionally) full weight download + inference.
"""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.base_sector import BaseSectorEvaluator
from app.core.cache import DiskCacheManager
from app.core.schemas import (
    AnalogComparisonReport,
    AnalogComparisonRequest,
    BoundingBox,
    MissionProfile,
    MISSION_WEIGHTS,
    SectorID,
    SectorResult,
    SiteDataBundle,
    SiteQuery,
)
from app.models.feature_extractor import MultiModalFeatureExtractor
from app.models.space_aligner import LatentSpaceAligner
from app.models.weights_manager import ModelWeightsManager


class _DummySectorEvaluator(BaseSectorEvaluator):
    """Concrete minimal subclass used to test BaseSectorEvaluator math primitives."""

    sector = SectorID.GEOMORPHOLOGY

    def evaluate(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
    ) -> SectorResult:
        sim = self.distribution_similarity_1d(
            earth_bundle.dem,
            target_bundle.dem,
            characteristic_scale=50.0,
        )
        return SectorResult(
            sector=self.sector,
            name=self.name,
            score=sim or 0.0,
            confidence=1.0 if sim is not None else 0.0,
        )


# ---------------------------------------------------------------------------
# 1. Fast Unit Tests (No network or heavy model weights required)
# ---------------------------------------------------------------------------


def test_bounding_box_validation() -> None:
    """Ensure invalid geographic bounds raise a Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        BoundingBox(min_lon=10.0, min_lat=20.0, max_lon=5.0, max_lat=30.0)

    with pytest.raises(ValidationError):
        BoundingBox(min_lon=10.0, min_lat=25.0, max_lon=20.0, max_lat=25.0)


def test_site_query_deterministic_cache_key(earth_site: SiteQuery) -> None:
    """Identical queries must produce identical cache keys across calls."""
    key_1 = earth_site.cache_key("s2_l2a")
    key_2 = earth_site.cache_key("s2_l2a")
    key_other = earth_site.cache_key("s1_grd")

    assert key_1 == key_2
    assert key_1 != key_other
    assert key_1.startswith("earth_s2_l2a_")


def test_disk_cache_atomic_roundtrip(tmp_cache: DiskCacheManager) -> None:
    """Verify compressed .npz arrays and .json metadata roundtrip integrity."""
    arr_dem = np.arange(100, dtype=np.float32).reshape(10, 10)
    tmp_cache.save_arrays("sample_dem", dem=arr_dem)

    assert tmp_cache.has_arrays("sample_dem")
    loaded = tmp_cache.load_arrays("sample_dem")
    assert loaded is not None
    np.testing.assert_allclose(loaded["dem"], arr_dem)

    meta = {"site": "Jezero", "valid_pixels": 100}
    tmp_cache.save_json("sample_meta", meta)
    assert tmp_cache.has_json("sample_meta")
    assert tmp_cache.load_json("sample_meta") == meta


def test_base_sector_similarity_primitives() -> None:
    """Verify cosine, Gaussian scalar, and 1D Wasserstein distribution similarities."""
    evaluator = _DummySectorEvaluator()

    # Cosine similarity bounds
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert evaluator.cosine_similarity(v1, v1) == pytest.approx(1.0)
    assert evaluator.cosine_similarity(v1, v2) == pytest.approx(0.0)

    # Gaussian scalar similarity
    assert evaluator.gaussian_scalar_similarity(15.0, 15.0, tolerance=5.0) == pytest.approx(1.0)
    assert evaluator.gaussian_scalar_similarity(15.0, None, tolerance=5.0) is None

    # 1D Wasserstein similarity with NaN handling
    rng = np.random.default_rng(7)
    raster_a = rng.normal(0.0, 1.0, size=(64, 64))
    raster_b = rng.normal(0.0, 1.0, size=(64, 64))
    raster_a[0, 0] = np.nan

    sim = evaluator.distribution_similarity_1d(raster_a, raster_b, characteristic_scale=1.0)
    assert sim is not None
    assert 0.90 < sim <= 1.0


def test_latent_space_aligner_fit_project_and_save(tmp_path) -> None:
    """Verify 2048D [S1 || S2] -> 768D Ridge projection, L2 normalization, and serialization."""
    rng = np.random.default_rng(42)
    n_samples = 40
    x_fused = rng.normal(size=(n_samples, 2048)).astype(np.float32)
    y_clip = rng.normal(size=(n_samples, 768)).astype(np.float32)

    aligner = LatentSpaceAligner(method="ridge", alpha=5.0)
    aligner.fit(x_fused, y_clip)

    saved_path = aligner.save(tmp_path / "test_aligner.joblib")
    reloaded = LatentSpaceAligner.load(saved_path)

    query_earth = rng.normal(size=(3, 2048)).astype(np.float32)
    projected = reloaded.project_earth_to_target_space(query_earth)

    assert projected.shape == (3, 768)
    norms = np.linalg.norm(projected, axis=-1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    with pytest.raises(ValueError):
        reloaded.project_earth_to_target_space(rng.normal(size=(1, 1024)).astype(np.float32))


def test_mission_weights_and_report_schemas(earth_site: SiteQuery, mars_site: SiteQuery) -> None:
    """Verify mission profile weight normalization and AnalogComparisonReport serialization."""
    # 1. Check all standard mission profiles sum to 1.0 across the 6 sectors
    for profile, weights in MISSION_WEIGHTS.items():
        assert len(weights) == 6, f"{profile} does not define all 6 sectors"
        total_weight = sum(weights.values())
        assert total_weight == pytest.approx(1.0, abs=1e-4), f"{profile} weights do not sum to 1.0"

    # 2. Check AnalogComparisonRequest construction
    req = AnalogComparisonRequest(
        earth_site=earth_site,
        target_site=mars_site,
        mission_profile=MissionProfile.BALANCED,
    )
    assert req.earth_site.name == earth_site.name

    # 3. Check AnalogComparisonReport schema with all 6 sectors
    results = [
        SectorResult(
            sector=s_id,
            name=f"Sector {s_id}",
            score=0.85,
            confidence=0.90,
            weighted_contribution=round(0.85 * (1.0 / 6.0), 4),
            explanation=f"Strong analog match for sector {s_id}.",
        )
        for s_id in SectorID
    ]

    report = AnalogComparisonReport(
        earth_site=earth_site,
        target_site=mars_site,
        mission_profile=MissionProfile.BALANCED,
        overall_similarity_score=0.85,
        overall_confidence=0.90,
        sector_results=results,
        key_strengths=["High mineralogical & geomorphic alignment"],
        key_divergences=["Atmospheric pressure divergence (1 bar vs 6 mbar)"],
    )

    dumped = report.model_dump()
    assert len(dumped["sector_results"]) == 6
    assert dumped["overall_similarity_score"] == 0.85


# ---------------------------------------------------------------------------
# 2. Integration Test (Downloads Hugging Face weights & runs PyTorch forward pass)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_weights_fetch_and_multimodal_inference(
    synthetic_earth_bundle: SiteDataBundle,
    synthetic_mars_bundle: SiteDataBundle,
) -> None:
    """
    End-to-end smoke test:
      1. Downloads/verifies Sentinel-1, Sentinel-2, and RemoteCLIP weights in app/models/weights.
      2. Extracts 1024D and 768D embeddings on CPU/GPU.
      3. Projects Earth fused embeddings into RemoteCLIP space.
    """
    manager = ModelWeightsManager()
    paths = manager.fetch_all()

    for key in ("sentinel_1", "sentinel_2", "mars_remoteclip"):
        assert manager.is_downloaded(key), f"Model '{key}' failed verification after download."

    extractor = MultiModalFeatureExtractor(model_paths=paths)

    # Fit a dummy aligner for shape/pipeline verification
    rng = np.random.default_rng(99)
    dummy_aligner = LatentSpaceAligner(method="ridge").fit(
        rng.normal(size=(20, 2048)).astype(np.float32),
        rng.normal(size=(20, 768)).astype(np.float32),
    )

    extractor.populate_bundle_embeddings(synthetic_earth_bundle, aligner=dummy_aligner)
    extractor.populate_bundle_embeddings(synthetic_mars_bundle)

    assert synthetic_earth_bundle.embeddings["s1_1024"].shape == (1024,)
    assert synthetic_earth_bundle.embeddings["s2_1024"].shape == (1024,)
    assert synthetic_earth_bundle.embeddings["clip_768"].shape == (768,)
    assert synthetic_earth_bundle.embeddings["earth_aligned_768"].shape == (768,)
    assert synthetic_mars_bundle.embeddings["clip_768"].shape == (768,)
