from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy.stats import wasserstein_distance
from app.core.schemas import SECTOR_NAMES, SectorID, SectorResult, SiteDataBundle


class BaseSectorEvaluator(ABC):
    """
    Abstract base class for all six Planetary Analog Framework sector evaluators.
    Provides standardized mathematical similarity primitives and fallback handling.
    """

    sector: SectorID

    def __init__(self) -> None:
        if not hasattr(self, "sector"):
            raise AttributeError(f"{self.__class__.__name__} must define class attribute 'sector'.")
        self.name: str = SECTOR_NAMES[self.sector]

    @abstractmethod
    def evaluate(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
    ) -> SectorResult:
        """Compute physical and/or latent similarity metrics for this scientific sector."""
        raise NotImplementedError

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute normalized cosine similarity mapped smoothly to [0.0, 1.0].
        """
        a = np.asarray(vec_a, dtype=np.float64).ravel()
        b = np.asarray(vec_b, dtype=np.float64).ravel()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        raw_cos = float(np.dot(a, b) / (norm_a * norm_b))
        # Clamp numerical drift and map [-1, 1] -> [0, 1]
        raw_cos = max(-1.0, min(1.0, raw_cos))
        return 0.5 * (raw_cos + 1.0)

    @staticmethod
    def gaussian_scalar_similarity(
        val_a: Optional[float],
        val_b: Optional[float],
        tolerance: float,
    ) -> Optional[float]:
        """
        Compute physically calibrated similarity between two scalar measurements:
            S = exp(-0.5 * ((val_a - val_b) / tolerance)^2)
        Returns None if either measurement is missing.
        """
        if val_a is None or val_b is None or np.isnan(val_a) or np.isnan(val_b):
            return None
        if tolerance <= 0.0:
            raise ValueError("Tolerance must be strictly positive.")
        z = (float(val_a) - float(val_b)) / float(tolerance)
        return float(np.exp(-0.5 * (z ** 2)))

    @staticmethod
    def distribution_similarity_1d(
        arr_a: Optional[np.ndarray],
        arr_b: Optional[np.ndarray],
        characteristic_scale: float,
        max_samples: int = 10_000,
    ) -> Optional[float]:
        """
        Compare two 2D spatial raster fields using 1st Wasserstein (Earth Mover's) distance
        over valid finite pixels, preserving spatial heterogeneity instead of collapsing to means.
        """
        if arr_a is None or arr_b is None:
            return None

        valid_a = arr_a[np.isfinite(arr_a)].ravel()
        valid_b = arr_b[np.isfinite(arr_b)].ravel()

        if valid_a.size == 0 or valid_b.size == 0:
            return None

        # Deterministic subsampling for fast O(N log N) Wasserstein computation on large rasters
        rng = np.random.default_rng(seed=42)
        if valid_a.size > max_samples:
            valid_a = rng.choice(valid_a, size=max_samples, replace=False)
        if valid_b.size > max_samples:
            valid_b = rng.choice(valid_b, size=max_samples, replace=False)

        w_dist = float(wasserstein_distance(valid_a, valid_b))
        return float(np.exp(-w_dist / float(characteristic_scale)))

    def _empty_result(self, reason: str) -> SectorResult:
        """Return a zero-confidence neutral result when required datasets are unavailable."""
        return SectorResult(
            sector=self.sector,
            name=self.name,
            score=0.0,
            confidence=0.0,
            metrics=[],
            explanation=f"Insufficient data for {self.name}: {reason}",
        )