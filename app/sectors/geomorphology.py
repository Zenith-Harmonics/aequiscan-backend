"""
Sector 1: Geomorphology & Topography Evaluator.

Evaluates terrain geometry, slope distributions, landing safety,
surface roughness, and visual morphological similarity.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.core.base_sector import BaseSectorEvaluator
from app.core.schemas import MetricComparison, SectorID, SectorResult, SiteDataBundle

logger = logging.getLogger(__name__)


class GeomorphologyEvaluator(BaseSectorEvaluator):
    """
    Sector 1 Evaluator: Geomorphology & Topography.

    Analyzes:
      - 2D slope angle maps derived from digital elevation models (DEM)
      - Detrended surface roughness (RMS deviation from local planar trend)
      - Hypsometric relief (elevation standard deviation and dynamic range)
      - Optical visual morphology via 768D RemoteCLIP latent alignment
    """

    sector = SectorID.GEOMORPHOLOGY

    @staticmethod
    def compute_slope_map(dem: np.ndarray, pixel_res_m: float = 30.0) -> np.ndarray:
        """
        Compute continuous slope angle in degrees from elevation grid.

        Slope = arctan(sqrt((dz/dy)^2 + (dz/dx)^2)) * (180 / pi)
        """
        if dem.ndim != 2 or dem.shape[0] < 3 or dem.shape[1] < 3:
            raise ValueError("DEM must be a 2D array of at least 3x3 pixels.")

        res = max(float(pixel_res_m), 1.0)
        gy, gx = np.gradient(dem, res, res)
        grad_mag = np.hypot(gy, gx)
        slope_deg = np.degrees(np.arctan(grad_mag))
        return slope_deg.astype(np.float32)

    @staticmethod
    def compute_rms_roughness(dem: np.ndarray) -> float:
        """
        Compute root-mean-square (RMS) surface roughness (standard deviation of detrended elevation).
        """
        valid = dem[np.isfinite(dem)]
        if valid.size < 4:
            return 0.0
        # Subtract mean elevation to detrend constant offset
        residual = valid - np.nanmean(valid)
        return float(np.sqrt(np.mean(residual**2)))

    def evaluate(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
    ) -> SectorResult:
        metrics: list[MetricComparison] = []
        scores: list[float] = []
        weights: list[float] = []

        # ------------------------------------------------------------------
        # 1. Slope Angle Distribution (DEM)
        # ------------------------------------------------------------------
        if earth_bundle.dem is not None and target_bundle.dem is not None:
            try:
                earth_slopes = self.compute_slope_map(
                    earth_bundle.dem, earth_bundle.pixel_resolution_m
                )
                target_slopes = self.compute_slope_map(
                    target_bundle.dem, target_bundle.pixel_resolution_m
                )

                # Distribution similarity using 1st Wasserstein distance (scale: 5.0 degrees)
                slope_sim = self.distribution_similarity_1d(
                    earth_slopes, target_slopes, characteristic_scale=5.0
                )
                if slope_sim is not None:
                    mean_e_slope = float(np.nanmean(earth_slopes))
                    mean_t_slope = float(np.nanmean(target_slopes))
                    metrics.append(
                        MetricComparison(
                            name="Slope Angle Distribution",
                            earth_value=f"{mean_e_slope:.1f}° (mean)",
                            target_value=f"{mean_t_slope:.1f}° (mean)",
                            unit="degrees",
                            similarity=round(float(slope_sim), 4),
                        )
                    )
                    scores.append(float(slope_sim))
                    weights.append(0.35)
            except Exception as exc:
                logger.warning("Geomorphology slope calculation error: %s", exc)

        # ------------------------------------------------------------------
        # 2. Surface RMS Roughness
        # ------------------------------------------------------------------
        if earth_bundle.dem is not None and target_bundle.dem is not None:
            try:
                e_roughness = self.compute_rms_roughness(earth_bundle.dem)
                t_roughness = self.compute_rms_roughness(target_bundle.dem)
                # Calibrated Gaussian similarity: tolerance = 8.0 meters
                roughness_sim = self.gaussian_scalar_similarity(
                    e_roughness, t_roughness, tolerance=8.0
                )
                if roughness_sim is not None:
                    metrics.append(
                        MetricComparison(
                            name="Surface RMS Roughness",
                            earth_value=round(e_roughness, 2),
                            target_value=round(t_roughness, 2),
                            unit="m",
                            similarity=round(float(roughness_sim), 4),
                        )
                    )
                    scores.append(float(roughness_sim))
                    weights.append(0.25)
            except Exception as exc:
                logger.warning("Geomorphology roughness calculation error: %s", exc)

        # ------------------------------------------------------------------
        # 3. Hypsometric Dynamic Relief
        # ------------------------------------------------------------------
        if earth_bundle.dem is not None and target_bundle.dem is not None:
            try:
                e_relief = float(np.nanmax(earth_bundle.dem) - np.nanmin(earth_bundle.dem))
                t_relief = float(np.nanmax(target_bundle.dem) - np.nanmin(target_bundle.dem))
                relief_sim = self.gaussian_scalar_similarity(
                    e_relief, t_relief, tolerance=max(30.0, 0.35 * max(e_relief, t_relief))
                )
                if relief_sim is not None:
                    metrics.append(
                        MetricComparison(
                            name="Hypsometric Relief Range",
                            earth_value=round(e_relief, 1),
                            target_value=round(t_relief, 1),
                            unit="m",
                            similarity=round(float(relief_sim), 4),
                        )
                    )
                    scores.append(float(relief_sim))
                    weights.append(0.15)
            except Exception as exc:
                logger.warning("Geomorphology relief calculation error: %s", exc)

        # ------------------------------------------------------------------
        # 4. Optical / Visual Morphology Latent Similarity (RemoteCLIP)
        # ------------------------------------------------------------------
        # Earth can use 'earth_aligned_768' or 'clip_768'; Target uses 'clip_768'
        e_clip = earth_bundle.embeddings.get(
            "earth_aligned_768", earth_bundle.embeddings.get("clip_768")
        )
        t_clip = target_bundle.embeddings.get("clip_768")

        if e_clip is not None and t_clip is not None:
            latent_sim = self.cosine_similarity(e_clip, t_clip)
            metrics.append(
                MetricComparison(
                    name="Visual Morphological Latent Match",
                    earth_value="RemoteCLIP-768D (Aligned)",
                    target_value="RemoteCLIP-768D",
                    unit="cosine",
                    similarity=round(float(latent_sim), 4),
                )
            )
            scores.append(float(latent_sim))
            weights.append(0.25)

        # ------------------------------------------------------------------
        # Aggregation & Explanations
        # ------------------------------------------------------------------
        if not scores:
            return self._empty_result("Missing DEM raster arrays and optical RGB embeddings.")

        total_weight = sum(weights)
        final_score = float(sum(s * w for s, w in zip(scores, weights)) / total_weight)
        confidence = float(min(1.0, len(scores) / 4.0))

        # Qualitative explanation
        mean_slope_str = metrics[0].similarity if metrics else "N/A"
        explanation = (
            f"Geomorphology alignment score: {final_score:.1%}. "
            f"Evaluated {len(metrics)} topographic and visual indicators. "
        )
        if final_score >= 0.75:
            explanation += "High terrain traversability and landing slope profile resemblance."
        elif final_score >= 0.50:
            explanation += "Moderate terrain resemblance with partial slope and roughness divergences."
        else:
            explanation += "Significant topographic divergence; distinct surface ruggedness."

        return SectorResult(
            sector=self.sector,
            name=self.name,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            metrics=metrics,
            explanation=explanation,
        )
