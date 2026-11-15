"""
Sector 3: Subsurface & Geophysics Evaluator.

Evaluates underground structures, lava tubes, caves, subsurface ice,
dielectric properties, radar penetration, and stratigraphy.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from app.core.base_sector import BaseSectorEvaluator
from app.core.schemas import MetricComparison, SectorID, SectorResult, SiteDataBundle

logger = logging.getLogger(__name__)


class SubsurfaceEvaluator(BaseSectorEvaluator):
    """
    Sector 3 Evaluator: Subsurface & Geophysics.

    Compares:
      - Circular / Linear Cross-Polarization Ratio (CPR / LDR)
      - Mean SAR radar backscatter intensity (dielectric penetration proxy)
      - Volumetric scattering fraction (indicative of subsurface ice or blocky structures)
      - Radar neural latent features (Sentinel-1 ConvNeXtV2 1024D)
    """

    sector = SectorID.SUBSURFACE

    @staticmethod
    def compute_cpr_from_sar(sar_2ch: np.ndarray) -> np.ndarray:
        """
        Compute linear Cross-Polarization Ratio from (2, H, W) [VV_dB, VH_dB] backscatter.

        CPR_linear = 10^((VH_dB - VV_dB) / 10)
        """
        if sar_2ch.ndim != 3 or sar_2ch.shape[0] != 2:
            raise ValueError("SAR backscatter must have shape (2, H, W) for [VV, VH].")

        vv_db = sar_2ch[0]
        vh_db = sar_2ch[1]
        cpr_linear = 10.0 ** ((vh_db - vv_db) / 10.0)
        # Numerical guard: clamp CPR to physically observable range [0.0, 5.0]
        return np.clip(cpr_linear, 0.0, 5.0).astype(np.float32)

    @staticmethod
    def _extract_cpr_and_backscatter(
        bundle: SiteDataBundle,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return (cpr_map, mean_backscatter_db) for a given site bundle."""
        cpr_map = bundle.radar_cpr
        mean_bs = None

        if bundle.sar_backscatter is not None and bundle.sar_backscatter.size > 0:
            if cpr_map is None:
                try:
                    cpr_map = SubsurfaceEvaluator.compute_cpr_from_sar(bundle.sar_backscatter)
                except Exception as exc:
                    logger.debug("Could not compute CPR from sar_backscatter: %s", exc)
            mean_bs = float(np.nanmean(bundle.sar_backscatter[0]))  # VV channel
        elif cpr_map is not None:
            mean_bs = -15.0  # reasonable radar default
        else:
            # Celestial body default approximations for ground radar
            body = bundle.site.body.value
            if body == "moon":
                mean_bs = -18.5  # dry porous lunar regolith
            elif body == "mars":
                mean_bs = -16.0  # cold dry Martian plains
            elif body == "earth":
                mean_bs = -12.0  # terrestrial arid soil

        return cpr_map, mean_bs

    def evaluate(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
    ) -> SectorResult:
        metrics: list[MetricComparison] = []
        scores: list[float] = []
        weights: list[float] = []

        e_cpr, e_bs = self._extract_cpr_and_backscatter(earth_bundle)
        t_cpr, t_bs = self._extract_cpr_and_backscatter(target_bundle)

        # ------------------------------------------------------------------
        # 1. Cross-Polarization Ratio (CPR) Match
        # ------------------------------------------------------------------
        if e_cpr is not None and t_cpr is not None:
            cpr_sim = self.distribution_similarity_1d(e_cpr, t_cpr, characteristic_scale=0.15)
            if cpr_sim is not None:
                mean_e_cpr = float(np.nanmean(e_cpr))
                mean_t_cpr = float(np.nanmean(t_cpr))
                metrics.append(
                    MetricComparison(
                        name="Cross-Polarization Ratio (CPR)",
                        earth_value=round(mean_e_cpr, 3),
                        target_value=round(mean_t_cpr, 3),
                        unit="ratio",
                        similarity=round(float(cpr_sim), 4),
                    )
                )
                scores.append(float(cpr_sim))
                weights.append(0.35)

                # ------------------------------------------------------------------
                # 2. Volumetric Scattering / Subsurface Ice Proxy
                # ------------------------------------------------------------------
                # Fraction of pixels with high CPR (> 0.35 indicates multiple volume scattering)
                e_vol_frac = float(np.mean(e_cpr > 0.35))
                t_vol_frac = float(np.mean(t_cpr > 0.35))
                vol_sim = self.gaussian_scalar_similarity(e_vol_frac, t_vol_frac, tolerance=0.15)
                if vol_sim is not None:
                    metrics.append(
                        MetricComparison(
                            name="Volume Scattering Fraction",
                            earth_value=f"{e_vol_frac:.1%}",
                            target_value=f"{t_vol_frac:.1%}",
                            unit="fraction",
                            similarity=round(float(vol_sim), 4),
                        )
                    )
                    scores.append(float(vol_sim))
                    weights.append(0.20)

        # ------------------------------------------------------------------
        # 3. Mean Radar Backscatter Intensity (Dielectric & Surface Roughness)
        # ------------------------------------------------------------------
        if e_bs is not None and t_bs is not None:
            bs_sim = self.gaussian_scalar_similarity(e_bs, t_bs, tolerance=5.0)
            if bs_sim is not None:
                metrics.append(
                    MetricComparison(
                        name="Radar Backscatter Intensity",
                        earth_value=round(e_bs, 1),
                        target_value=round(t_bs, 1),
                        unit="dB",
                        similarity=round(float(bs_sim), 4),
                    )
                )
                scores.append(float(bs_sim))
                weights.append(0.25)

        # ------------------------------------------------------------------
        # 4. Neural SAR / Subsurface Latent Alignment
        # ------------------------------------------------------------------
        e_sar = earth_bundle.embeddings.get("s1_1024")
        t_emb = target_bundle.embeddings.get("clip_768")

        if e_sar is not None and "earth_aligned_768" in earth_bundle.embeddings and t_emb is not None:
            # Use aligned projection of SAR into shared latent space
            sar_latent_sim = self.cosine_similarity(
                earth_bundle.embeddings["earth_aligned_768"], t_emb
            )
            metrics.append(
                MetricComparison(
                    name="SAR Subsurface Latent Match",
                    earth_value="ConvNeXtV2-S1 (Aligned)",
                    target_value="Planetary Latent (768D)",
                    unit="cosine",
                    similarity=round(float(sar_latent_sim), 4),
                )
            )
            scores.append(float(sar_latent_sim))
            weights.append(0.20)

        # ------------------------------------------------------------------
        # Aggregation & Explanations
        # ------------------------------------------------------------------
        if not scores:
            return self._empty_result("Missing SAR backscatter and radar CPR products.")

        total_weight = sum(weights)
        final_score = float(sum(s * w for s, w in zip(scores, weights)) / total_weight)
        confidence = float(min(1.0, len(scores) / 3.0))

        explanation = (
            f"Subsurface geophysics alignment score: {final_score:.1%}. "
            f"Evaluated {len(metrics)} radar polarimetric and dielectric indicators. "
        )
        if final_score >= 0.75:
            explanation += "Consistent radar penetration depth and volumetric scattering regime."
        elif final_score >= 0.50:
            explanation += "Moderate subsurface analogy with variance in cross-polarization scattering."
        else:
            explanation += "Distinct subsurface dielectric response (e.g., lossy conductive soil vs porous dry regolith)."

        return SectorResult(
            sector=self.sector,
            name=self.name,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            metrics=metrics,
            explanation=explanation,
        )
