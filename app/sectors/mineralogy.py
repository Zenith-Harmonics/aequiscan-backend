"""
Sector 2: Mineralogy & Geochemistry Evaluator.

Evaluates surface composition, diagnostic mineral axes (mafic, plagioclase,
phyllosilicates, sulfates, carbonates, iron oxides), bolometric albedo,
and multi-spectral / hyperspectral signatures.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from app.core.base_sector import BaseSectorEvaluator
from app.core.constants import MINERAL_CLASSES
from app.core.schemas import MetricComparison, SectorID, SectorResult, SiteDataBundle
from app.services.similarity import compute_mineral_asi

logger = logging.getLogger(__name__)


class MineralogyEvaluator(BaseSectorEvaluator):
    """
    Sector 2 Evaluator: Mineralogy & Geochemistry.

    Compares:
      - Diagnostic 6-axis mineralogical profile via Analog Similarity Index (ASI)
      - Surface bolometric albedo
      - Multi-spectral band ratios (Clay/SWIR, Iron Oxide/VIS, Evaporite/NDSI)
      - Spectral latent representation alignment (Sentinel-2 1024D / RemoteCLIP)
    """

    sector = SectorID.MINERALOGY

    @staticmethod
    def _extract_profile(bundle: SiteDataBundle) -> Optional[Dict[str, float]]:
        """Extract or estimate a 6-axis mineral fraction profile from a SiteDataBundle."""
        # 1. Directly in bundle mineral_indices under 'harmonized_profile'
        if "harmonized_profile" in bundle.mineral_indices:
            prof = bundle.mineral_indices["harmonized_profile"]
            if isinstance(prof, dict):
                return {cls: float(prof.get(cls, 0.0)) for cls in MINERAL_CLASSES}

        # 2. Check if mineral_indices keys contain the mineral classes directly
        matched_classes = {
            cls: float(np.nanmean(bundle.mineral_indices[cls]))
            for cls in MINERAL_CLASSES
            if cls in bundle.mineral_indices and np.isscalar(bundle.mineral_indices[cls]) or (
                isinstance(bundle.mineral_indices.get(cls), np.ndarray)
                and bundle.mineral_indices[cls].size > 0
            )
        }
        if len(matched_classes) >= 3:
            total = sum(matched_classes.values())
            if total > 0:
                return {cls: val / total for cls, val in matched_classes.items()}
            return matched_classes

        # 3. Fallback based on celestial body if reference profile is known
        body = bundle.site.body.value
        if body == "mars":
            return {
                "mafic_olivine_pyroxene": 0.22,
                "plagioclase_anorthosite": 0.05,
                "phyllosilicates_clays": 0.34,
                "sulfates_evaporites": 0.26,
                "carbonates": 0.03,
                "iron_oxides_hydroxides": 0.10,
            }
        elif body == "moon":
            return {
                "mafic_olivine_pyroxene": 0.14,
                "plagioclase_anorthosite": 0.81,
                "phyllosilicates_clays": 0.00,
                "sulfates_evaporites": 0.00,
                "carbonates": 0.00,
                "iron_oxides_hydroxides": 0.05,
            }
        elif body == "earth":
            # Default terrestrial arid evaporite playa baseline (Qaidam/Dalangtan style)
            return {
                "mafic_olivine_pyroxene": 0.09,
                "plagioclase_anorthosite": 0.04,
                "phyllosilicates_clays": 0.23,
                "sulfates_evaporites": 0.50,
                "carbonates": 0.06,
                "iron_oxides_hydroxides": 0.08,
            }
        return None

    @staticmethod
    def _compute_mean_albedo(bundle: SiteDataBundle) -> Optional[float]:
        """Extract or approximate surface bolometric albedo [0.0, 1.0]."""
        if bundle.albedo is not None:
            valid = bundle.albedo[np.isfinite(bundle.albedo)]
            if valid.size > 0:
                return float(np.clip(np.nanmean(valid), 0.0, 1.0))

        # Approximate albedo from optical RGB or S2 multispectral bands if available
        if bundle.s2_multispectral is not None and bundle.s2_multispectral.ndim == 3:
            # S2 reflectance is scaled 0-10000; visible & NIR mean
            vis_nir = np.nanmean(bundle.s2_multispectral[:4])
            if vis_nir > 1.0:
                return float(np.clip(vis_nir / 10000.0, 0.0, 1.0))
            return float(np.clip(vis_nir, 0.0, 1.0))

        if bundle.rgb_image is not None and bundle.rgb_image.size > 0:
            return float(np.nanmean(bundle.rgb_image) / 255.0)

        # Celestial body default typical albedo
        if bundle.site.body.value == "moon":
            return 0.12  # Lunar regolith average
        elif bundle.site.body.value == "mars":
            return 0.25  # Martian global dust/surface average
        elif bundle.site.body.value == "earth":
            return 0.30  # Desert playa / soil average
        return None

    def evaluate(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
    ) -> SectorResult:
        metrics: list[MetricComparison] = []
        scores: list[float] = []
        weights: list[float] = []

        # ------------------------------------------------------------------
        # 1. 6-Axis Mineralogical Analog Similarity Index (ASI)
        # ------------------------------------------------------------------
        earth_prof = self._extract_profile(earth_bundle)
        target_prof = self._extract_profile(target_bundle)

        if earth_prof is not None and target_prof is not None:
            asi_score, _ = compute_mineral_asi(earth_prof, target_prof)
            # Find top dominant mineral in each
            top_e = max(earth_prof.items(), key=lambda kv: kv[1])[0]
            top_t = max(target_prof.items(), key=lambda kv: kv[1])[0]

            metrics.append(
                MetricComparison(
                    name="6-Axis Mineralogy ASI",
                    earth_value=f"Primary: {top_e} ({earth_prof[top_e]:.1%})",
                    target_value=f"Primary: {top_t} ({target_prof[top_t]:.1%})",
                    unit="ASI",
                    similarity=round(float(asi_score), 4),
                )
            )
            scores.append(float(asi_score))
            weights.append(0.50)

        # ------------------------------------------------------------------
        # 2. Surface Bolometric Albedo Match
        # ------------------------------------------------------------------
        e_albedo = self._compute_mean_albedo(earth_bundle)
        t_albedo = self._compute_mean_albedo(target_bundle)

        if e_albedo is not None and t_albedo is not None:
            # Gaussian similarity with tolerance = 0.10
            albedo_sim = self.gaussian_scalar_similarity(e_albedo, t_albedo, tolerance=0.10)
            if albedo_sim is not None:
                metrics.append(
                    MetricComparison(
                        name="Bolometric Albedo",
                        earth_value=round(e_albedo, 3),
                        target_value=round(t_albedo, 3),
                        unit="reflectance",
                        similarity=round(float(albedo_sim), 4),
                    )
                )
                scores.append(float(albedo_sim))
                weights.append(0.20)

        # ------------------------------------------------------------------
        # 3. Multi-spectral / RemoteCLIP Latent Space Match
        # ------------------------------------------------------------------
        e_emb = earth_bundle.embeddings.get(
            "earth_aligned_768", earth_bundle.embeddings.get("clip_768")
        )
        t_emb = target_bundle.embeddings.get("clip_768")

        if e_emb is not None and t_emb is not None:
            spectral_sim = self.cosine_similarity(e_emb, t_emb)
            metrics.append(
                MetricComparison(
                    name="Spectral Latent Embedding Match",
                    earth_value="MultiModal Aligned (768D)",
                    target_value="Planetary Latent (768D)",
                    unit="cosine",
                    similarity=round(float(spectral_sim), 4),
                )
            )
            scores.append(float(spectral_sim))
            weights.append(0.30)

        # ------------------------------------------------------------------
        # Aggregation & Explanations
        # ------------------------------------------------------------------
        if not scores:
            return self._empty_result("Missing mineral profiles, albedo, and spectral embeddings.")

        total_weight = sum(weights)
        final_score = float(sum(s * w for s, w in zip(scores, weights)) / total_weight)
        confidence = float(min(1.0, len(scores) / 3.0))

        explanation = (
            f"Mineralogy alignment score: {final_score:.1%}. "
            f"Evaluated {len(metrics)} geochemical and spectral dimensions. "
        )
        if final_score >= 0.75:
            explanation += "Strong geochemical parity with compatible alteration or lithology assemblages."
        elif final_score >= 0.50:
            explanation += "Partial compositional overlap with notable mineralogical fraction shifts."
        else:
            explanation += "Pronounced lithological divergence (e.g., highly altered clays/sulfates vs pristine anorthosite)."

        return SectorResult(
            sector=self.sector,
            name=self.name,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            metrics=metrics,
            explanation=explanation,
        )
