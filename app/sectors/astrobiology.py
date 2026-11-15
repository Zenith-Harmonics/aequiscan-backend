"""
Sector 6: Astrobiology & Habitability Evaluator.

Evaluates extreme survival limits, biosignature preservation potential (BPP),
soil water activity (a_w), geochemical envelopes (pH, salinity), and oxidant stress.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from app.core.base_sector import BaseSectorEvaluator
from app.core.schemas import MetricComparison, SectorID, SectorResult, SiteDataBundle

logger = logging.getLogger(__name__)


class AstrobiologyEvaluator(BaseSectorEvaluator):
    """
    Sector 6 Evaluator: Astrobiology & Habitability.

    Analyzes:
      - Soil water activity envelope (a_w vs terrestrial thermodynamic life limits ~0.605)
      - Biosignature Preservation Potential (BPP) of the mineral matrix
      - Geochemical environmental stress (pH and hypersalinity)
      - Oxidant and perchlorate stress levels
    """

    sector = SectorID.ASTROBIOLOGY

    @staticmethod
    def _get_astrobiology_profile(bundle: SiteDataBundle) -> Dict[str, float]:
        """Extract astrobiology scalars with physics-anchored defaults."""
        scalars = dict(bundle.astrobiology_scalars)
        body = bundle.site.body.value

        defaults = {
            "moon": {
                "soil_water_activity_aw": 0.0,
                "organic_preservation_potential": 0.05,  # irradiated regolith
                "ph": 7.0,                               # neutral anhydrous
                "salinity_ppt": 0.0,
                "perchlorate_concentration_pct": 0.0,
            },
            "mars": {
                "soil_water_activity_aw": 0.15,          # sub-critical water activity
                "organic_preservation_potential": 0.75,  # high in clay/sulfate beds
                "ph": 4.5,                               # acid-sulfate to neutral
                "salinity_ppt": 120.0,                   # evaporitic brines
                "perchlorate_concentration_pct": 0.60,   # Phoenix/Curiosity oxidants
            },
            "earth": {
                # Extreme desert playa / salt flat analog default (Qaidam/Atacama)
                "soil_water_activity_aw": 0.35,
                "organic_preservation_potential": 0.80,  # clay/evaporite preservation
                "ph": 6.8,
                "salinity_ppt": 180.0,                   # hypersaline halite/gypsum
                "perchlorate_concentration_pct": 0.05,   # natural Atacama perchlorates
            },
        }

        profile = defaults.get(body, defaults["earth"]).copy()
        profile.update(scalars)
        return profile

    def evaluate(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
    ) -> SectorResult:
        metrics: list[MetricComparison] = []
        scores: list[float] = []
        weights: list[float] = []

        e_bio = self._get_astrobiology_profile(earth_bundle)
        t_bio = self._get_astrobiology_profile(target_bundle)

        # ------------------------------------------------------------------
        # 1. Soil Water Activity (a_w) Thermodynamic Limit
        # ------------------------------------------------------------------
        e_aw = float(np.clip(e_bio.get("soil_water_activity_aw", 0.35), 0.0, 1.0))
        t_aw = float(np.clip(t_bio.get("soil_water_activity_aw", 0.15), 0.0, 1.0))
        # Tolerance: 0.20 water activity
        aw_sim = self.gaussian_scalar_similarity(e_aw, t_aw, tolerance=0.20)

        if aw_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Water Activity (a_w) Envelope",
                    earth_value=f"{e_aw:.3f} a_w",
                    target_value=f"{t_aw:.3f} a_w",
                    unit="a_w",
                    similarity=round(float(aw_sim), 4),
                )
            )
            scores.append(float(aw_sim))
            weights.append(0.35)

        # ------------------------------------------------------------------
        # 2. Biosignature Preservation Potential (BPP)
        # ------------------------------------------------------------------
        e_bpp = float(np.clip(e_bio.get("organic_preservation_potential", 0.80), 0.0, 1.0))
        t_bpp = float(np.clip(t_bio.get("organic_preservation_potential", 0.75), 0.0, 1.0))
        bpp_sim = self.gaussian_scalar_similarity(e_bpp, t_bpp, tolerance=0.25)

        if bpp_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Biosignature Preservation Potential",
                    earth_value=f"{e_bpp:.1%}",
                    target_value=f"{t_bpp:.1%}",
                    unit="score",
                    similarity=round(float(bpp_sim), 4),
                )
            )
            scores.append(float(bpp_sim))
            weights.append(0.30)

        # ------------------------------------------------------------------
        # 3. Geochemical Extremophile Envelope (pH & Salinity)
        # ------------------------------------------------------------------
        e_ph = float(e_bio.get("ph", 6.8))
        t_ph = float(t_bio.get("ph", 4.5))
        ph_sim = self.gaussian_scalar_similarity(e_ph, t_ph, tolerance=3.0)

        e_sal = float(e_bio.get("salinity_ppt", 180.0))
        t_sal = float(t_bio.get("salinity_ppt", 120.0))
        sal_sim = self.gaussian_scalar_similarity(e_sal, t_sal, tolerance=80.0)

        if ph_sim is not None and sal_sim is not None:
            geochem_sim = 0.5 * (ph_sim + sal_sim)
            metrics.append(
                MetricComparison(
                    name="Geochemical Envelope (pH & Salinity)",
                    earth_value=f"pH {e_ph:.1f}, {e_sal:.0f} ppt",
                    target_value=f"pH {t_ph:.1f}, {t_sal:.0f} ppt",
                    unit="pH/ppt",
                    similarity=round(float(geochem_sim), 4),
                )
            )
            scores.append(float(geochem_sim))
            weights.append(0.20)

        # ------------------------------------------------------------------
        # 4. Oxidant / Perchlorate Stress Level
        # ------------------------------------------------------------------
        e_perc = float(e_bio.get("perchlorate_concentration_pct", 0.05))
        t_perc = float(t_bio.get("perchlorate_concentration_pct", 0.60))
        perc_sim = self.gaussian_scalar_similarity(e_perc, t_perc, tolerance=0.50)

        if perc_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Oxidant & Perchlorate Stress",
                    earth_value=f"{e_perc:.2f}% ClO4-",
                    target_value=f"{t_perc:.2f}% ClO4-",
                    unit="wt %",
                    similarity=round(float(perc_sim), 4),
                )
            )
            scores.append(float(perc_sim))
            weights.append(0.15)

        # ------------------------------------------------------------------
        # Aggregation & Explanations
        # ------------------------------------------------------------------
        total_weight = sum(weights)
        final_score = float(sum(s * w for s, w in zip(scores, weights)) / total_weight)
        confidence = float(min(1.0, len(scores) / 4.0))

        explanation = (
            f"Astrobiology and habitability alignment score: {final_score:.1%}. "
            f"Evaluated {len(metrics)} preservation and thermodynamic survival criteria. "
        )
        if final_score >= 0.75:
            explanation += "High biosignature preservation synergy; ideal analog for life detection instrument testing."
        elif final_score >= 0.50:
            explanation += "Moderate habitability analogy; compatible geochemical matrix with partial oxidant divergence."
        else:
            explanation += "Low astrobiological preservation parity or extreme habitability divergence."

        return SectorResult(
            sector=self.sector,
            name=self.name,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            metrics=metrics,
            explanation=explanation,
        )
