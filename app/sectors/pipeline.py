"""
Planetary Analog Multi-Sector Pipeline.

Orchestrates all six scientific pillars to produce a unified,
mission-weighted AnalogComparisonReport with explainability.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.core.schemas import (
    MISSION_WEIGHTS,
    AnalogComparisonReport,
    AnalogComparisonRequest,
    MissionProfile,
    SectorID,
    SectorResult,
    SiteDataBundle,
)
from app.sectors.astrobiology import AstrobiologyEvaluator
from app.sectors.atmospheric import AtmosphericEvaluator
from app.sectors.geomorphology import GeomorphologyEvaluator
from app.sectors.mineralogy import MineralogyEvaluator
from app.sectors.subsurface import SubsurfaceEvaluator
from app.sectors.thermal import ThermalEvaluator

logger = logging.getLogger(__name__)


class PlanetaryAnalogPipeline:
    """
    Central comparative engine executing the 6 scientific pillars:
      1. Geomorphology & Topography
      2. Mineralogy & Geochemistry
      3. Subsurface & Geophysics
      4. Thermal Environment
      5. Atmospheric & Aeolian Processes
      6. Astrobiology & Habitability
    """

    def __init__(self) -> None:
        self.evaluators: Dict[SectorID, Any] = {
            SectorID.GEOMORPHOLOGY: GeomorphologyEvaluator(),
            SectorID.MINERALOGY: MineralogyEvaluator(),
            SectorID.SUBSURFACE: SubsurfaceEvaluator(),
            SectorID.THERMAL: ThermalEvaluator(),
            SectorID.ATMOSPHERIC: AtmosphericEvaluator(),
            SectorID.ASTROBIOLOGY: AstrobiologyEvaluator(),
        }

    def _resolve_weights(
        self,
        mission_profile: MissionProfile,
        custom_weights: Optional[Dict[SectorID, float]] = None,
    ) -> Dict[SectorID, float]:
        """Resolve and normalize sector weights to sum strictly to 1.0."""
        if custom_weights:
            raw_weights = {s: float(custom_weights.get(s, 0.0)) for s in SectorID}
            total = sum(raw_weights.values())
            if total <= 0:
                raise ValueError("Custom weights must sum to a positive number.")
            return {s: w / total for s, w in raw_weights.items()}

        profile_weights = MISSION_WEIGHTS.get(mission_profile, MISSION_WEIGHTS[MissionProfile.BALANCED])
        total = sum(profile_weights.values())
        return {s: w / total for s, w in profile_weights.items()}

    def compare(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
        mission_profile: MissionProfile = MissionProfile.BALANCED,
        custom_weights: Optional[Dict[SectorID, float]] = None,
    ) -> AnalogComparisonReport:
        """
        Execute full cross-domain multi-sector evaluation.

        Parameters
        ----------
        earth_bundle : SiteDataBundle
            Terrestrial analog candidate data bundle.
        target_bundle : SiteDataBundle
            Lunar or Martian target site data bundle.
        mission_profile : MissionProfile
            NASA mission architecture profile (e.g. balanced, traversability, astrobiology).
        custom_weights : Optional[Dict[SectorID, float]]
            User-defined sector weight overrides.
        """
        weights = self._resolve_weights(mission_profile, custom_weights)
        sector_results: List[SectorResult] = []
        overall_score = 0.0
        overall_conf = 0.0

        key_strengths: List[str] = []
        key_divergences: List[str] = []

        # Evaluate each of the 6 pillars
        for sector_id in SectorID:
            evaluator = self.evaluators[sector_id]
            try:
                result = evaluator.evaluate(earth_bundle, target_bundle)
            except Exception as exc:
                logger.error("Sector %s evaluation error: %s", sector_id, exc, exc_info=True)
                result = evaluator._empty_result(f"Evaluation error: {exc}")

            w = weights[sector_id]
            result.weighted_contribution = round(result.score * w, 4)
            sector_results.append(result)

            overall_score += result.score * w
            overall_conf += result.confidence * w

            # Synthesize automated explainability
            if result.score >= 0.72:
                top_metrics = [m.name for m in result.metrics if m.similarity >= 0.75]
                metric_desc = f" ({', '.join(top_metrics)})" if top_metrics else ""
                key_strengths.append(
                    f"{result.name}: High parity score ({result.score:.1%}){metric_desc}"
                )
            elif result.score < 0.45:
                low_metrics = [
                    f"{m.name} ({m.earth_value} vs {m.target_value})"
                    for m in result.metrics
                    if m.similarity < 0.40
                ]
                metric_desc = f" [{'; '.join(low_metrics)}]" if low_metrics else ""
                key_divergences.append(
                    f"{result.name}: Substantial divergence ({result.score:.1%}){metric_desc}"
                )

        if not key_strengths:
            key_strengths.append("Baseline operational analogy with moderate cross-sector alignment.")
        if not key_divergences:
            key_divergences.append("No critical physical divergences identified across evaluated sectors.")

        return AnalogComparisonReport(
            earth_site=earth_bundle.site,
            target_site=target_bundle.site,
            mission_profile=mission_profile,
            overall_similarity_score=round(float(overall_score), 4),
            overall_confidence=round(float(overall_conf), 4),
            sector_results=sector_results,
            key_strengths=key_strengths,
            key_divergences=key_divergences,
        )
