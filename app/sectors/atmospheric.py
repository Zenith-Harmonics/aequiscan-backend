"""
Sector 5: Atmospheric & Aeolian Processes Evaluator.

Evaluates wind dynamics, aeolian bedforms (dunes, yardangs), aridity index,
atmospheric surface pressure, dust transport, and UV radiation exposure.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from app.core.base_sector import BaseSectorEvaluator
from app.core.schemas import MetricComparison, SectorID, SectorResult, SiteDataBundle

logger = logging.getLogger(__name__)


class AtmosphericEvaluator(BaseSectorEvaluator):
    """
    Sector 5 Evaluator: Atmospheric & Aeolian Processes.

    Compares:
      - Aridity & desiccation index (moisture deficit benchmark)
      - Aeolian wind speed regime & friction dynamics
      - Atmospheric surface pressure (logarithmic barometric scale)
      - Surface ultraviolet (UV) radiation flux
    """

    sector = SectorID.ATMOSPHERIC

    @staticmethod
    def _get_atmospheric_profile(bundle: SiteDataBundle) -> Dict[str, float]:
        """Extract atmospheric scalars with physics-anchored celestial body defaults."""
        scalars = dict(bundle.atmospheric_scalars)
        body = bundle.site.body.value

        defaults = {
            "moon": {
                "surface_pressure_pa": 1e-10,  # lunar surface vacuum
                "mean_wind_speed_ms": 0.0,     # no atmosphere
                "aridity_index": 0.0,          # zero moisture
                "uv_flux_wm2": 110.0,          # unshielded solar UV
                "dust_optical_depth": 0.0,
            },
            "mars": {
                "surface_pressure_pa": 610.0,  # ~6.1 mbar mean surface pressure
                "mean_wind_speed_ms": 5.5,     # typical daytime saltation wind
                "aridity_index": 0.0,          # completely hyper-arid
                "uv_flux_wm2": 65.0,           # thin CO2 atmosphere, high UV-C
                "dust_optical_depth": 0.55,    # background Martian atmospheric dust
            },
            "earth": {
                "surface_pressure_pa": 101325.0,  # 1 atm sea level / elevation adjusted
                "mean_wind_speed_ms": 6.0,        # desert wind
                "aridity_index": 0.03,            # hyper-arid analog (Atacama / Qaidam)
                "uv_flux_wm2": 35.0,              # terrestrial ground UV flux
                "dust_optical_depth": 0.20,
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

        e_atm = self._get_atmospheric_profile(earth_bundle)
        t_atm = self._get_atmospheric_profile(target_bundle)

        # ------------------------------------------------------------------
        # 1. Aridity & Extreme Desiccation Match
        # ------------------------------------------------------------------
        # In planetary science, hyper-arid analogs (P/PET < 0.05) best match Mars/Moon desiccation
        e_arid = float(e_atm.get("aridity_index", 0.03))
        t_arid = float(t_atm.get("aridity_index", 0.0))
        aridity_sim = self.gaussian_scalar_similarity(e_arid, t_arid, tolerance=0.08)

        if aridity_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Aridity & Desiccation Index",
                    earth_value=f"{e_arid:.3f} (Hyper-arid)",
                    target_value=f"{t_arid:.3f} (Zero moisture)",
                    unit="P/PET",
                    similarity=round(float(aridity_sim), 4),
                )
            )
            scores.append(float(aridity_sim))
            weights.append(0.35)

        # ------------------------------------------------------------------
        # 2. Aeolian Wind Regime & Surface Dynamics
        # ------------------------------------------------------------------
        e_wind = float(e_atm.get("mean_wind_speed_ms", 6.0))
        t_wind = float(t_atm.get("mean_wind_speed_ms", 0.0))
        # Gaussian tolerance for aeolian shaping: 4.0 m/s
        wind_sim = self.gaussian_scalar_similarity(e_wind, t_wind, tolerance=4.0)

        if wind_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Aeolian Wind Regime",
                    earth_value=f"{e_wind:.1f} m/s",
                    target_value=f"{t_wind:.1f} m/s",
                    unit="m/s",
                    similarity=round(float(wind_sim), 4),
                )
            )
            scores.append(float(wind_sim))
            weights.append(0.25)

        # ------------------------------------------------------------------
        # 3. Atmospheric Surface Pressure (Logarithmic Scale)
        # ------------------------------------------------------------------
        e_pres = max(1e-12, float(e_atm.get("surface_pressure_pa", 101325.0)))
        t_pres = max(1e-12, float(t_atm.get("surface_pressure_pa", 610.0)))

        # Compare orders of magnitude difference in barometric pressure
        log_e = float(np.log10(e_pres))
        log_t = float(np.log10(t_pres))
        # Tolerance: 2.5 orders of magnitude
        pres_sim = self.gaussian_scalar_similarity(log_e, log_t, tolerance=2.5)

        if pres_sim is not None:
            e_str = f"{e_pres:.0f} Pa" if e_pres > 10 else f"{e_pres:.1e} Pa"
            t_str = f"{t_pres:.0f} Pa" if t_pres > 10 else f"{t_pres:.1e} Pa"
            metrics.append(
                MetricComparison(
                    name="Atmospheric Pressure Regime",
                    earth_value=e_str,
                    target_value=t_str,
                    unit="Pascals",
                    similarity=round(float(pres_sim), 4),
                )
            )
            scores.append(float(pres_sim))
            weights.append(0.20)

        # ------------------------------------------------------------------
        # 4. Surface UV Radiation Exposure
        # ------------------------------------------------------------------
        e_uv = float(e_atm.get("uv_flux_wm2", 35.0))
        t_uv = float(t_atm.get("uv_flux_wm2", 65.0))
        uv_sim = self.gaussian_scalar_similarity(e_uv, t_uv, tolerance=35.0)

        if uv_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Surface UV Radiation Flux",
                    earth_value=f"{e_uv:.1f} W/m²",
                    target_value=f"{t_uv:.1f} W/m²",
                    unit="W/m²",
                    similarity=round(float(uv_sim), 4),
                )
            )
            scores.append(float(uv_sim))
            weights.append(0.20)

        # ------------------------------------------------------------------
        # Aggregation & Explanations
        # ------------------------------------------------------------------
        total_weight = sum(weights)
        final_score = float(sum(s * w for s, w in zip(scores, weights)) / total_weight)
        confidence = float(min(1.0, len(scores) / 4.0))

        explanation = (
            f"Atmospheric processes alignment score: {final_score:.1%}. "
            f"Evaluated {len(metrics)} aridity, barometric, and aeolian factors. "
        )
        if final_score >= 0.70:
            explanation += "Strong analog match in hyper-aridity and aeolian transport dynamics."
        elif final_score >= 0.40:
            explanation += "Moderate aeolian bedform analogy, but notable barometric density difference."
        else:
            explanation += "Substantial atmospheric divergence (e.g., terrestrial dense troposphere vs vacuum/near-vacuum)."

        return SectorResult(
            sector=self.sector,
            name=self.name,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            metrics=metrics,
            explanation=explanation,
        )
