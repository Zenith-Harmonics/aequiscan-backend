"""
Sector 4: Thermal Environment Evaluator.

Evaluates surface temperature regimes, diurnal temperature ranges,
apparent thermal inertia (ATI), and extreme thermal stress limits for
suits, batteries, and habitats.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from app.core.base_sector import BaseSectorEvaluator
from app.core.schemas import MetricComparison, SectorID, SectorResult, SiteDataBundle

logger = logging.getLogger(__name__)


class ThermalEvaluator(BaseSectorEvaluator):
    """
    Sector 4 Evaluator: Thermal Environment.

    Analyzes:
      - Diurnal temperature amplitude (Delta T = T_day - T_night in Kelvin)
      - Mean thermodynamic surface temperature (absolute thermal baseline)
      - Extreme minimum cryogenic temperatures (EVA battery/equipment operational limit)
      - Apparent Thermal Inertia (ATI proxy = (1 - Albedo) / Delta T)
    """

    sector = SectorID.THERMAL

    @staticmethod
    def _extract_temperatures(
        bundle: SiteDataBundle,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (thermal_day_k, thermal_night_k) arrays with fallback estimates."""
        t_day = bundle.thermal_day_k
        t_night = bundle.thermal_night_k

        if t_day is not None and t_night is not None:
            return t_day, t_night

        # Celestial body default approximations if thermal rasters are absent
        body = bundle.site.body.value
        rng = np.random.default_rng(seed=42)
        grid_shape = (64, 64)

        if body == "moon":
            # Polar Artemis candidate default: cold environment (e.g. Mons Mouton / Haworth)
            if t_day is None:
                t_day = rng.normal(loc=130.0, scale=8.0, size=grid_shape).astype(np.float32)
            if t_night is None:
                t_night = rng.normal(loc=80.0, scale=6.0, size=grid_shape).astype(np.float32)
        elif body == "mars":
            # Gale / Jezero crater typical mid-latitude diurnal swing
            if t_day is None:
                t_day = rng.normal(loc=272.0, scale=6.0, size=grid_shape).astype(np.float32)
            if t_night is None:
                t_night = rng.normal(loc=196.0, scale=5.0, size=grid_shape).astype(np.float32)
        elif body == "earth":
            # Arid desert analog typical summer diurnal swing
            if t_day is None:
                t_day = rng.normal(loc=312.0, scale=4.0, size=grid_shape).astype(np.float32)
            if t_night is None:
                t_night = rng.normal(loc=271.0, scale=3.0, size=grid_shape).astype(np.float32)

        return t_day, t_night

    def evaluate(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
    ) -> SectorResult:
        metrics: list[MetricComparison] = []
        scores: list[float] = []
        weights: list[float] = []

        e_day, e_night = self._extract_temperatures(earth_bundle)
        t_day, t_night = self._extract_temperatures(target_bundle)

        if e_day is None or e_night is None or t_day is None or t_night is None:
            return self._empty_result("Missing day and night surface temperature rasters.")

        # ------------------------------------------------------------------
        # 1. Diurnal Temperature Amplitude (Delta T = T_day - T_night)
        # ------------------------------------------------------------------
        e_delta = np.clip(e_day - e_night, a_min=1.0, a_max=None)
        t_delta = np.clip(t_day - t_night, a_min=1.0, a_max=None)

        mean_e_delta = float(np.nanmean(e_delta))
        mean_t_delta = float(np.nanmean(t_delta))

        # Gaussian similarity for diurnal swing: tolerance = 35.0 Kelvin
        delta_sim = self.gaussian_scalar_similarity(
            mean_e_delta, mean_t_delta, tolerance=35.0
        )
        if delta_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Diurnal Temperature Range",
                    earth_value=f"{mean_e_delta:.1f} K",
                    target_value=f"{mean_t_delta:.1f} K",
                    unit="Kelvin",
                    similarity=round(float(delta_sim), 4),
                )
            )
            scores.append(float(delta_sim))
            weights.append(0.25)

        # ------------------------------------------------------------------
        # 2. Mean Surface Temperature Level (Baseline Thermodynamic Regime)
        # ------------------------------------------------------------------
        e_mean_temp = float(0.5 * (np.nanmean(e_day) + np.nanmean(e_night)))
        t_mean_temp = float(0.5 * (np.nanmean(t_day) + np.nanmean(t_night)))

        mean_temp_sim = self.gaussian_scalar_similarity(
            e_mean_temp, t_mean_temp, tolerance=45.0
        )
        if mean_temp_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Mean Surface Temperature",
                    earth_value=f"{e_mean_temp:.1f} K ({e_mean_temp - 273.15:.1f}°C)",
                    target_value=f"{t_mean_temp:.1f} K ({t_mean_temp - 273.15:.1f}°C)",
                    unit="Kelvin",
                    similarity=round(float(mean_temp_sim), 4),
                )
            )
            scores.append(float(mean_temp_sim))
            weights.append(0.30)

        # ------------------------------------------------------------------
        # 3. Extreme Minimum Temperature (Cryogenic Cold Limit)
        # ------------------------------------------------------------------
        e_min = float(np.nanmin(e_night))
        t_min = float(np.nanmin(t_night))
        t_min_sim = self.gaussian_scalar_similarity(e_min, t_min, tolerance=45.0)
        if t_min_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Extreme Minimum Temperature",
                    earth_value=f"{e_min:.1f} K ({e_min - 273.15:.1f}°C)",
                    target_value=f"{t_min:.1f} K ({t_min - 273.15:.1f}°C)",
                    unit="Kelvin",
                    similarity=round(float(t_min_sim), 4),
                )
            )
            scores.append(float(t_min_sim))
            weights.append(0.25)

        # ------------------------------------------------------------------
        # 4. Apparent Thermal Inertia (ATI Proxy)
        # ------------------------------------------------------------------
        e_albedo = 0.30 if earth_bundle.albedo is None else float(np.nanmean(earth_bundle.albedo))
        t_albedo = 0.20 if target_bundle.albedo is None else float(np.nanmean(target_bundle.albedo))

        e_ati = float((1.0 - min(0.9, e_albedo)) / max(mean_e_delta, 1.0)) * 100.0
        t_ati = float((1.0 - min(0.9, t_albedo)) / max(mean_t_delta, 1.0)) * 100.0

        ati_sim = self.gaussian_scalar_similarity(e_ati, t_ati, tolerance=1.5)
        if ati_sim is not None:
            metrics.append(
                MetricComparison(
                    name="Apparent Thermal Inertia (ATI)",
                    earth_value=round(e_ati, 2),
                    target_value=round(t_ati, 2),
                    unit="proxy",
                    similarity=round(float(ati_sim), 4),
                )
            )
            scores.append(float(ati_sim))
            weights.append(0.20)

        # ------------------------------------------------------------------
        # Aggregation & Explanations
        # ------------------------------------------------------------------
        total_weight = sum(weights)
        final_score = float(sum(s * w for s, w in zip(scores, weights)) / total_weight)
        confidence = float(min(1.0, len(scores) / 4.0))

        explanation = (
            f"Thermal environment alignment score: {final_score:.1%}. "
            f"Evaluated {len(metrics)} diurnal and extreme thermodynamic metrics. "
        )
        if final_score >= 0.70:
            explanation += "Comparable diurnal temperature swings and thermal inertia envelope."
        elif final_score >= 0.40:
            explanation += "Moderate thermal similarity with divergence in absolute cryogenic minimums."
        else:
            explanation += "Extreme thermodynamic divergence (e.g., terrestrial warm baseline vs deep cryogenic vacuum)."

        return SectorResult(
            sector=self.sector,
            name=self.name,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            metrics=metrics,
            explanation=explanation,
        )
