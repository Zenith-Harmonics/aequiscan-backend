"""
Centralised settings loaded from environment / .env file.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────
    app_name: str = "AequiScan API"
    app_version: str = "0.1.0"
    aequiscan_env: str = "development"
    debug: bool = False

    # ── CORS ─────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ── NASA Earthdata ───────────────────────────────────────────────
    earthdata_username: str = ""
    earthdata_password: str = ""
    earthdata_token: str = ""

    # ── Planetary data ───────────────────────────────────────────────
    pds_ode_base_url: str = "https://oderest.rsl.wustl.edu/live2/"
    nasa_trek_base_url: str = "https://trek.nasa.gov"


@lru_cache
def get_settings() -> Settings:
    return Settings()
