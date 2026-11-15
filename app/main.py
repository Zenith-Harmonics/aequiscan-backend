"""
AequiScan API - FastAPI application entry point.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import v1_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Cross-planetary analog comparison dashboard API. "
            "Compares Earth, Moon, and Mars surface properties across "
            "seven analytical axes for mission planning and science discovery."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -- CORS -----------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Routers --------------------------------------------------
    app.include_router(v1_router)

    # -- Logging --------------------------------------------------
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    return app


app = create_app()
