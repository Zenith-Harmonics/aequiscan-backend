"""
API version 1 - aggregated router.
"""
from fastapi import APIRouter

from app.api.v1.analogs import router as analogs_router
from app.api.v1.health import router as health_router
from app.api.v1.mineralogy import router as mineralogy_router
from app.api.v1.targets import router as targets_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health_router)
v1_router.include_router(targets_router)
v1_router.include_router(mineralogy_router)
v1_router.include_router(analogs_router)