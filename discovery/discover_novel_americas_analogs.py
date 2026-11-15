"""
Self-Contained AequiScan Terrestrial Analog Discovery Engine (North & South America).

Integrates the complete AequiScan pipeline into a single standalone script:
1. Core Schemas, Constants, Mission Weights & Disk Cache
2. Live NASA/ESA/ISRIC Ingestion:
   - Microsoft Planetary Computer STAC (Copernicus DEM 30m, Sentinel-2 L2A, Landsat C2 L2 Thermal, Sentinel-1 RTC SAR + Lee Filter)
   - NASA POWER API (Meteorology, Radiation, UNESCO Hargreaves-Samani Aridity)
   - ISRIC SoilGrids v2.0 (Topsoil pH & Soil Organic Carbon)
   - NASA earthaccess EMITL2BMIN (60m Hyperspectral Mineralogy)
   - NASA PDS Orbital Data Explorer (MRO CRISM/HiRISE/MCS, ODY THEMIS, LRO LROC/Diviner/Mini-RF, Chandrayaan-1 M3)
3. Neural Feature Extractors & Cross-Domain Aligner:
   - BIFOLD BigEarthNet v2.0 ConvNeXtV2-Base (Sentinel-1 & Sentinel-2)
   - RemoteCLIP ViT-L-14
   - LatentSpaceAligner (Ridge / CCA)
4. All 6 Scientific Sector Evaluators + PlanetaryAnalogPipeline
5. Americas Frontier Discovery Scanner:
   - Excludes famous/already-used analog sites (Mojave, Atacama Yungay, MDRS Utah, Devon Island, etc.)
   - Fast Tier-0 NASA POWER aridity pre-filter
   - Two-stage coarse-to-fine spatial sweep with live console telemetry & incremental saving
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import hashlib
import inspect
import io
import json
import logging
import math
import os
import re
import time
import urllib.parse
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin, get_type_hints

import httpx
import joblib
import numpy as np
import open_clip
import planetary_computer
import pystac_client
import rasterio
import torch
import torch.nn.functional as F
from configilm.ConfigILM import ConfigILM as ILMModel
from configilm.ConfigILM import ILMConfiguration
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.ndimage import uniform_filter
from scipy.stats import wasserstein_distance
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge

try:
    import earthaccess
except ModuleNotFoundError:
    earthaccess = None  # type: ignore[assignment]

try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None  # type: ignore[assignment]

try:
    from global_land_mask import globe
except ImportError:
    globe = None

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("rasterio").setLevel(logging.WARNING)

logger = logging.getLogger("aequiscan_standalone")


# ===========================================================================
# 1. CONFIGURATION, CONSTANTS & SCHEMAS
# ===========================================================================
PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
SOILGRIDS_QUERY_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PDS_ODE_BASE_URL = "https://oderest.rsl.wustl.edu/live2/"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".aequiscan_cache"

MINERAL_CLASSES: List[str] = [
    "mafic_olivine_pyroxene",
    "plagioclase_anorthosite",
    "phyllosilicates_clays",
    "sulfates_evaporites",
    "carbonates",
    "iron_oxides_hydroxides",
]

EMIT_KEYWORD_MAP: Dict[str, str] = {
    "olivine": "mafic_olivine_pyroxene",
    "forsterite": "mafic_olivine_pyroxene",
    "fayalite": "mafic_olivine_pyroxene",
    "pyroxene": "mafic_olivine_pyroxene",
    "enstatite": "mafic_olivine_pyroxene",
    "augite": "mafic_olivine_pyroxene",
    "diopside": "mafic_olivine_pyroxene",
    "plagioclase": "plagioclase_anorthosite",
    "anorthite": "plagioclase_anorthosite",
    "albite": "plagioclase_anorthosite",
    "labradorite": "plagioclase_anorthosite",
    "montmorillonite": "phyllosilicates_clays",
    "smectite": "phyllosilicates_clays",
    "nontronite": "phyllosilicates_clays",
    "saponite": "phyllosilicates_clays",
    "kaolinite": "phyllosilicates_clays",
    "illite": "phyllosilicates_clays",
    "chlorite": "phyllosilicates_clays",
    "serpentine": "phyllosilicates_clays",
    "muscovite": "phyllosilicates_clays",
    "gypsum": "sulfates_evaporites",
    "anhydrite": "sulfates_evaporites",
    "jarosite": "sulfates_evaporites",
    "alunite": "sulfates_evaporites",
    "halite": "sulfates_evaporites",
    "kieserite": "sulfates_evaporites",
    "epsomite": "sulfates_evaporites",
    "thenardite": "sulfates_evaporites",
    "calcite": "carbonates",
    "dolomite": "carbonates",
    "magnesite": "carbonates",
    "siderite": "carbonates",
    "hematite": "iron_oxides_hydroxides",
    "goethite": "iron_oxides_hydroxides",
    "ferrihydrite": "iron_oxides_hydroxides",
    "magnetite": "iron_oxides_hydroxides",
}


class PlanetBody(str, Enum):
    EARTH = "earth"
    MARS = "mars"
    MOON = "moon"


class SectorID(str, Enum):
    GEOMORPHOLOGY = "geomorphology"
    MINERALOGY = "mineralogy"
    SUBSURFACE = "subsurface"
    THERMAL = "thermal"
    ATMOSPHERIC = "atmospheric"
    ASTROBIOLOGY = "astrobiology"


class MissionProfile(str, Enum):
    BALANCED = "balanced"
    ASTROBIOLOGY = "astrobiology"
    TRAVERSABILITY = "traversability"
    ISRU = "isru"


MISSION_WEIGHTS: Dict[MissionProfile, Dict[SectorID, float]] = {
    MissionProfile.BALANCED: {
        SectorID.GEOMORPHOLOGY: 0.20,
        SectorID.MINERALOGY: 0.20,
        SectorID.SUBSURFACE: 0.15,
        SectorID.THERMAL: 0.15,
        SectorID.ATMOSPHERIC: 0.15,
        SectorID.ASTROBIOLOGY: 0.15,
    },
    MissionProfile.ASTROBIOLOGY: {
        SectorID.GEOMORPHOLOGY: 0.10,
        SectorID.MINERALOGY: 0.25,
        SectorID.SUBSURFACE: 0.10,
        SectorID.THERMAL: 0.10,
        SectorID.ATMOSPHERIC: 0.10,
        SectorID.ASTROBIOLOGY: 0.35,
    },
    MissionProfile.TRAVERSABILITY: {
        SectorID.GEOMORPHOLOGY: 0.35,
        SectorID.MINERALOGY: 0.10,
        SectorID.SUBSURFACE: 0.20,
        SectorID.THERMAL: 0.20,
        SectorID.ATMOSPHERIC: 0.10,
        SectorID.ASTROBIOLOGY: 0.05,
    },
    MissionProfile.ISRU: {
        SectorID.GEOMORPHOLOGY: 0.15,
        SectorID.MINERALOGY: 0.30,
        SectorID.SUBSURFACE: 0.30,
        SectorID.THERMAL: 0.15,
        SectorID.ATMOSPHERIC: 0.05,
        SectorID.ASTROBIOLOGY: 0.05,
    },
}


@dataclass
class BoundingBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.min_lon + self.max_lon), 0.5 * (self.min_lat + self.max_lat))


@dataclass
class SiteQuery:
    site_id: str
    name: str
    body: PlanetBody
    bbox: BoundingBox
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"

    def cache_key(self, prefix: str) -> str:
        raw = f"{prefix}_{self.body.value}_{self.bbox.to_tuple()}_{self.start_date}_{self.end_date}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{self.site_id}_{digest}"


@dataclass
class SiteDataBundle:
    site: SiteQuery
    pixel_resolution_m: float = 30.0
    dem: Optional[np.ndarray] = None
    rgb_image: Optional[np.ndarray] = None
    s2_multispectral: Optional[np.ndarray] = None
    albedo: Optional[np.ndarray] = None
    mineral_indices: Dict[str, Any] = field(default_factory=dict)
    sar_backscatter: Optional[np.ndarray] = None
    radar_cpr: Optional[np.ndarray] = None
    thermal_day_k: Optional[np.ndarray] = None
    thermal_night_k: Optional[np.ndarray] = None
    atmospheric_scalars: Dict[str, float] = field(default_factory=dict)
    astrobiology_scalars: Dict[str, float] = field(default_factory=dict)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class MetricComparison:
    name: str
    earth_value: Any
    target_value: Any
    unit: str
    similarity: float


@dataclass
class SectorResult:
    sector: SectorID
    name: str
    score: float
    confidence: float
    metrics: List[MetricComparison]
    explanation: str
    weighted_contribution: float = 0.0


@dataclass
class AnalogComparisonReport:
    earth_site: SiteQuery
    target_site: SiteQuery
    mission_profile: MissionProfile
    overall_similarity_score: float
    overall_confidence: float
    sector_results: List[SectorResult]
    key_strengths: List[str]
    key_divergences: List[str]


# ===========================================================================
# 2. DISK CACHE MANAGER & MINERALOGY ASI
# ===========================================================================
class DiskCacheManager:
    """Atomic disk cache for rasters (.npz) and telemetry metadata (.json)."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _npz_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def _json_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def has_arrays(self, key: str) -> bool:
        return self._npz_path(key).is_file()

    def load_arrays(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        p = self._npz_path(key)
        if not p.is_file():
            return None
        try:
            with np.load(p, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}
        except Exception:
            return None

    def save_arrays(self, key: str, **arrays: np.ndarray) -> None:
        p = self._npz_path(key)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(p)

    def has_json(self, key: str) -> bool:
        return self._json_path(key).is_file()

    def load_json(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._json_path(key)
        if not p.is_file():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_json(self, key: str, payload: Dict[str, Any]) -> None:
        p = self._json_path(key)
        tmp = p.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(p)


def compute_mineral_asi(
    earth_profile: Dict[str, float],
    planetary_profile: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute weighted Bray-Curtis Analog Similarity Index (ASI) in [0, 1]."""
    if weights is None:
        weights = {k: 1.0 for k in MINERAL_CLASSES}

    details: List[Dict[str, Any]] = []
    weighted_abs_diff = 0.0
    weighted_sum = 0.0

    for cls in MINERAL_CLASSES:
        e_val = float(earth_profile.get(cls, 0.0))
        p_val = float(planetary_profile.get(cls, 0.0))
        w = max(0.0, float(weights.get(cls, 1.0)))

        diff = abs(e_val - p_val)
        weighted_abs_diff += w * diff
        weighted_sum += w * (e_val + p_val)
        details.append(
            {
                "mineral_class": cls,
                "earth_fraction": round(e_val, 4),
                "planetary_fraction": round(p_val, 4),
                "abs_delta": round(diff, 4),
                "weight": w,
            }
        )

    asi = 0.0 if weighted_sum <= 0.0 else 1.0 - (weighted_abs_diff / weighted_sum)
    return float(np.clip(asi, 0.0, 1.0)), details


# ===========================================================================
# 3. NEURAL WEIGHTS MANAGER, FEATURE EXTRACTOR & LATENT ALIGNER
# ===========================================================================
BEN_V2_S1_MEAN = np.array([-12.619994, -19.290446], dtype=np.float32).reshape(1, 2, 1, 1)
BEN_V2_S1_STD = np.array([5.115912, 5.464428], dtype=np.float32).reshape(1, 2, 1, 1)

BEN_V2_S2_MEAN = np.array(
    [429.94, 614.22, 590.24, 950.68, 1792.46, 2075.47, 2218.95, 2266.46, 1594.43, 1009.33],
    dtype=np.float32,
).reshape(1, 10, 1, 1)
BEN_V2_S2_STD = np.array(
    [572.42, 582.88, 675.89, 729.89, 1096.01, 1273.45, 1365.46, 1356.14, 1079.03, 818.87],
    dtype=np.float32,
).reshape(1, 10, 1, 1)


class ModelWeightsManager:
    """Manages local verification and Hugging Face Hub downloading for ConvNeXtV2 & RemoteCLIP."""

    def __init__(self, base_dir: Union[str, Path] = DEFAULT_WEIGHTS_DIR) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, dict] = {
            "sentinel_1": {
                "repo_id": "BIFOLD-BigEarthNetv2-0/convnextv2_base-s1-v0.2.0",
                "type": "snapshot",
                "target_dir": self.base_dir / "convnextv2_base-s1-v0.2.0",
                "marker_file": "config.json",
            },
            "sentinel_2": {
                "repo_id": "BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0",
                "type": "snapshot",
                "target_dir": self.base_dir / "convnextv2_base-s2-v0.2.0",
                "marker_file": "config.json",
            },
            "mars_remoteclip": {
                "repo_id": "chendelong/RemoteCLIP",
                "type": "file",
                "filename": "RemoteCLIP-ViT-L-14.pt",
                "target_dir": self.base_dir / "RemoteCLIP",
            },
        }

    def is_downloaded(self, model_key: str) -> bool:
        spec = self.registry[model_key]
        target_dir: Path = spec["target_dir"]
        if not target_dir.exists():
            return False
        if spec["type"] == "file":
            file_path = target_dir / spec["filename"]
            return file_path.is_file() and file_path.stat().st_size > 0
        marker_path = target_dir / spec["marker_file"]
        has_weights = any(target_dir.glob("*.safetensors")) or any(target_dir.glob("*.bin"))
        return marker_path.is_file() and has_weights

    def get_model_path(self, model_key: str) -> Path:
        spec = self.registry[model_key]
        target_dir: Path = spec["target_dir"]
        return (target_dir / spec["filename"]) if spec["type"] == "file" else target_dir

    def fetch_model(self, model_key: str, force_download: bool = False, local_files_only: bool = False) -> Path:
        spec = self.registry[model_key]
        target_dir: Path = spec["target_dir"]
        if not force_download and self.is_downloaded(model_key):
            return self.get_model_path(model_key)
        if local_files_only:
            raise FileNotFoundError(f"Model '{model_key}' missing in '{target_dir}'.")
        target_dir.mkdir(parents=True, exist_ok=True)
        if spec["type"] == "file":
            return Path(
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    filename=spec["filename"],
                    local_dir=target_dir,
                    force_download=force_download,
                )
            )
        return Path(
            snapshot_download(
                repo_id=spec["repo_id"],
                local_dir=target_dir,
                force_download=force_download,
            )
        )

    def fetch_all(self, force_download: bool = False, local_files_only: bool = False) -> Dict[str, Path]:
        return {
            key: self.fetch_model(key, force_download=force_download, local_files_only=local_files_only)
            for key in self.registry
        }


class LatentSpaceAligner:
    """Projects Earth Sentinel-1 / Sentinel-2 embeddings (1024D or 2048D) into 768D RemoteCLIP space."""

    def __init__(self, method: str = "ridge", alpha: float = 10.0, n_components: int = 256) -> None:
        self.method = method
        self.input_dim: Optional[int] = None
        self.target_dim: int = 768
        self.model = Ridge(alpha=alpha, fit_intercept=True) if method == "ridge" else CCA(n_components=n_components, max_iter=1000)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.clip(norms, a_min=1e-8, a_max=None)

    def project_earth_to_target_space(self, sentinel_embeddings: np.ndarray) -> np.ndarray:
        if self.input_dim is None:
            raise RuntimeError("LatentSpaceAligner must be fitted or loaded before projection.")
        x = self._l2_normalize(sentinel_embeddings)
        if self.method == "ridge":
            projected = self.model.predict(x)
        else:
            projected, _ = self.model.transform(x, np.zeros((x.shape[0], self.target_dim)))
        return self._l2_normalize(projected)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LatentSpaceAligner":
        return joblib.load(Path(path))


class MultiModalFeatureExtractor:
    """Lazy-loading inference engine for Sentinel-1 SAR, Sentinel-2 Multispectral, and RemoteCLIP ViT-L-14."""

    def __init__(self, model_paths: Optional[Dict[str, Path]] = None, device: Optional[str] = None, auto_fetch: bool = False) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        if model_paths is None:
            manager = ModelWeightsManager()
            self.model_paths = manager.fetch_all() if auto_fetch else {k: manager.get_model_path(k) for k in manager.registry}
        else:
            self.model_paths = {k: Path(v) for k, v in model_paths.items()}
        self._models: Dict[str, torch.nn.Module] = {}
        self._backbones: Dict[str, torch.nn.Module] = {}
        self._clip_preprocess = None

    @staticmethod
    def _coerce_enum_value(field_type: Any, raw_value: Any) -> Any:
        if raw_value is None:
            return None
        origin = get_origin(field_type)
        candidate_types = get_args(field_type) if origin is not None else (field_type,)
        for candidate in candidate_types:
            if isinstance(candidate, type) and issubclass(candidate, Enum):
                if isinstance(raw_value, candidate):
                    return raw_value
                try:
                    return candidate(int(raw_value))
                except (ValueError, TypeError):
                    return candidate(raw_value)
        return raw_value

    @classmethod
    def _build_ilm_config(cls, config_path: Path) -> ILMConfiguration:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)
        normalized_dict = {k.lstrip("_"): v for k, v in raw_dict.items()}
        sig = inspect.signature(ILMConfiguration.__init__)
        valid_params = {p for p in sig.parameters if p != "self"}
        type_hints = get_type_hints(ILMConfiguration)
        filtered_kwargs: Dict[str, Any] = {}
        for key, value in normalized_dict.items():
            if key in valid_params:
                field_type = type_hints.get(key)
                filtered_kwargs[key] = cls._coerce_enum_value(field_type, value) if field_type is not None else value
        if dataclasses.is_dataclass(ILMConfiguration):
            for dc_field in dataclasses.fields(ILMConfiguration):
                if dc_field.name in filtered_kwargs:
                    filtered_kwargs[dc_field.name] = cls._coerce_enum_value(dc_field.type, filtered_kwargs[dc_field.name])
        return ILMConfiguration(**filtered_kwargs)

    @staticmethod
    def _load_local_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
        safetensors_files = sorted(model_dir.glob("*.safetensors"))
        if safetensors_files:
            from safetensors.torch import load_file
            return load_file(str(safetensors_files[0]))
        bin_files = sorted(model_dir.glob("*.bin"))
        if bin_files:
            return torch.load(bin_files[0], map_location="cpu", weights_only=True)
        raise FileNotFoundError(f"No .safetensors or .bin weights found in '{model_dir}'.")

    def _load_sentinel_model(self, key: str) -> torch.nn.Module:
        if key not in self._models:
            path = self.model_paths[key]
            model: Optional[torch.nn.Module] = None
            if hasattr(ILMModel, "from_pretrained"):
                try:
                    model = ILMModel.from_pretrained(str(path))
                except Exception:
                    model = None
            if model is None:
                config = self._build_ilm_config(path / "config.json")
                model = ILMModel(config)
                state_dict = self._load_local_state_dict(path)
                model_keys = set(model.state_dict().keys())
                if state_dict and not (set(state_dict.keys()) & model_keys):
                    stripped = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
                    if set(stripped.keys()) & model_keys:
                        state_dict = stripped
                model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self._models[key] = model
        return self._models[key]

    def _resolve_timm_backbone(self, key: str) -> torch.nn.Module:
        if key not in self._backbones:
            model = self._load_sentinel_model(key)
            for module in model.modules():
                if module is not model and hasattr(module, "forward_features") and hasattr(module, "forward_head"):
                    self._backbones[key] = module
                    break
            else:
                self._backbones[key] = model
        return self._backbones[key]

    def _load_remoteclip_model(self) -> torch.nn.Module:
        key = "mars_remoteclip"
        if key not in self._models:
            ckpt_path = self.model_paths[key]
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._models[key] = model
            self._clip_preprocess = preprocess
        return self._models[key]

    @staticmethod
    def _prepare_multiband_tensor(data: Union[np.ndarray, torch.Tensor], expected_channels: int, target_size: int = 120) -> torch.Tensor:
        tensor = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-2:] != (target_size, target_size):
            tensor = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return tensor

    def _forward_sentinel_backbone(self, key: str, x: torch.Tensor, l2_normalize: bool = True) -> np.ndarray:
        backbone = self._resolve_timm_backbone(key)
        x = x.to(self.device)
        if hasattr(backbone, "forward_features") and hasattr(backbone, "forward_head"):
            features = backbone.forward_features(x)
            embeddings = backbone.forward_head(features, pre_logits=True)
        elif hasattr(backbone, "forward_features"):
            features = backbone.forward_features(x)
            embeddings = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        else:
            embeddings = backbone(x)
            if embeddings.ndim > 2:
                embeddings = embeddings.flatten(1)
        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().numpy()

    @torch.inference_mode()
    def extract_sentinel1(self, sar_patch: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        x = self._prepare_multiband_tensor(sar_patch, expected_channels=2)
        x = (x - torch.from_numpy(BEN_V2_S1_MEAN)) / torch.from_numpy(BEN_V2_S1_STD)
        return self._forward_sentinel_backbone("sentinel_1", x)

    @torch.inference_mode()
    def extract_sentinel2(self, ms_patch: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        x = self._prepare_multiband_tensor(ms_patch, expected_channels=10)
        x = (x - torch.from_numpy(BEN_V2_S2_MEAN)) / torch.from_numpy(BEN_V2_S2_STD)
        return self._forward_sentinel_backbone("sentinel_2", x)

    @torch.inference_mode()
    def extract_optical_clip(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        model = self._load_remoteclip_model()
        if isinstance(image, np.ndarray):
            arr = image
            if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
                arr = np.clip(arr * 255.0, 0.0, 255.0)
            arr_u8 = arr.astype(np.uint8)
            pil_img = Image.fromarray(arr_u8).convert("RGB") if arr_u8.ndim == 2 else Image.fromarray(arr_u8[..., :3]).convert("RGB")
        else:
            pil_img = image.convert("RGB")
        tensor = self._clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        embeddings = model.encode_image(tensor)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().numpy()


# ===========================================================================
# 4. LIVE SATELLITE & PLANETARY DATA INGESTORS
# ===========================================================================
S2_BANDS_10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
_POWER_FLUX_TO_WM2: Dict[str, float] = {
    "W/m^2": 1.0,
    "W/m2": 1.0,
    "MJ/m^2/day": 1_000_000.0 / 86_400.0,
    "MJ/m2/day": 1_000_000.0 / 86_400.0,
    "kWh/m^2/day": 1_000.0 / 24.0,
    "kWh/m2/day": 1_000.0 / 24.0,
}


def _utm_crs_from_bbox(bbox: BoundingBox) -> CRS:
    center_lon, center_lat = bbox.center()
    zone = int(math.floor((center_lon + 180.0) / 6.0)) + 1
    epsg = (32600 + zone) if center_lat >= 0.0 else (32700 + zone)
    return CRS.from_epsg(epsg)


def lee_speckle_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0.0)
    if not np.any(valid):
        return arr.astype(np.float32)
    fill_val = float(np.nanmedian(arr[valid]))
    filled = np.where(valid, arr, fill_val)
    mean = uniform_filter(filled, size=window_size)
    sq_mean = uniform_filter(filled ** 2, size=window_size)
    variance = np.maximum(sq_mean - mean ** 2, 0.0)
    overall_variance = float(np.var(filled[valid]))
    weights = variance / (variance + max(overall_variance, 1e-12))
    filtered = mean + weights * (filled - mean)
    return np.where(valid, filtered, np.nan).astype(np.float32)


def fetch_sentinel1_rtc_raster(
    site: SiteQuery,
    cache: Optional[DiskCacheManager] = None,
    grid_size: int = 120,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    cache_mgr = cache if cache is not None else DiskCacheManager()
    cache_key = site.cache_key(f"s1_rtc_{grid_size}")
    if cache_mgr.has_arrays(cache_key) and cache_mgr.has_json(cache_key):
        cached_arrs = cache_mgr.load_arrays(cache_key)
        cached_meta = cache_mgr.load_json(cache_key)
        if cached_arrs is not None and cached_meta is not None:
            return cached_arrs.get("sar_backscatter"), cached_arrs.get("radar_cpr"), cached_meta

    bbox_tuple = site.bbox.to_tuple()
    dst_crs = _utm_crs_from_bbox(site.bbox)
    utm_bounds = transform_bounds("EPSG:4326", dst_crs, *bbox_tuple)
    dst_transform = from_bounds(*utm_bounds, width=grid_size, height=grid_size)

    try:
        client = pystac_client.Client.open(PLANETARY_COMPUTER_STAC_URL, modifier=planetary_computer.sign_inplace)
        search = client.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox_tuple,
            datetime=f"{site.start_date}/{site.end_date}",
            max_items=5,
        )
        items = [it for it in search.items() if "vv" in it.assets and "vh" in it.assets]
        if not items:
            return None, None, {}

        top_item = items[0]

        def _read_pol(asset_key: str) -> np.ndarray:
            dst = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
            with rasterio.open(top_item.assets[asset_key].href) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
            return lee_speckle_filter(dst, window_size=5)

        vv_lin = _read_pol("vv")
        vh_lin = _read_pol("vh")
        valid_mask = np.isfinite(vv_lin) & (vv_lin > 1e-6) & np.isfinite(vh_lin) & (vh_lin > 1e-6)
        if not np.any(valid_mask):
            return None, None, {}

        vv_db = np.where(valid_mask, 10.0 * np.log10(np.maximum(vv_lin, 1e-6)), np.nan).astype(np.float32)
        vh_db = np.where(valid_mask, 10.0 * np.log10(np.maximum(vh_lin, 1e-6)), np.nan).astype(np.float32)
        cpr = np.where(valid_mask, np.clip(vh_lin / np.maximum(vv_lin, 1e-6), 0.0, 2.0), np.nan).astype(np.float32)

        vv_clean = np.where(np.isfinite(vv_db), vv_db, float(np.nanmedian(vv_db)))
        vh_clean = np.where(np.isfinite(vh_db), vh_db, float(np.nanmedian(vh_db)))
        sar_stack = np.stack([vv_clean, vh_clean], axis=0).astype(np.float32)

        telemetry = {
            "stac_item_id": top_item.id,
            "sar_vv_db": round(float(np.nanmean(vv_db)), 3),
            "sar_vh_db": round(float(np.nanmean(vh_db)), 3),
            "cpr_linear": round(float(np.nanmean(cpr)), 4),
        }
        cache_mgr.save_arrays(cache_key, sar_backscatter=sar_stack, radar_cpr=cpr)
        cache_mgr.save_json(cache_key, telemetry)
        return sar_stack, cpr, telemetry
    except Exception as exc:
        logger.debug("S1 RTC failed for %s: %s", site.site_id, exc)
        return None, None, {}


class PlanetaryDataOrchestrator:
    """Streams and caches co-registered Earth STAC rasters onto a uniform UTM grid."""

    def __init__(self, cache: Optional[DiskCacheManager] = None, grid_size: int = 120) -> None:
        self.cache = cache if cache is not None else DiskCacheManager()
        self.grid_size = grid_size

    def _reproject_asset(self, href: str, dst_transform: Any, dst_crs: CRS, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
        dst = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)
        with rasterio.open(href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
        return dst

    def fetch_earth_stac_rasters(self, site: SiteQuery) -> Dict[str, Any]:
        cache_key = site.cache_key(f"earth_stac_{self.grid_size}")
        if self.cache.has_arrays(cache_key):
            cached = self.cache.load_arrays(cache_key)
            if cached is not None:
                return dict(cached)

        bbox_tuple = site.bbox.to_tuple()
        dst_crs = _utm_crs_from_bbox(site.bbox)
        utm_bounds = transform_bounds("EPSG:4326", dst_crs, *bbox_tuple)
        dst_transform = from_bounds(*utm_bounds, width=self.grid_size, height=self.grid_size)
        dt_range = f"{site.start_date}/{site.end_date}"
        out: Dict[str, np.ndarray] = {}

        try:
            client = pystac_client.Client.open(PLANETARY_COMPUTER_STAC_URL, modifier=planetary_computer.sign_inplace)

            dem_items = list(client.search(collections=["cop-dem-glo-30"], bbox=bbox_tuple, max_items=1).items())
            if dem_items and "data" in dem_items[0].assets:
                dem = self._reproject_asset(dem_items[0].assets["data"].href, dst_transform, dst_crs)
                if np.any(np.isfinite(dem)):
                    med = float(np.nanmedian(dem))
                    out["dem"] = np.where(np.isfinite(dem), dem, med).astype(np.float32)

            s2_search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox_tuple,
                datetime=dt_range,
                query={"eo:cloud_cover": {"lt": 25}},
                max_items=5,
            )
            s2_items = sorted(s2_search.items(), key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)))
            if s2_items and all(b in s2_items[0].assets for b in S2_BANDS_10):
                top_s2 = s2_items[0]
                bands = []
                for b in S2_BANDS_10:
                    arr = self._reproject_asset(top_s2.assets[b].href, dst_transform, dst_crs)
                    med = float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else 0.0
                    bands.append(np.where(np.isfinite(arr), arr, med))
                s2_stack = np.stack(bands, axis=0).astype(np.float32)
                out["s2_multispectral"] = s2_stack

                rgb_f = np.stack([s2_stack[2], s2_stack[1], s2_stack[0]], axis=-1) / 3000.0
                out["rgb_image"] = np.clip(rgb_f * 255.0, 0.0, 255.0).astype(np.uint8)

                vis_swir = (
                    0.356 * s2_stack[0]
                    + 0.130 * s2_stack[2]
                    + 0.373 * s2_stack[6]
                    + 0.085 * s2_stack[8]
                    + 0.072 * s2_stack[9]
                ) / 10000.0
                out["albedo"] = np.clip(vis_swir, 0.01, 0.95).astype(np.float32)

            ls_search = client.search(
                collections=["landsat-c2-l2"],
                bbox=bbox_tuple,
                datetime=dt_range,
                query={"eo:cloud_cover": {"lt": 25}},
                max_items=5,
            )
            ls_items = sorted(ls_search.items(), key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)))
            if ls_items and "lwir11" in ls_items[0].assets:
                dn = self._reproject_asset(ls_items[0].assets["lwir11"].href, dst_transform, dst_crs, resampling=Resampling.nearest)
                valid_dn = np.isfinite(dn) & (dn > 0.0)
                if np.any(valid_dn):
                    temp_k = np.where(valid_dn, dn * 0.00341802 + 149.0, np.nan).astype(np.float32)
                    med_k = float(np.nanmedian(temp_k))
                    out["thermal_day_k"] = np.where(np.isfinite(temp_k), temp_k, med_k).astype(np.float32)

            if out:
                self.cache.save_arrays(cache_key, **out)
            return out
        except Exception as exc:
            logger.debug("Earth STAC failed for %s: %s", site.site_id, exc)
            return out


def _compute_unesco_aridity_index(t_mean_c: float, t_max_c: float, t_min_c: float, precip_mm_day: float, sw_kwh_m2_day: float) -> float:
    td = max(0.5, t_max_c - t_min_c)
    ra_mm_day = max(0.5, sw_kwh_m2_day * 1.4688)
    pet_mm_day = max(0.1, 0.0023 * (t_mean_c + 17.8) * math.sqrt(td) * ra_mm_day)
    return max(0.0001, precip_mm_day / pet_mm_day)


def _convert_power_flux_to_wm2(value: Optional[float], unit_str: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if unit_str and unit_str.strip() in _POWER_FLUX_TO_WM2:
        return value * _POWER_FLUX_TO_WM2[unit_str.strip()]
    return value * _POWER_FLUX_TO_WM2["MJ/m^2/day"]


async def fetch_soilgrids_properties(lat: float, lon: float, client: httpx.AsyncClient) -> Dict[str, Any]:
    params = [
        ("lon", round(lon, 4)),
        ("lat", round(lat, 4)),
        ("property", "phh2o"),
        ("property", "soc"),
        ("depth", "0-5cm"),
        ("value", "mean"),
    ]
    try:
        resp = await client.get(SOILGRIDS_QUERY_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
        layers = payload.get("properties", {}).get("layers", [])
        extracted: Dict[str, Any] = {"ph": None, "soc_g_kg": None}
        for layer in layers:
            name = layer.get("name")
            depths = layer.get("depths", [])
            if not depths:
                continue
            mean_val = depths[0].get("values", {}).get("mean")
            if mean_val is None:
                continue
            if name == "phh2o":
                extracted["ph"] = round(float(mean_val) / 10.0, 2)
            elif name == "soc":
                extracted["soc_g_kg"] = round(float(mean_val) / 10.0, 2)
        return extracted
    except Exception:
        return {"ph": None, "soc_g_kg": None}


async def fetch_earth_climate_and_atmosphere(
    lat: float,
    lon: float,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    cache: Optional[DiskCacheManager] = None,
) -> Dict[str, Any]:
    cache_mgr = cache if cache is not None else DiskCacheManager()
    c_key = f"power_soil_{round(lat, 3)}_{round(lon, 3)}_{start_date}_{end_date}"
    if cache_mgr.has_json(c_key):
        cached = cache_mgr.load_json(c_key)
        if cached is not None:
            return cached

    power_params = {
        "parameters": "T2M,T2M_MAX,T2M_MIN,TS,PS,WS10M,RH2M,GWETTOP,PRECTOTCORR,AOD_55,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_UVA,ALLSKY_SFC_UVB",
        "community": "RE",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            power_resp, soil_props = await asyncio.gather(
                client.get(NASA_POWER_BASE_URL, params=power_params),
                fetch_soilgrids_properties(lat, lon, client),
            )
        power_resp.raise_for_status()
        data = power_resp.json()
        param_dict = data.get("properties", {}).get("parameter", {})
        param_meta = data.get("header", {}).get("parameter", {})

        def _mean_val(key: str) -> Optional[float]:
            series = param_dict.get(key, {})
            vals = [float(v) for v in series.values() if v is not None and float(v) > -900.0]
            return float(sum(vals) / len(vals)) if vals else None

        def _unit_for(key: str) -> Optional[str]:
            meta = param_meta.get(key)
            return meta.get("units") if isinstance(meta, dict) else None

        t_mean_c = _mean_val("T2M")
        t_max_c = _mean_val("T2M_MAX")
        t_min_c = _mean_val("T2M_MIN")
        if t_mean_c is None or t_max_c is None or t_min_c is None:
            return {}

        ps_kpa = _mean_val("PS")
        ws_ms = _mean_val("WS10M")
        rh_pct = _mean_val("RH2M")
        gwettop = _mean_val("GWETTOP")
        precip_mm_day = _mean_val("PRECTOTCORR")
        aod_55 = _mean_val("AOD_55")
        sw_wm2 = _convert_power_flux_to_wm2(_mean_val("ALLSKY_SFC_SW_DWN"), _unit_for("ALLSKY_SFC_SW_DWN"))
        uva_wm2 = _convert_power_flux_to_wm2(_mean_val("ALLSKY_SFC_UVA"), _unit_for("ALLSKY_SFC_UVA"))
        uvb_wm2 = _convert_power_flux_to_wm2(_mean_val("ALLSKY_SFC_UVB"), _unit_for("ALLSKY_SFC_UVB"))

        t_mean_k = t_mean_c + 273.15
        t_max_k = max(t_mean_k, t_max_c + 273.15)
        t_min_k = min(t_mean_k, t_min_c + 273.15)
        sw_kwh = (sw_wm2 * 24.0 / 1000.0) if sw_wm2 is not None else 5.0
        aridity_index = _compute_unesco_aridity_index(t_mean_c, t_max_c, t_min_c, precip_mm_day or 0.0, sw_kwh)

        if uva_wm2 is not None and uvb_wm2 is not None:
            uv_flux_wm2 = uva_wm2 + uvb_wm2
        elif sw_wm2 is not None:
            uv_flux_wm2 = sw_wm2 * 0.045
        else:
            uv_flux_wm2 = None

        soil_wetness = float(min(1.0, max(0.0, gwettop))) if gwettop is not None else None
        if soil_wetness is not None:
            water_activity = soil_wetness
        elif rh_pct is not None:
            water_activity = float(min(1.0, max(0.0, (rh_pct / 100.0) * 0.5)))
        else:
            water_activity = None

        cold_factor = min(1.0, max(0.0, (305.0 - t_mean_k) / 50.0))
        arid_factor = min(1.0, max(0.0, 1.0 - (water_activity if water_activity is not None else 0.3)))
        preservation_potential = round(0.5 * cold_factor + 0.5 * arid_factor, 4)

        result = {
            "thermal": {
                "t_max_k": round(t_max_k, 2),
                "t_min_k": round(t_min_k, 2),
            },
            "atmospheric": {
                "surface_pressure_pa": round(ps_kpa * 1000.0, 1) if ps_kpa is not None else None,
                "mean_wind_speed_ms": round(ws_ms, 2) if ws_ms is not None else None,
                "aridity_index": round(aridity_index, 5),
                "uv_flux_wm2": round(uv_flux_wm2, 2) if uv_flux_wm2 is not None else None,
                "dust_optical_depth": round(aod_55, 4) if aod_55 is not None else None,
            },
            "astrobiology": {
                "soil_water_activity_aw": round(water_activity, 4) if water_activity is not None else None,
                "organic_preservation_potential": preservation_potential,
                "ph": soil_props["ph"],
                "soil_organic_carbon_g_kg": soil_props["soc_g_kg"],
            },
        }
        cache_mgr.save_json(c_key, result)
        return result
    except Exception:
        return {}


_EMIT_AUTHENTICATED = False


def _login_earthdata_once() -> bool:
    global _EMIT_AUTHENTICATED
    if earthaccess is None or xr is None:
        return False
    if _EMIT_AUTHENTICATED or getattr(earthaccess.__auth__, "authenticated", False):
        _EMIT_AUTHENTICATED = True
        return True
    try:
        earthaccess.login(strategy="environment")
        _EMIT_AUTHENTICATED = True
        return True
    except Exception:
        try:
            earthaccess.login(strategy="netrc")
            _EMIT_AUTHENTICATED = True
            return True
        except Exception:
            return False


def extract_emit_mineralogy(lat: float, lon: float, radius_m: float = 2000.0, max_granules: int = 8) -> Dict[str, Any]:
    """Query NASA EMITL2BMIN for 60m hyperspectral mineralogy inside +/-54 deg latitude."""
    if abs(lat) > 54.0 or not _login_earthdata_once():
        return {}

    try:
        results = earthaccess.search_data(
            short_name="EMITL2BMIN",
            version="001",
            circle=(lon, lat, radius_m),
            temporal=("2022-08-09", "2026-08-31"),
            count=max_granules,
        )
        if not results:
            return {}

        geod = Geod(ellps="WGS84")
        for granule in results:
            opened_files = earthaccess.open([granule])
            if not opened_files:
                continue
            fobj = opened_files[0]
            try:
                with (
                    xr.open_dataset(fobj, engine="h5netcdf") as ds_root,
                    xr.open_dataset(fobj, group="mineral_metadata", engine="h5netcdf") as ds_meta,
                    xr.open_dataset(fobj, group="location", engine="h5netcdf") as ds_loc,
                ):
                    lat_grid = ds_loc["lat"].values
                    lon_grid = ds_loc["lon"].values
                    valid_coords = np.isfinite(lat_grid) & np.isfinite(lon_grid)
                    dist_m = np.full(lat_grid.shape, np.inf, dtype=np.float64)
                    _, _, dist_m[valid_coords] = geod.inv(
                        np.full(int(np.sum(valid_coords)), lon, dtype=np.float64),
                        np.full(int(np.sum(valid_coords)), lat, dtype=np.float64),
                        lon_grid[valid_coords].astype(np.float64),
                        lat_grid[valid_coords].astype(np.float64),
                    )
                    footprint_mask = dist_m <= radius_m
                    if int(np.sum(footprint_mask)) == 0:
                        continue

                    group1_id = ds_root["group_1_mineral_id"].values[footprint_mask]
                    band_depth = ds_root["group_1_band_depth"].values[footprint_mask]
                    if "group_2_mineral_id" in ds_root:
                        g2_id = ds_root["group_2_mineral_id"].values[footprint_mask]
                        g2_bd = ds_root["group_2_band_depth"].values[footprint_mask]
                        use_g2 = (group1_id <= 0) & (g2_id > 0)
                        group1_id = np.where(use_g2, g2_id, group1_id)
                        band_depth = np.where(use_g2, g2_bd, band_depth)

                    valid_px = (group1_id > 0) & np.isfinite(band_depth) & (band_depth > 0.01)
                    if int(np.sum(valid_px)) < 15:
                        continue

                    mineral_names = [
                        n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in ds_meta["name"].values
                    ]
                    id_to_name = {i: name for i, name in enumerate(mineral_names)}
                    class_counts: Dict[str, float] = {c: 0.0 for c in MINERAL_CLASSES}

                    for m_id in group1_id[valid_px]:
                        m_lower = id_to_name.get(int(m_id), "Unknown").lower()
                        for kw, harm_class in EMIT_KEYWORD_MAP.items():
                            if kw in m_lower:
                                class_counts[harm_class] += 1.0
                                break

                    total_classified = sum(class_counts.values())
                    if total_classified <= 0:
                        continue
                    return {
                        "harmonized_profile": {k: v / total_classified for k, v in class_counts.items()},
                    }
            finally:
                if hasattr(fobj, "close"):
                    fobj.close()
    except Exception:
        return {}
    return {}


# NASA PDS ODE Mars & Moon Ingestion
_PDS3_KV_PATTERN = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.+?)\s*$", re.MULTILINE)


def _parse_pds3_label(label_text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for key, raw_val in _PDS3_KV_PATTERN.findall(label_text):
        cleaned = re.sub(r"<[^>]+>", "", raw_val).strip().strip('"').strip("'")
        parsed[key] = cleaned
    return parsed


def _parse_float_field(meta: Dict[str, str], candidate_keys: Tuple[str, ...]) -> Optional[float]:
    for k in candidate_keys:
        if k in meta:
            try:
                val = float(meta[k])
                if np.isfinite(val) and abs(val) < 1e15:
                    return val
            except ValueError:
                continue
    return None


async def query_pds_ode_products(
    target: str,
    instrument: str,
    product_type: str,
    lat: float,
    lon: float,
    radius_deg: float = 0.25,
    limit: int = 3,
    ihid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    lon_360 = lon % 360.0
    if ihid is None:
        if target == "mars":
            ihid = "MRO" if instrument.upper() in ("CRISM", "HIRISE", "MCS", "SHARAD") else "ODY"
        else:
            ihid = "CH1-ORB" if instrument.upper() == "M3" else "LRO"

    params = {
        "query": "product",
        "results": "m",
        "output": "JSON",
        "target": target,
        "ihid": ihid,
        "iid": instrument,
        "pt": product_type,
        "minlat": max(-90.0, lat - radius_deg),
        "maxlat": min(90.0, lat + radius_deg),
        "westernlon": (lon_360 - radius_deg) % 360.0,
        "easternlon": (lon_360 + radius_deg) % 360.0,
        "limit": limit,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(PDS_ODE_BASE_URL, params=params)
            resp.raise_for_status()
        data = resp.json()
        products_container = data.get("ODEResults", {}).get("Products")
        if isinstance(products_container, dict):
            products = products_container.get("Product", [])
            return [products] if isinstance(products, dict) else list(products)
    except Exception:
        pass
    return []


async def _fetch_label_and_browse(client: httpx.AsyncClient, product: Dict[str, Any], target_size: int = 224) -> Tuple[Dict[str, str], Optional[np.ndarray]]:
    label_meta: Dict[str, str] = {}
    browse_rgb: Optional[np.ndarray] = None
    if product.get("LabelURL"):
        try:
            r = await client.get(product["LabelURL"])
            if r.status_code == 200:
                label_meta = _parse_pds3_label(r.text)
        except Exception:
            pass
    if product.get("BrowseURL"):
        try:
            r = await client.get(product["BrowseURL"])
            if r.status_code == 200:
                pil_img = Image.open(io.BytesIO(r.content)).convert("RGB").resize((target_size, target_size), Image.Resampling.BILINEAR)
                browse_rgb = np.asarray(pil_img, dtype=np.uint8)
        except Exception:
            pass
    return label_meta, browse_rgb


async def fetch_mars_orbital_telemetry(lat: float, lon: float) -> Dict[str, Any]:
    crism_prods, hirise_prods, themis_prods, mcs_prods = await asyncio.gather(
        query_pds_ode_products("mars", "CRISM", "MTRDR", lat, lon, radius_deg=0.25, limit=3),
        query_pds_ode_products("mars", "HIRISE", "RDRV11", lat, lon, radius_deg=0.20, limit=2),
        query_pds_ode_products("mars", "THEMIS", "IRGEO2", lat, lon, radius_deg=0.35, limit=3, ihid="ODY"),
        query_pds_ode_products("mars", "MCS", "DDR", lat, lon, radius_deg=0.50, limit=2),
    )
    if not crism_prods:
        crism_prods = await query_pds_ode_products("mars", "CRISM", "MRDR", lat, lon, radius_deg=0.50, limit=3)

    result: Dict[str, Any] = {"rgb_image": None, "harmonized_profile": None, "thermal": {}, "atmospheric": {}}
    async with httpx.AsyncClient(timeout=15.0) as client:
        if crism_prods:
            lbl_meta, crism_rgb = await _fetch_label_and_browse(client, crism_prods[0])
            inc_angle = _parse_float_field(lbl_meta, ("INCIDENCE_ANGLE", "SOLAR_INCIDENCE_ANGLE"))
            if crism_rgb is not None:
                rgb = np.asarray(crism_rgb, dtype=np.float64) / 255.0
                if inc_angle is not None and 0.0 <= inc_angle < 85.0:
                    rgb = np.clip(rgb / max(0.2, float(np.cos(np.radians(inc_angle)))), 0.0, 1.0)
                r_m, g_m, b_m = float(np.mean(rgb[..., 0])), float(np.mean(rgb[..., 1])), float(np.mean(rgb[..., 2]))
                raw = {
                    "mafic_olivine_pyroxene": max(0.01, r_m * 0.55),
                    "plagioclase_anorthosite": max(0.01, (1.0 - max(r_m, g_m, b_m)) * 0.25),
                    "phyllosilicates_clays": max(0.01, g_m * 0.85),
                    "sulfates_evaporites": max(0.01, b_m * 0.80),
                    "carbonates": max(0.01, min(g_m, b_m) * 0.30),
                    "iron_oxides_hydroxides": max(0.01, r_m * 0.45),
                }
                tot = sum(raw.values())
                result["harmonized_profile"] = {k: round(v / tot, 4) for k, v in raw.items()}
                result["rgb_image"] = crism_rgb
        if hirise_prods:
            _, hirise_rgb = await _fetch_label_and_browse(client, hirise_prods[0])
            if hirise_rgb is not None:
                result["rgb_image"] = hirise_rgb
        if themis_prods:
            th_meta, _ = await _fetch_label_and_browse(client, themis_prods[0])
            t_max = _parse_float_field(th_meta, ("MAXIMUM_BRIGHTNESS_TEMPERATURE", "MAX_TEMPERATURE"))
            t_min = _parse_float_field(th_meta, ("MINIMUM_BRIGHTNESS_TEMPERATURE", "MIN_TEMPERATURE"))
            if t_max is not None:
                result["thermal"]["t_max_k"] = t_max
            if t_min is not None:
                result["thermal"]["t_min_k"] = t_min
        if mcs_prods:
            mcs_meta, _ = await _fetch_label_and_browse(client, mcs_prods[0])
            surf_p = _parse_float_field(mcs_meta, ("SURFACE_PRESSURE", "PRESSURE"))
            dust_tau = _parse_float_field(mcs_meta, ("DUST_OPTICAL_DEPTH", "COLUMN_DUST_OPTICAL_DEPTH"))
            if surf_p is not None:
                result["atmospheric"]["surface_pressure_pa"] = surf_p
            if dust_tau is not None:
                result["atmospheric"]["dust_optical_depth"] = dust_tau
    return result


async def fetch_moon_orbital_telemetry(lat: float, lon: float) -> Dict[str, Any]:
    _, lroc_prods, diviner_prods, minirf_prods = await asyncio.gather(
        query_pds_ode_products("moon", "M3", "REFIMG", lat, lon, radius_deg=0.50, limit=2),
        query_pds_ode_products("moon", "LROC", "CDR", lat, lon, radius_deg=0.35, limit=2, ihid="LRO"),
        query_pds_ode_products("moon", "DLRE", "GDR", lat, lon, radius_deg=0.50, limit=2, ihid="LRO"),
        query_pds_ode_products("moon", "MRF", "RDR", lat, lon, radius_deg=0.50, limit=2, ihid="LRO"),
    )
    result: Dict[str, Any] = {"rgb_image": None, "harmonized_profile": None, "albedo": None, "thermal": {}, "cpr_mean": None}
    async with httpx.AsyncClient(timeout=15.0) as client:
        if lroc_prods:
            lroc_meta, lroc_rgb = await _fetch_label_and_browse(client, lroc_prods[0])
            if lroc_rgb is not None:
                result["rgb_image"] = lroc_rgb
                inc = _parse_float_field(lroc_meta, ("INCIDENCE_ANGLE",))
                cos_i = max(0.2, float(np.cos(np.radians(inc)))) if inc is not None and inc < 85.0 else 0.7
                est_albedo = float(np.clip((np.mean(lroc_rgb) / 255.0) * 0.25 / cos_i, 0.04, 0.35))
                result["albedo"] = round(est_albedo, 4)
                mare_w = float(np.clip((0.18 - est_albedo) / 0.11, 0.05, 0.90))
                result["harmonized_profile"] = {
                    "mafic_olivine_pyroxene": round(0.12 + 0.48 * mare_w, 4),
                    "plagioclase_anorthosite": round(0.83 - 0.55 * mare_w, 4),
                    "phyllosilicates_clays": 0.0,
                    "sulfates_evaporites": 0.0,
                    "carbonates": 0.0,
                    "iron_oxides_hydroxides": round(0.05 + 0.07 * mare_w, 4),
                }
        if diviner_prods:
            div_meta, _ = await _fetch_label_and_browse(client, diviner_prods[0])
            t_max = _parse_float_field(div_meta, ("MAXIMUM_TEMPERATURE", "MAX_BRIGHTNESS_TEMPERATURE"))
            t_min = _parse_float_field(div_meta, ("MINIMUM_TEMPERATURE", "MIN_BRIGHTNESS_TEMPERATURE"))
            if t_max is not None:
                result["thermal"]["t_max_k"] = t_max
            if t_min is not None:
                result["thermal"]["t_min_k"] = t_min
        if minirf_prods:
            mrf_meta, mrf_rgb = await _fetch_label_and_browse(client, minirf_prods[0])
            cpr_val = _parse_float_field(mrf_meta, ("CIRCULAR_POLARIZATION_RATIO", "MEAN_CPR"))
            if cpr_val is not None:
                result["cpr_mean"] = cpr_val
            elif mrf_rgb is not None:
                result["cpr_mean"] = round(float(np.mean(mrf_rgb[..., 1]) / 255.0) * 1.2, 4)
    return result


# ===========================================================================
# 5. UNIFIED PLANETARY DATA PROVIDER
# ===========================================================================
class UnifiedPlanetaryDataProvider:
    def __init__(self, cache: Optional[DiskCacheManager] = None, grid_size: int = 120, aligner: Optional[LatentSpaceAligner] = None) -> None:
        self.cache = cache if cache is not None else DiskCacheManager()
        self.grid_size = grid_size
        self.stac_orchestrator = PlanetaryDataOrchestrator(cache=self.cache, grid_size=grid_size)
        self.weights_manager = ModelWeightsManager()
        self.feature_extractor = MultiModalFeatureExtractor(auto_fetch=False)
        self.aligner = aligner

    def _populate_available_embeddings(self, bundle: SiteDataBundle) -> None:
        try:
            if bundle.rgb_image is not None and self.weights_manager.is_downloaded("mars_remoteclip"):
                bundle.embeddings["clip_768"] = self.feature_extractor.extract_optical_clip(bundle.rgb_image)[0]

            if bundle.site.body == PlanetBody.EARTH:
                if bundle.sar_backscatter is not None and self.weights_manager.is_downloaded("sentinel_1"):
                    bundle.embeddings["s1_1024"] = self.feature_extractor.extract_sentinel1(bundle.sar_backscatter)[0]
                if bundle.s2_multispectral is not None and self.weights_manager.is_downloaded("sentinel_2"):
                    bundle.embeddings["s2_1024"] = self.feature_extractor.extract_sentinel2(bundle.s2_multispectral)[0]
                if self.aligner is not None and self.aligner.input_dim is not None:
                    if self.aligner.input_dim == 2048 and "s1_1024" in bundle.embeddings and "s2_1024" in bundle.embeddings:
                        fused = np.concatenate([bundle.embeddings["s1_1024"], bundle.embeddings["s2_1024"]])
                        bundle.embeddings["earth_aligned_768"] = self.aligner.project_earth_to_target_space(fused)[0]
                    elif self.aligner.input_dim == 1024 and "s2_1024" in bundle.embeddings:
                        bundle.embeddings["earth_aligned_768"] = self.aligner.project_earth_to_target_space(bundle.embeddings["s2_1024"])[0]
        except Exception as exc:
            logger.debug("Embedding extraction skipped for %s: %s", bundle.site.site_id, exc)

    async def load_bundle_async(self, site: SiteQuery) -> SiteDataBundle:
        center_lon, center_lat = site.bbox.center()

        if site.body == PlanetBody.EARTH:
            bundle = SiteDataBundle(site=site, pixel_resolution_m=30.0)
            stac_res, s1_res, climate_res, emit_res = await asyncio.gather(
                asyncio.to_thread(self.stac_orchestrator.fetch_earth_stac_rasters, site),
                asyncio.to_thread(fetch_sentinel1_rtc_raster, site, self.cache, self.grid_size),
                fetch_earth_climate_and_atmosphere(center_lat, center_lon, site.start_date, site.end_date, self.cache),
                asyncio.to_thread(extract_emit_mineralogy, center_lat, center_lon),
                return_exceptions=True,
            )

            if isinstance(stac_res, dict):
                bundle.dem = stac_res.get("dem")
                bundle.s2_multispectral = stac_res.get("s2_multispectral")
                bundle.rgb_image = stac_res.get("rgb_image")
                bundle.albedo = stac_res.get("albedo")
                bundle.thermal_day_k = stac_res.get("thermal_day_k")

            if isinstance(s1_res, tuple):
                bundle.sar_backscatter, bundle.radar_cpr, _ = s1_res

            if isinstance(climate_res, dict) and climate_res:
                atm = climate_res.get("atmospheric", {})
                astro = climate_res.get("astrobiology", {})
                therm = climate_res.get("thermal", {})
                for key in ("aridity_index", "surface_pressure_pa", "uv_flux_wm2", "mean_wind_speed_ms", "dust_optical_depth"):
                    if atm.get(key) is not None:
                        bundle.atmospheric_scalars[key] = float(atm[key])
                for key in ("soil_water_activity_aw", "organic_preservation_potential", "ph", "soil_organic_carbon_g_kg"):
                    if astro.get(key) is not None:
                        bundle.astrobiology_scalars[key] = float(astro[key])
                if bundle.thermal_day_k is None and therm.get("t_max_k") is not None:
                    bundle.thermal_day_k = np.full((self.grid_size, self.grid_size), float(therm["t_max_k"]), dtype=np.float32)
                if bundle.thermal_night_k is None and therm.get("t_min_k") is not None:
                    bundle.thermal_night_k = np.full((self.grid_size, self.grid_size), float(therm["t_min_k"]), dtype=np.float32)

            if isinstance(emit_res, dict) and isinstance(emit_res.get("harmonized_profile"), dict):
                bundle.mineral_indices["harmonized_profile"] = emit_res["harmonized_profile"]

            await asyncio.to_thread(self._populate_available_embeddings, bundle)
            return bundle

        if site.body == PlanetBody.MARS:
            bundle = SiteDataBundle(site=site, pixel_resolution_m=18.0)
            mars_data = await fetch_mars_orbital_telemetry(center_lat, center_lon)
            bundle.rgb_image = mars_data.get("rgb_image")
            if isinstance(mars_data.get("harmonized_profile"), dict):
                bundle.mineral_indices["harmonized_profile"] = mars_data["harmonized_profile"]
            therm = mars_data.get("thermal", {})
            if therm.get("t_max_k") is not None:
                bundle.thermal_day_k = np.full((self.grid_size, self.grid_size), float(therm["t_max_k"]), dtype=np.float32)
            if therm.get("t_min_k") is not None:
                bundle.thermal_night_k = np.full((self.grid_size, self.grid_size), float(therm["t_min_k"]), dtype=np.float32)
            atm = mars_data.get("atmospheric", {})
            if atm.get("surface_pressure_pa") is not None:
                bundle.atmospheric_scalars["surface_pressure_pa"] = float(atm["surface_pressure_pa"])
            if atm.get("dust_optical_depth") is not None:
                bundle.atmospheric_scalars["dust_optical_depth"] = float(atm["dust_optical_depth"])
            await asyncio.to_thread(self._populate_available_embeddings, bundle)
            return bundle

        bundle = SiteDataBundle(site=site, pixel_resolution_m=20.0)
        moon_data = await fetch_moon_orbital_telemetry(center_lat, center_lon)
        bundle.rgb_image = moon_data.get("rgb_image")
        if moon_data.get("albedo") is not None:
            bundle.albedo = np.full((self.grid_size, self.grid_size), float(moon_data["albedo"]), dtype=np.float32)
        if isinstance(moon_data.get("harmonized_profile"), dict):
            bundle.mineral_indices["harmonized_profile"] = moon_data["harmonized_profile"]
        therm = moon_data.get("thermal", {})
        if therm.get("t_max_k") is not None:
            bundle.thermal_day_k = np.full((self.grid_size, self.grid_size), float(therm["t_max_k"]), dtype=np.float32)
        if therm.get("t_min_k") is not None:
            bundle.thermal_night_k = np.full((self.grid_size, self.grid_size), float(therm["t_min_k"]), dtype=np.float32)
        if moon_data.get("cpr_mean") is not None:
            bundle.radar_cpr = np.full((self.grid_size, self.grid_size), float(moon_data["cpr_mean"]), dtype=np.float32)
        await asyncio.to_thread(self._populate_available_embeddings, bundle)
        return bundle


# ===========================================================================
# 6. ALL 6 SCIENTIFIC SECTOR EVALUATORS & PIPELINE
# ===========================================================================
class BaseSectorEvaluator:
    sector: SectorID

    @property
    def name(self) -> str:
        return self.sector.value.replace("_", " ").title()

    @staticmethod
    def gaussian_scalar_similarity(a: Optional[float], b: Optional[float], tolerance: float) -> Optional[float]:
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            return None
        diff = float(a) - float(b)
        return float(np.exp(-0.5 * (diff / max(tolerance, 1e-8)) ** 2))

    @staticmethod
    def distribution_similarity_1d(arr_a: np.ndarray, arr_b: np.ndarray, characteristic_scale: float) -> Optional[float]:
        va = arr_a[np.isfinite(arr_a)].ravel()
        vb = arr_b[np.isfinite(arr_b)].ravel()
        if va.size < 4 or vb.size < 4:
            return None
        w_dist = float(wasserstein_distance(va, vb))
        return float(np.exp(-w_dist / max(characteristic_scale, 1e-8)))

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        a = np.asarray(vec_a, dtype=np.float32).ravel()
        b = np.asarray(vec_b, dtype=np.float32).ravel()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        cos = float(np.dot(a, b) / denom)
        return float(np.clip(0.5 * (cos + 1.0), 0.0, 1.0))

    def _empty_result(self, reason: str) -> SectorResult:
        return SectorResult(sector=self.sector, name=self.name, score=0.0, confidence=0.0, metrics=[], explanation=reason)

    def evaluate(self, earth_bundle: SiteDataBundle, target_bundle: SiteDataBundle) -> SectorResult:
        raise NotImplementedError


class GeomorphologyEvaluator(BaseSectorEvaluator):
    sector = SectorID.GEOMORPHOLOGY

    @staticmethod
    def compute_slope_map(dem: np.ndarray, pixel_res_m: float = 30.0) -> np.ndarray:
        res = max(float(pixel_res_m), 1.0)
        gy, gx = np.gradient(dem, res, res)
        return np.degrees(np.arctan(np.hypot(gy, gx))).astype(np.float32)

    @staticmethod
    def compute_rms_roughness(dem: np.ndarray) -> float:
        valid = dem[np.isfinite(dem)]
        if valid.size < 4:
            return 0.0
        residual = valid - np.nanmean(valid)
        return float(np.sqrt(np.mean(residual ** 2)))

    def evaluate(self, earth_bundle: SiteDataBundle, target_bundle: SiteDataBundle) -> SectorResult:
        metrics: List[MetricComparison] = []
        scores: List[float] = []
        weights: List[float] = []

        if earth_bundle.dem is not None and target_bundle.dem is not None:
            earth_slopes = self.compute_slope_map(earth_bundle.dem, earth_bundle.pixel_resolution_m)
            target_slopes = self.compute_slope_map(target_bundle.dem, target_bundle.pixel_resolution_m)
            slope_sim = self.distribution_similarity_1d(earth_slopes, target_slopes, characteristic_scale=5.0)
            if slope_sim is not None:
                metrics.append(MetricComparison("Slope Angle Distribution", f"{np.nanmean(earth_slopes):.1f}°", f"{np.nanmean(target_slopes):.1f}°", "degrees", round(slope_sim, 4)))
                scores.append(slope_sim)
                weights.append(0.35)

            e_rough = self.compute_rms_roughness(earth_bundle.dem)
            t_rough = self.compute_rms_roughness(target_bundle.dem)
            rough_sim = self.gaussian_scalar_similarity(e_rough, t_rough, tolerance=8.0)
            if rough_sim is not None:
                metrics.append(MetricComparison("Surface RMS Roughness", round(e_rough, 2), round(t_rough, 2), "m", round(rough_sim, 4)))
                scores.append(rough_sim)
                weights.append(0.25)

            e_relief = float(np.nanmax(earth_bundle.dem) - np.nanmin(earth_bundle.dem))
            t_relief = float(np.nanmax(target_bundle.dem) - np.nanmin(target_bundle.dem))
            relief_sim = self.gaussian_scalar_similarity(e_relief, t_relief, tolerance=max(30.0, 0.35 * max(e_relief, t_relief)))
            if relief_sim is not None:
                metrics.append(MetricComparison("Hypsometric Relief Range", round(e_relief, 1), round(t_relief, 1), "m", round(relief_sim, 4)))
                scores.append(relief_sim)
                weights.append(0.15)

        e_clip = earth_bundle.embeddings.get("earth_aligned_768", earth_bundle.embeddings.get("clip_768"))
        t_clip = target_bundle.embeddings.get("clip_768")
        if e_clip is not None and t_clip is not None:
            latent_sim = self.cosine_similarity(e_clip, t_clip)
            metrics.append(MetricComparison("Visual Morphological Latent Match", "RemoteCLIP-768D", "RemoteCLIP-768D", "cosine", round(latent_sim, 4)))
            scores.append(latent_sim)
            weights.append(0.25)

        if not scores:
            return self._empty_result("Missing DEM raster arrays and optical RGB embeddings.")

        final_score = float(sum(s * w for s, w in zip(scores, weights)) / sum(weights))
        confidence = float(min(1.0, len(scores) / 4.0))
        return SectorResult(self.sector, self.name, round(final_score, 4), round(confidence, 4), metrics, f"Geomorphology alignment: {final_score:.1%}.")


class MineralogyEvaluator(BaseSectorEvaluator):
    sector = SectorID.MINERALOGY

    @staticmethod
    def _extract_profile(bundle: SiteDataBundle) -> Optional[Dict[str, float]]:
        if "harmonized_profile" in bundle.mineral_indices and isinstance(bundle.mineral_indices["harmonized_profile"], dict):
            prof = bundle.mineral_indices["harmonized_profile"]
            return {cls: float(prof.get(cls, 0.0)) for cls in MINERAL_CLASSES}

        body = bundle.site.body.value
        if body == "mars":
            return {"mafic_olivine_pyroxene": 0.22, "plagioclase_anorthosite": 0.05, "phyllosilicates_clays": 0.34, "sulfates_evaporites": 0.26, "carbonates": 0.03, "iron_oxides_hydroxides": 0.10}
        if body == "moon":
            return {"mafic_olivine_pyroxene": 0.14, "plagioclase_anorthosite": 0.81, "phyllosilicates_clays": 0.00, "sulfates_evaporites": 0.00, "carbonates": 0.00, "iron_oxides_hydroxides": 0.05}
        return None  # Do not hallucinate Qaidam minerals when Earth lacks EMIT coverage

    @staticmethod
    def _compute_mean_albedo(bundle: SiteDataBundle) -> Optional[float]:
        if bundle.albedo is not None:
            valid = bundle.albedo[np.isfinite(bundle.albedo)]
            if valid.size > 0:
                return float(np.clip(np.nanmean(valid), 0.0, 1.0))
        if bundle.s2_multispectral is not None and bundle.s2_multispectral.ndim == 3:
            vis_nir = np.nanmean(bundle.s2_multispectral[:4])
            return float(np.clip(vis_nir / 10000.0 if vis_nir > 1.0 else vis_nir, 0.0, 1.0))
        if bundle.rgb_image is not None and bundle.rgb_image.size > 0:
            return float(np.nanmean(bundle.rgb_image) / 255.0)
        return {"moon": 0.12, "mars": 0.25, "earth": 0.30}.get(bundle.site.body.value)

    def evaluate(self, earth_bundle: SiteDataBundle, target_bundle: SiteDataBundle) -> SectorResult:
        metrics: List[MetricComparison] = []
        scores: List[float] = []
        weights: List[float] = []

        e_prof = self._extract_profile(earth_bundle)
        t_prof = self._extract_profile(target_bundle)
        if e_prof is not None and t_prof is not None:
            asi_score, _ = compute_mineral_asi(e_prof, t_prof)
            top_e = max(e_prof.items(), key=lambda kv: kv[1])[0]
            top_t = max(t_prof.items(), key=lambda kv: kv[1])[0]
            metrics.append(MetricComparison("6-Axis Mineralogy ASI", f"{top_e} ({e_prof[top_e]:.1%})", f"{top_t} ({t_prof[top_t]:.1%})", "ASI", round(asi_score, 4)))
            scores.append(asi_score)
            weights.append(0.50)

        e_albedo = self._compute_mean_albedo(earth_bundle)
        t_albedo = self._compute_mean_albedo(target_bundle)
        if e_albedo is not None and t_albedo is not None:
            albedo_sim = self.gaussian_scalar_similarity(e_albedo, t_albedo, tolerance=0.10)
            if albedo_sim is not None:
                metrics.append(MetricComparison("Bolometric Albedo", round(e_albedo, 3), round(t_albedo, 3), "reflectance", round(albedo_sim, 4)))
                scores.append(albedo_sim)
                weights.append(0.20)

        e_emb = earth_bundle.embeddings.get("earth_aligned_768", earth_bundle.embeddings.get("clip_768"))
        t_emb = target_bundle.embeddings.get("clip_768")
        if e_emb is not None and t_emb is not None:
            spectral_sim = self.cosine_similarity(e_emb, t_emb)
            metrics.append(MetricComparison("Spectral Latent Embedding Match", "Aligned-768D", "Planetary-768D", "cosine", round(spectral_sim, 4)))
            scores.append(spectral_sim)
            weights.append(0.30)

        if not scores:
            return self._empty_result("Missing mineral profiles, albedo, and spectral embeddings.")

        final_score = float(sum(s * w for s, w in zip(scores, weights)) / sum(weights))
        confidence = float(min(1.0, len(scores) / 3.0))
        return SectorResult(self.sector, self.name, round(final_score, 4), round(confidence, 4), metrics, f"Mineralogy alignment: {final_score:.1%}.")


class SubsurfaceEvaluator(BaseSectorEvaluator):
    sector = SectorID.SUBSURFACE

    def evaluate(self, earth_bundle: SiteDataBundle, target_bundle: SiteDataBundle) -> SectorResult:
        metrics: List[MetricComparison] = []
        scores: List[float] = []
        weights: List[float] = []

        e_cpr = earth_bundle.radar_cpr
        t_cpr = target_bundle.radar_cpr
        if e_cpr is not None and t_cpr is not None:
            cpr_sim = self.distribution_similarity_1d(e_cpr, t_cpr, characteristic_scale=0.15)
            if cpr_sim is not None:
                metrics.append(MetricComparison("Cross-Polarization Ratio (CPR)", round(float(np.nanmean(e_cpr)), 3), round(float(np.nanmean(t_cpr)), 3), "ratio", round(cpr_sim, 4)))
                scores.append(cpr_sim)
                weights.append(0.35)

                e_vol = float(np.mean(e_cpr > 0.35))
                t_vol = float(np.mean(t_cpr > 0.35))
                vol_sim = self.gaussian_scalar_similarity(e_vol, t_vol, tolerance=0.15)
                if vol_sim is not None:
                    metrics.append(MetricComparison("Volume Scattering Fraction", f"{e_vol:.1%}", f"{t_vol:.1%}", "fraction", round(vol_sim, 4)))
                    scores.append(vol_sim)
                    weights.append(0.20)

        e_bs = float(np.nanmean(earth_bundle.sar_backscatter[0])) if earth_bundle.sar_backscatter is not None else -12.0
        t_bs = float(np.nanmean(target_bundle.sar_backscatter[0])) if target_bundle.sar_backscatter is not None else (-18.5 if target_bundle.site.body == PlanetBody.MOON else -16.0)
        bs_sim = self.gaussian_scalar_similarity(e_bs, t_bs, tolerance=5.0)
        if bs_sim is not None:
            metrics.append(MetricComparison("Radar Backscatter Intensity", round(e_bs, 1), round(t_bs, 1), "dB", round(bs_sim, 4)))
            scores.append(bs_sim)
            weights.append(0.25)

        if "s1_1024" in earth_bundle.embeddings and "earth_aligned_768" in earth_bundle.embeddings and "clip_768" in target_bundle.embeddings:
            sar_sim = self.cosine_similarity(earth_bundle.embeddings["earth_aligned_768"], target_bundle.embeddings["clip_768"])
            metrics.append(MetricComparison("SAR Subsurface Latent Match", "ConvNeXtV2-S1", "Planetary-768D", "cosine", round(sar_sim, 4)))
            scores.append(sar_sim)
            weights.append(0.20)

        if not scores:
            return self._empty_result("Missing SAR backscatter and radar CPR products.")

        final_score = float(sum(s * w for s, w in zip(scores, weights)) / sum(weights))
        confidence = float(min(1.0, len(scores) / 3.0))
        return SectorResult(self.sector, self.name, round(final_score, 4), round(confidence, 4), metrics, f"Subsurface alignment: {final_score:.1%}.")


class ThermalEvaluator(BaseSectorEvaluator):
    sector = SectorID.THERMAL

    def evaluate(self, earth_bundle: SiteDataBundle, target_bundle: SiteDataBundle) -> SectorResult:
        e_day, e_night = earth_bundle.thermal_day_k, earth_bundle.thermal_night_k
        t_day, t_night = target_bundle.thermal_day_k, target_bundle.thermal_night_k
        if e_day is None or e_night is None or t_day is None or t_night is None:
            return self._empty_result("Missing day and night surface temperature rasters.")

        metrics: List[MetricComparison] = []
        scores: List[float] = []
        weights: List[float] = []

        mean_e_delta = float(np.nanmean(np.clip(e_day - e_night, 1.0, None)))
        mean_t_delta = float(np.nanmean(np.clip(t_day - t_night, 1.0, None)))
        delta_sim = self.gaussian_scalar_similarity(mean_e_delta, mean_t_delta, tolerance=35.0)
        if delta_sim is not None:
            metrics.append(MetricComparison("Diurnal Temperature Range", f"{mean_e_delta:.1f} K", f"{mean_t_delta:.1f} K", "Kelvin", round(delta_sim, 4)))
            scores.append(delta_sim)
            weights.append(0.25)

        e_mean = float(0.5 * (np.nanmean(e_day) + np.nanmean(e_night)))
        t_mean = float(0.5 * (np.nanmean(t_day) + np.nanmean(t_night)))
        mean_sim = self.gaussian_scalar_similarity(e_mean, t_mean, tolerance=45.0)
        if mean_sim is not None:
            metrics.append(MetricComparison("Mean Surface Temperature", f"{e_mean:.1f} K", f"{t_mean:.1f} K", "Kelvin", round(mean_sim, 4)))
            scores.append(mean_sim)
            weights.append(0.30)

        e_min, t_min = float(np.nanmin(e_night)), float(np.nanmin(t_night))
        min_sim = self.gaussian_scalar_similarity(e_min, t_min, tolerance=45.0)
        if min_sim is not None:
            metrics.append(MetricComparison("Extreme Minimum Temperature", f"{e_min:.1f} K", f"{t_min:.1f} K", "Kelvin", round(min_sim, 4)))
            scores.append(min_sim)
            weights.append(0.25)

        e_alb = 0.30 if earth_bundle.albedo is None else float(np.nanmean(earth_bundle.albedo))
        t_alb = 0.20 if target_bundle.albedo is None else float(np.nanmean(target_bundle.albedo))
        e_ati = float((1.0 - min(0.9, e_alb)) / max(mean_e_delta, 1.0)) * 100.0
        t_ati = float((1.0 - min(0.9, t_alb)) / max(mean_t_delta, 1.0)) * 100.0
        ati_sim = self.gaussian_scalar_similarity(e_ati, t_ati, tolerance=1.5)
        if ati_sim is not None:
            metrics.append(MetricComparison("Apparent Thermal Inertia (ATI)", round(e_ati, 2), round(t_ati, 2), "proxy", round(ati_sim, 4)))
            scores.append(ati_sim)
            weights.append(0.20)

        final_score = float(sum(s * w for s, w in zip(scores, weights)) / sum(weights))
        confidence = float(min(1.0, len(scores) / 4.0))
        return SectorResult(self.sector, self.name, round(final_score, 4), round(confidence, 4), metrics, f"Thermal alignment: {final_score:.1%}.")


class AtmosphericEvaluator(BaseSectorEvaluator):
    sector = SectorID.ATMOSPHERIC

    @staticmethod
    def _get_profile(bundle: SiteDataBundle) -> Dict[str, float]:
        defaults = {
            "moon": {"surface_pressure_pa": 1e-10, "mean_wind_speed_ms": 0.0, "aridity_index": 0.0, "uv_flux_wm2": 110.0},
            "mars": {"surface_pressure_pa": 610.0, "mean_wind_speed_ms": 5.5, "aridity_index": 0.0, "uv_flux_wm2": 65.0},
            "earth": {"surface_pressure_pa": 101325.0, "mean_wind_speed_ms": 6.0, "aridity_index": 0.03, "uv_flux_wm2": 35.0},
        }
        profile = defaults.get(bundle.site.body.value, defaults["earth"]).copy()
        profile.update({k: float(v) for k, v in (bundle.atmospheric_scalars or {}).items() if v is not None})
        return profile

    def evaluate(self, earth_bundle: SiteDataBundle, target_bundle: SiteDataBundle) -> SectorResult:
        e_atm, t_atm = self._get_profile(earth_bundle), self._get_profile(target_bundle)
        metrics: List[MetricComparison] = []
        scores: List[float] = []
        weights: List[float] = []

        aridity_sim = self.gaussian_scalar_similarity(e_atm["aridity_index"], t_atm["aridity_index"], tolerance=0.08)
        if aridity_sim is not None:
            metrics.append(MetricComparison("Aridity Index", round(e_atm["aridity_index"], 3), round(t_atm["aridity_index"], 3), "P/PET", round(aridity_sim, 4)))
            scores.append(aridity_sim)
            weights.append(0.35)

        wind_sim = self.gaussian_scalar_similarity(e_atm["mean_wind_speed_ms"], t_atm["mean_wind_speed_ms"], tolerance=4.0)
        if wind_sim is not None:
            metrics.append(MetricComparison("Aeolian Wind Regime", f"{e_atm['mean_wind_speed_ms']:.1f} m/s", f"{t_atm['mean_wind_speed_ms']:.1f} m/s", "m/s", round(wind_sim, 4)))
            scores.append(wind_sim)
            weights.append(0.25)

        pres_sim = self.gaussian_scalar_similarity(
            float(np.log10(max(1e-12, e_atm["surface_pressure_pa"]))),
            float(np.log10(max(1e-12, t_atm["surface_pressure_pa"]))),
            tolerance=1.0,
        )
        if pres_sim is not None:
            metrics.append(MetricComparison("Atmospheric Pressure Regime", f"{e_atm['surface_pressure_pa']:.0f} Pa", f"{t_atm['surface_pressure_pa']:.1e} Pa", "Pa", round(pres_sim, 4)))
            scores.append(pres_sim)
            weights.append(0.20)

        uv_sim = self.gaussian_scalar_similarity(e_atm["uv_flux_wm2"], t_atm["uv_flux_wm2"], tolerance=35.0)
        if uv_sim is not None:
            metrics.append(MetricComparison("Surface UV Radiation Flux", f"{e_atm['uv_flux_wm2']:.1f} W/m²", f"{t_atm['uv_flux_wm2']:.1f} W/m²", "W/m²", round(uv_sim, 4)))
            scores.append(uv_sim)
            weights.append(0.20)

        final_score = float(sum(s * w for s, w in zip(scores, weights)) / sum(weights))
        confidence = float(min(1.0, len(scores) / 4.0))
        return SectorResult(self.sector, self.name, round(final_score, 4), round(confidence, 4), metrics, f"Atmospheric alignment: {final_score:.1%}.")


class AstrobiologyEvaluator(BaseSectorEvaluator):
    sector = SectorID.ASTROBIOLOGY

    @staticmethod
    def _get_profile(bundle: SiteDataBundle) -> Dict[str, float]:
        scalars = dict(bundle.astrobiology_scalars)
        if bundle.site.body == PlanetBody.MOON:
            base = {"soil_water_activity_aw": 0.0, "organic_preservation_potential": 0.05, "ph": 7.0}
            base.update(scalars)
            return base
        return scalars

    def evaluate(self, earth_bundle: SiteDataBundle, target_bundle: SiteDataBundle) -> SectorResult:
        e_bio, t_bio = self._get_profile(earth_bundle), self._get_profile(target_bundle)
        metrics: List[MetricComparison] = []
        scores: List[float] = []
        weights: List[float] = []

        if e_bio.get("soil_water_activity_aw") is not None and t_bio.get("soil_water_activity_aw") is not None:
            aw_sim = self.gaussian_scalar_similarity(float(np.clip(e_bio["soil_water_activity_aw"], 0.0, 1.0)), float(np.clip(t_bio["soil_water_activity_aw"], 0.0, 1.0)), tolerance=0.20)
            if aw_sim is not None:
                metrics.append(MetricComparison("Water Activity (a_w)", round(e_bio["soil_water_activity_aw"], 3), round(t_bio["soil_water_activity_aw"], 3), "a_w", round(aw_sim, 4)))
                scores.append(aw_sim)
                weights.append(0.35)

        if e_bio.get("organic_preservation_potential") is not None and t_bio.get("organic_preservation_potential") is not None:
            bpp_sim = self.gaussian_scalar_similarity(float(np.clip(e_bio["organic_preservation_potential"], 0.0, 1.0)), float(np.clip(t_bio["organic_preservation_potential"], 0.0, 1.0)), tolerance=0.25)
            if bpp_sim is not None:
                metrics.append(MetricComparison("Biosignature Preservation Potential", f"{e_bio['organic_preservation_potential']:.1%}", f"{t_bio['organic_preservation_potential']:.1%}", "score", round(bpp_sim, 4)))
                scores.append(bpp_sim)
                weights.append(0.30)

        ph_sim = self.gaussian_scalar_similarity(e_bio.get("ph"), t_bio.get("ph"), tolerance=3.0)
        if ph_sim is not None:
            metrics.append(MetricComparison("Soil Topsoil pH", round(float(e_bio["ph"]), 1), round(float(t_bio["ph"]), 1), "pH", round(ph_sim, 4)))
            scores.append(ph_sim)
            weights.append(0.20)

        perc_sim = self.gaussian_scalar_similarity(e_bio.get("perchlorate_concentration_pct"), t_bio.get("perchlorate_concentration_pct"), tolerance=0.50)
        if perc_sim is not None:
            metrics.append(MetricComparison("Oxidant & Perchlorate Stress", f"{e_bio['perchlorate_concentration_pct']:.2f}%", f"{t_bio['perchlorate_concentration_pct']:.2f}%", "wt %", round(perc_sim, 4)))
            scores.append(perc_sim)
            weights.append(0.15)

        if not scores:
            return self._empty_result("Missing soil moisture, preservation, and geochemical measurements.")

        final_score = float(sum(s * w for s, w in zip(scores, weights)) / sum(weights))
        confidence = float(min(1.0, len(scores) / 4.0))
        return SectorResult(self.sector, self.name, round(final_score, 4), round(confidence, 4), metrics, f"Astrobiology alignment: {final_score:.1%}.")


class PlanetaryAnalogPipeline:
    def __init__(self) -> None:
        self.evaluators: Dict[SectorID, BaseSectorEvaluator] = {
            SectorID.GEOMORPHOLOGY: GeomorphologyEvaluator(),
            SectorID.MINERALOGY: MineralogyEvaluator(),
            SectorID.SUBSURFACE: SubsurfaceEvaluator(),
            SectorID.THERMAL: ThermalEvaluator(),
            SectorID.ATMOSPHERIC: AtmosphericEvaluator(),
            SectorID.ASTROBIOLOGY: AstrobiologyEvaluator(),
        }

    def compare(
        self,
        earth_bundle: SiteDataBundle,
        target_bundle: SiteDataBundle,
        mission_profile: MissionProfile = MissionProfile.BALANCED,
    ) -> AnalogComparisonReport:
        profile_weights = MISSION_WEIGHTS.get(mission_profile, MISSION_WEIGHTS[MissionProfile.BALANCED])
        total_w = sum(profile_weights.values())
        weights = {s: w / total_w for s, w in profile_weights.items()}

        sector_results: List[SectorResult] = []
        overall_score = 0.0
        overall_conf = 0.0
        key_strengths: List[str] = []
        key_divergences: List[str] = []

        for sector_id in SectorID:
            evaluator = self.evaluators[sector_id]
            try:
                res = evaluator.evaluate(earth_bundle, target_bundle)
            except Exception as exc:
                res = evaluator._empty_result(f"Evaluation error: {exc}")

            w = weights[sector_id]
            res.weighted_contribution = round(res.score * w, 4)
            sector_results.append(res)
            overall_score += res.score * w
            overall_conf += res.confidence * w

            if res.score >= 0.72:
                top_m = [m.name for m in res.metrics if m.similarity >= 0.75]
                desc = f" ({', '.join(top_m)})" if top_m else ""
                key_strengths.append(f"{res.name}: High parity ({res.score:.1%}){desc}")
            elif res.score < 0.45:
                key_divergences.append(f"{res.name}: Divergence ({res.score:.1%})")

        if not key_strengths:
            key_strengths.append("Baseline operational analogy across evaluated sectors.")
        if not key_divergences:
            key_divergences.append("No critical physical divergences identified.")

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


# ===========================================================================
# 7. AMERICAS FRONTIER DISCOVERY SCANNER (EXCLUDES FAMOUS SITES)
# ===========================================================================
PLANETARY_TARGETS: Dict[str, dict] = {
    "jezero_crater": {
        "site_id": "jezero_crater",
        "name": "Jezero Crater (Mars - Perseverance Landing Site)",
        "body": PlanetBody.MARS,
        "bbox": BoundingBox(min_lon=77.40, min_lat=18.40, max_lon=77.50, max_lat=18.50),
    },
    "gale_crater": {
        "site_id": "gale_crater",
        "name": "Gale Crater (Mars - Glen Torridon / Mount Sharp)",
        "body": PlanetBody.MARS,
        "bbox": BoundingBox(min_lon=137.35, min_lat=-4.70, max_lon=137.45, max_lat=-4.60),
    },
    "mons_mouton_plateau": {
        "site_id": "mons_mouton_plateau",
        "name": "Mons Mouton Plateau (Moon - Artemis South Pole Candidate)",
        "body": PlanetBody.MOON,
        "bbox": BoundingBox(min_lon=30.90, min_lat=-84.65, max_lon=31.10, max_lat=-84.55),
    },
    "haworth_crater": {
        "site_id": "haworth_crater",
        "name": "Haworth Crater Rim (Moon - Lunar South Pole)",
        "body": PlanetBody.MOON,
        "bbox": BoundingBox(min_lon=-5.20, min_lat=-87.60, max_lon=-4.80, max_lat=-87.40),
    },
}

KNOWN_ANALOG_EXCLUSIONS: List[Tuple[str, float, float, float]] = [
    ("Atacama_Yungay_Core", -24.10, -70.00, 65.0),
    ("Salar_de_Uyuni_Tourist_Core", -20.13, -67.49, 55.0),
    ("Mojave_Desert_Testbed", 35.14, -116.10, 85.0),
    ("Death_Valley_Mars_Hill", 36.50, -117.08, 65.0),
    ("MDRS_Hanksville_Utah", 38.41, -110.79, 55.0),
    ("White_Sands_New_Mexico", 32.78, -106.33, 55.0),
    ("Craters_of_the_Moon_Idaho", 43.42, -113.52, 45.0),
    ("Pinacate_Sonora_Mexico", 31.77, -113.50, 45.0),
    ("Devon_Island_Haughton_Crater", 75.43, -89.85, 75.0),
    ("Axel_Heiberg_McGill_Station", 79.41, -90.75, 55.0),
    ("Kilbourne_Hole_New_Mexico", 31.97, -106.97, 35.0),
    ("Flagstaff_Cinder_Lake_AZ", 35.33, -111.53, 45.0),
]

UNEXPLORED_CORRIDORS: Dict[str, List[Tuple[str, float, float, float, float]]] = {
    "north_america_frontier": [
        ("Owyhee_Alvord_Basin", -119.5, 40.5, -115.5, 43.5),
        ("Bolson_de_Mapimi_Mexico", -104.5, 25.5, -101.0, 29.0),
        ("Vizcaino_Baja_Mesas", -115.0, 26.5, -112.0, 29.5),
        ("Sevier_West_Desert_UT", -114.0, 38.5, -112.5, 41.2),
        ("Mackenzie_Peel_Plateau", -136.0, 64.5, -130.0, 67.5),
    ],
    "south_america_frontier": [
        ("Sechura_Coastal_Desert_PE", -81.2, -6.8, -79.5, -4.8),
        ("Nazca_Pampa_Plains_PE", -76.0, -15.5, -74.2, -13.8),
        ("Catamarca_Puna_Salars_AR", -68.8, -27.8, -66.5, -25.2),
        ("Payunia_Volcanic_Field_AR", -69.8, -37.2, -68.0, -35.5),
        ("Somuncura_Deseado_Massif_AR", -69.5, -48.5, -66.5, -41.0),
    ],
    "all_americas_frontier": [
        ("Owyhee_Alvord_Basin", -119.5, 40.5, -115.5, 43.5),
        ("Bolson_de_Mapimi_Mexico", -104.5, 25.5, -101.0, 29.0),
        ("Vizcaino_Baja_Mesas", -115.0, 26.5, -112.0, 29.5),
        ("Sevier_West_Desert_UT", -114.0, 38.5, -112.5, 41.2),
        ("Mackenzie_Peel_Plateau", -136.0, 64.5, -130.0, 67.5),
        ("Sechura_Coastal_Desert_PE", -81.2, -6.8, -79.5, -4.8),
        ("Nazca_Pampa_Plains_PE", -76.0, -15.5, -74.2, -13.8),
        ("Catamarca_Puna_Salars_AR", -68.8, -27.8, -66.5, -25.2),
        ("Payunia_Volcanic_Field_AR", -69.8, -37.2, -68.0, -35.5),
        ("Somuncura_Deseado_Massif_AR", -69.5, -48.5, -66.5, -41.0),
    ],
}


@dataclass
class NovelAnalogFind:
    rank: int
    stage: str
    tile_id: str
    corridor: str
    center_lat: float
    center_lon: float
    bbox: Tuple[float, float, float, float]
    nearest_known_site: str
    distance_to_known_km: float
    aridity_index: Optional[float]
    overall_score: float
    overall_confidence: float
    geomorphology: float
    mineralogy: float
    subsurface: float
    thermal: float
    atmospheric: float
    astrobiology: float
    sensors_active: List[str]
    models_active: List[str]
    key_strengths: List[str]
    key_divergences: List[str]
    elapsed_s: float


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def check_novelty_geofence(lat: float, lon: float) -> Tuple[bool, str, float]:
    nearest_name = "None"
    min_dist = float("inf")
    for name, k_lat, k_lon, excl_radius_km in KNOWN_ANALOG_EXCLUSIONS:
        d = _haversine_km(lat, lon, k_lat, k_lon)
        if d < min_dist:
            min_dist = d
            nearest_name = name
        if d <= excl_radius_km:
            return False, name, round(d, 1)
    return True, nearest_name, round(min_dist, 1)


def generate_novel_tiles(
    corridor_key: str,
    tile_deg: float = 0.45,
    custom_bbox: Optional[Tuple[float, float, float, float]] = None,
    stage_prefix: str = "s1",
) -> Tuple[List[Tuple[str, SiteQuery]], int]:
    boxes = [("Custom_Frontier_Area", *custom_bbox)] if custom_bbox is not None else UNEXPLORED_CORRIDORS[corridor_key]
    tiles: List[Tuple[str, SiteQuery]] = []
    excluded_known_count = 0

    for corridor_name, min_lon, min_lat, max_lon, max_lat in boxes:
        lat = min_lat
        row = 0
        while lat + tile_deg <= max_lat + 1e-6:
            lon = min_lon
            col = 0
            c_lat = lat + 0.5 * tile_deg
            while lon + tile_deg <= max_lon + 1e-6:
                c_lon = lon + 0.5 * tile_deg
                if globe is not None and not bool(globe.is_land(c_lat, c_lon)):
                    lon += tile_deg
                    col += 1
                    continue

                is_novel, _, _ = check_novelty_geofence(c_lat, c_lon)
                if not is_novel:
                    excluded_known_count += 1
                    lon += tile_deg
                    col += 1
                    continue

                bbox = BoundingBox(round(lon, 5), round(lat, 5), round(lon + tile_deg, 5), round(lat + tile_deg, 5))
                tile_id = f"{stage_prefix}_{corridor_name}_r{row:03d}_c{col:03d}"
                tiles.append((corridor_name, SiteQuery(tile_id, f"{corridor_name} ({c_lat:.3f}, {c_lon:.3f})", PlanetBody.EARTH, bbox)))
                lon += tile_deg
                col += 1
            lat += tile_deg
            row += 1
    return tiles, excluded_known_count


def subdivide_top_finds(top_finds: List[NovelAnalogFind], fine_deg: float = 0.10) -> List[Tuple[str, SiteQuery]]:
    fine_tiles: List[Tuple[str, SiteQuery]] = []
    for parent in top_finds:
        sub, _ = generate_novel_tiles("custom", tile_deg=fine_deg, custom_bbox=parent.bbox, stage_prefix=f"s2_zoom_{parent.tile_id}")
        fine_tiles.extend([(parent.corridor, q) for _, q in sub])
    return fine_tiles


class DynamicNovelStore:
    def __init__(self, out_dir: Path, run_tag: str) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / f"{run_tag}_stream.jsonl"
        self.csv_path = self.out_dir / f"{run_tag}_ranked.csv"
        self.json_path = self.out_dir / f"{run_tag}_ranked.json"
        self.records: Dict[str, NovelAnalogFind] = {}
        self.seen_ids: set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.jsonl_path.is_file():
            return
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                self.seen_ids.add(raw["tile_id"])
                if raw.get("status") == "filtered_out":
                    continue
                raw.pop("status", None)
                raw["bbox"] = tuple(raw["bbox"])
                rec = NovelAnalogFind(**raw)
                self.records[rec.tile_id] = rec
        if self.seen_ids:
            print(f"[RESUME] Loaded {len(self.records)} saved analogs ({len(self.seen_ids)} processed) from {self.jsonl_path}")

    def is_processed(self, tile_id: str) -> bool:
        return tile_id in self.seen_ids

    def record_filtered_out(self, tile_id: str, reason: str) -> None:
        self.seen_ids.add(tile_id)
        with open(self.jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps({"tile_id": tile_id, "status": "filtered_out", "reason": reason}) + "\n")

    def save_find(self, find: NovelAnalogFind) -> List[NovelAnalogFind]:
        self.seen_ids.add(find.tile_id)
        self.records[find.tile_id] = find
        with open(self.jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(asdict(find)) + "\n")

        ranked = sorted(self.records.values(), key=lambda r: r.overall_score, reverse=True)
        for idx, item in enumerate(ranked, start=1):
            item.rank = idx

        with open(self.json_path, "w", encoding="utf-8") as out_j:
            json.dump([asdict(r) for r in ranked], out_j, indent=2)

        fieldnames = [
            "rank", "stage", "tile_id", "corridor", "center_lat", "center_lon",
            "nearest_known_site", "distance_to_known_km", "aridity_index",
            "overall_score", "overall_confidence", "geomorphology", "mineralogy",
            "subsurface", "thermal", "atmospheric", "astrobiology",
            "sensors_active", "models_active", "key_strengths",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as out_c:
            writer = csv.DictWriter(out_c, fieldnames=fieldnames)
            writer.writeheader()
            for r in ranked:
                row = asdict(r)
                row["sensors_active"] = "+".join(r.sensors_active)
                row["models_active"] = "+".join(r.models_active)
                row["key_strengths"] = " | ".join(r.key_strengths)
                writer.writerow({k: row[k] for k in fieldnames})
        return ranked


def _inspect_bundle_telemetry(bundle: SiteDataBundle) -> Tuple[List[str], List[str]]:
    sensors = []
    if bundle.dem is not None:
        sensors.append("DEM")
    if bundle.s2_multispectral is not None:
        sensors.append("S2")
    if bundle.sar_backscatter is not None:
        sensors.append("S1_SAR")
    if bundle.thermal_day_k is not None:
        sensors.append("Thermal")
    if bundle.atmospheric_scalars:
        sensors.append("POWER")
    if bundle.astrobiology_scalars:
        sensors.append("SoilGrids")
    if "harmonized_profile" in bundle.mineral_indices:
        sensors.append("EMIT")

    models = []
    if "clip_768" in bundle.embeddings:
        models.append("RemoteCLIP")
    if "s1_1024" in bundle.embeddings:
        models.append("ConvNeXtV2-S1")
    if "s2_1024" in bundle.embeddings:
        models.append("ConvNeXtV2-S2")
    if "earth_aligned_768" in bundle.embeddings:
        models.append("LatentAligner")
    return sensors, models


async def scan_corridor_batch(
    tiles: List[Tuple[str, SiteQuery]],
    stage_label: str,
    target_bundle: SiteDataBundle,
    provider: UnifiedPlanetaryDataProvider,
    pipeline: PlanetaryAnalogPipeline,
    mission_profile: MissionProfile,
    store: DynamicNovelStore,
    cache: DiskCacheManager,
    max_aridity: float = 0.25,
    concurrency: int = 3,
    min_confidence: float = 0.30,
) -> List[NovelAnalogFind]:
    pending = [(corr, q) for corr, q in tiles if not store.is_processed(q.site_id)]
    total = len(tiles)
    done_count = total - len(pending)

    print(f"\n>>> [{stage_label.upper()}] {total} novel frontier tiles | {done_count} cached | {len(pending)} remaining")
    semaphore = asyncio.Semaphore(concurrency)
    io_lock = asyncio.Lock()

    async def _worker(corridor: str, query: SiteQuery) -> None:
        nonlocal done_count
        async with semaphore:
            t0 = time.perf_counter()
            c_lon, c_lat = query.bbox.center()
            _, nearest_known, dist_km = check_novelty_geofence(c_lat, c_lon)

            try:
                quick_clim = await fetch_earth_climate_and_atmosphere(c_lat, c_lon, query.start_date, query.end_date, cache)
                aridity = quick_clim.get("atmospheric", {}).get("aridity_index") if isinstance(quick_clim, dict) else None
                if aridity is not None and aridity > max_aridity:
                    elapsed = round(time.perf_counter() - t0, 2)
                    async with io_lock:
                        done_count += 1
                        store.record_filtered_out(query.site_id, f"aridity={aridity:.3f}>{max_aridity}")
                        print(f"  [PRE-SKIP] ({done_count}/{total}) {corridor} ({c_lat:+.2f}, {c_lon:+.2f}) | Humid (P/PET={aridity:.3f} > {max_aridity}) [{elapsed}s]")
                    return

                print(f"  [EVAL    ] {corridor} ({c_lat:+.3f}, {c_lon:+.3f}) | {dist_km:.0f}km from {nearest_known} | Aridity={aridity if aridity is not None else 'N/A'}...")
                earth_bundle = await provider.load_bundle_async(query)
                report = pipeline.compare(earth_bundle, target_bundle, mission_profile)
                elapsed = round(time.perf_counter() - t0, 2)
                sensors, models = _inspect_bundle_telemetry(earth_bundle)
                s_map = {sr.sector: sr.score for sr in report.sector_results}

                find = NovelAnalogFind(
                    rank=0,
                    stage=stage_label,
                    tile_id=query.site_id,
                    corridor=corridor,
                    center_lat=round(c_lat, 5),
                    center_lon=round(c_lon, 5),
                    bbox=query.bbox.to_tuple(),
                    nearest_known_site=nearest_known,
                    distance_to_known_km=dist_km,
                    aridity_index=round(float(aridity), 4) if aridity is not None else None,
                    overall_score=report.overall_similarity_score,
                    overall_confidence=report.overall_confidence,
                    geomorphology=s_map.get(SectorID.GEOMORPHOLOGY, 0.0),
                    mineralogy=s_map.get(SectorID.MINERALOGY, 0.0),
                    subsurface=s_map.get(SectorID.SUBSURFACE, 0.0),
                    thermal=s_map.get(SectorID.THERMAL, 0.0),
                    atmospheric=s_map.get(SectorID.ATMOSPHERIC, 0.0),
                    astrobiology=s_map.get(SectorID.ASTROBIOLOGY, 0.0),
                    sensors_active=sensors,
                    models_active=models,
                    key_strengths=report.key_strengths,
                    key_divergences=report.key_divergences,
                    elapsed_s=elapsed,
                )

                async with io_lock:
                    done_count += 1
                    if report.overall_confidence < min_confidence:
                        store.record_filtered_out(query.site_id, f"conf={report.overall_confidence:.2f}")
                        print(f"  [LOW-CONF] ({done_count}/{total}) {query.site_id} | Confidence {report.overall_confidence:.0%} [{elapsed}s]")
                        return

                    ranked = store.save_find(find)
                    print(
                        f"  [FIND #{find.rank:02d}] ({done_count}/{total}) {corridor} ({c_lat:+.3f}, {c_lon:+.3f}) | "
                        f"{dist_km:.0f}km from {nearest_known} | SCORE: {find.overall_score:.1%} (Conf:{find.overall_confidence:.0%}) | "
                        f"Geo:{find.geomorphology:.0%} Min:{find.mineralogy:.0%} Sub:{find.subsurface:.0%} "
                        f"Thm:{find.thermal:.0%} Atm:{find.atmospheric:.0%} Bio:{find.astrobiology:.0%} | "
                        f"Data:[{','.join(sensors)}] Models:[{','.join(models)}] [{elapsed}s]"
                    )
                    if find.rank == 1 and len(ranked) > 1:
                        print(f"  *** NEW #1 UNEXPLORED ANALOG: {find.corridor} ({find.center_lat:+.3f}, {find.center_lon:+.3f}) -> {find.overall_score:.1%} ***")
            except Exception as exc:
                async with io_lock:
                    done_count += 1
                    print(f"  [ERROR   ] ({done_count}/{total}) {query.site_id}: {exc}")

    await asyncio.gather(*[_worker(corr, q) for corr, q in pending])
    return sorted(store.records.values(), key=lambda r: r.overall_score, reverse=True)


async def main_async(args: argparse.Namespace) -> None:
    print("=" * 84)
    print("AEQUISCAN STANDALONE DISCOVERY ENGINE — UNEXPLORED AMERICAS ANALOGS")
    print("=" * 84)

    weights_mgr = ModelWeightsManager()
    if args.ensure_weights:
        print("[INIT] Verifying/downloading ConvNeXtV2-S1, ConvNeXtV2-S2 & RemoteCLIP weights...")
        weights_mgr.fetch_all(local_files_only=False)
    else:
        print(f"[INIT] Neural weights status on disk: {{k: weights_mgr.is_downloaded(k) for k in weights_mgr.registry}}")

    aligner = None
    for candidate in [
        Path(args.aligner_path) if args.aligner_path else None,
        DEFAULT_WEIGHTS_DIR / "latent_aligner.joblib",
        DEFAULT_WEIGHTS_DIR / "space_aligner.joblib",
    ]:
        if candidate is not None and candidate.is_file():
            aligner = LatentSpaceAligner.load(candidate)
            print(f"[INIT] Loaded LatentSpaceAligner from {candidate} (input_dim={aligner.input_dim})")
            break

    cache = DiskCacheManager()
    provider = UnifiedPlanetaryDataProvider(cache=cache, grid_size=args.grid_size, aligner=aligner)
    pipeline = PlanetaryAnalogPipeline()
    mission_profile = MissionProfile(args.mission_profile.lower())

    if args.target_body and args.target_lat is not None and args.target_lon is not None:
        body_enum = PlanetBody(args.target_body.lower())
        t_query = SiteQuery(
            f"{body_enum.value}_{args.target_lat:.2f}_{args.target_lon:.2f}",
            f"Custom {body_enum.value.title()} ({args.target_lat:.3f}, {args.target_lon:.3f})",
            body_enum,
            BoundingBox(args.target_lon - 0.05, args.target_lat - 0.05, args.target_lon + 0.05, args.target_lat + 0.05),
        )
    else:
        spec = PLANETARY_TARGETS[args.target_id]
        t_query = SiteQuery(spec["site_id"], spec["name"], spec["body"], spec["bbox"])

    print(f"[TARGET] Fetching live NASA PDS orbital data for: {t_query.name} ...")
    target_bundle = await provider.load_bundle_async(t_query)
    t_sensors, t_models = _inspect_bundle_telemetry(target_bundle)
    print(f"[TARGET] Ready | Telemetry: {t_sensors} | Models: {t_models}")

    run_tag = f"novel_{args.corridor}_{t_query.site_id}_{mission_profile.value}"
    store = DynamicNovelStore(out_dir=Path(args.output_dir), run_tag=run_tag)
    print(f"[OUTPUT] Live results dynamically saved to:\n         -> {store.csv_path}\n         -> {store.json_path}")

    custom_bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
    coarse_tiles, excluded_count = generate_novel_tiles(args.corridor, args.coarse_deg, custom_bbox, "s1")
    print(f"[GRID  ] Generated {len(coarse_tiles)} unexplored frontier tiles (excluded {excluded_count} tiles inside known NASA/ESA testbeds).")

    ranked_finds = await scan_corridor_batch(
        coarse_tiles, "stage1_coarse", target_bundle, provider, pipeline, mission_profile, store, cache, args.max_aridity, args.concurrency, args.min_confidence
    )

    if args.zoom_top_k > 0 and ranked_finds:
        top_coarse = [r for r in ranked_finds if r.stage == "stage1_coarse"][: args.zoom_top_k]
        fine_tiles = subdivide_top_finds(top_coarse, fine_deg=args.fine_deg)
        if fine_tiles:
            print(f"\n[ZOOM  ] Subdividing top {len(top_coarse)} novel discoveries into {len(fine_tiles)} high-res tiles ({args.fine_deg}°)...")
            ranked_finds = await scan_corridor_batch(
                fine_tiles, "stage2_fine", target_bundle, provider, pipeline, mission_profile, store, cache, args.max_aridity, args.concurrency, args.min_confidence
            )

    print("\n" + "=" * 84)
    print(f"TOP 10 NEWLY DISCOVERED UNEXPLORED ANALOGS FOR {t_query.name.upper()}")
    print("=" * 84)
    for r in ranked_finds[:10]:
        print(f"#{r.rank:02d} [{r.stage:<13}] {r.corridor:<26} ({r.center_lat:+7.3f}, {r.center_lon:+8.3f}) | Score: {r.overall_score:.1%} | {r.distance_to_known_km:.0f} km from {r.nearest_known_site}")
        print(f"    Sectors -> Geo:{r.geomorphology:.0%} | Min:{r.mineralogy:.0%} | Sub:{r.subsurface:.0%} | Thm:{r.thermal:.0%} | Atm:{r.atmospheric:.0%} | Bio:{r.astrobiology:.0%}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone AequiScan Novel Analog Discovery Engine.")
    p.add_argument("--target-id", type=str, default="jezero_crater", help="jezero_crater, gale_crater, mons_mouton_plateau, haworth_crater")
    p.add_argument("--target-body", type=str, choices=["mars", "moon"], default=None)
    p.add_argument("--target-lat", type=float, default=None)
    p.add_argument("--target-lon", type=float, default=None)
    p.add_argument("--corridor", type=str, default="north_america_frontier", help="north_america_frontier, south_america_frontier, all_americas_frontier")
    p.add_argument("--bbox", type=str, default=None, help="Optional custom box 'min_lon,min_lat,max_lon,max_lat'")
    p.add_argument("--coarse-deg", type=float, default=0.45)
    p.add_argument("--fine-deg", type=float, default=0.10)
    p.add_argument("--zoom-top-k", type=int, default=4)
    p.add_argument("--max-aridity", type=float, default=0.25)
    p.add_argument("--mission-profile", type=str, default="balanced")
    p.add_argument("--ensure-weights", action="store_true")
    p.add_argument("--aligner-path", type=str, default=None)
    p.add_argument("--grid-size", type=int, default=120)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--min-confidence", type=float, default=0.30)
    p.add_argument("--output-dir", type=str, default="novel_analog_discoveries")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
