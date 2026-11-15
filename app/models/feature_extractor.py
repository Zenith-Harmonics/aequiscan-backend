import dataclasses
import inspect
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union, get_args, get_origin, get_type_hints

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from configilm.ConfigILM import ConfigILM as ILMModel
from configilm.ConfigILM import ILMConfiguration
from PIL import Image

from app.core.schemas import PlanetBody, SiteDataBundle
from app.models.space_aligner import LatentSpaceAligner
from app.models.weights_manager import ModelWeightsManager

# BigEarthNet v2.0 channel-wise normalization statistics
# Sentinel-1 order: [VV_dB, VH_dB]
BEN_V2_S1_MEAN = np.array([-12.619994, -19.290446], dtype=np.float32).reshape(1, 2, 1, 1)
BEN_V2_S1_STD = np.array([5.115912, 5.464428], dtype=np.float32).reshape(1, 2, 1, 1)

# Sentinel-2 L2A 10-band order: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
BEN_V2_S2_MEAN = np.array(
    [429.94, 614.22, 590.24, 950.68, 1792.46, 2075.47, 2218.95, 2266.46, 1594.43, 1009.33],
    dtype=np.float32,
).reshape(1, 10, 1, 1)
BEN_V2_S2_STD = np.array(
    [572.42, 582.88, 675.89, 729.89, 1096.01, 1273.45, 1365.46, 1356.14, 1079.03, 818.87],
    dtype=np.float32,
).reshape(1, 10, 1, 1)


class MultiModalFeatureExtractor:
    """
    Lazy-loading inference engine for Sentinel-1 SAR, Sentinel-2 Multispectral,
    and RemoteCLIP ViT-L-14 embeddings.
    """

    def __init__(
        self,
        model_paths: Optional[Dict[str, Path]] = None,
        device: Optional[str] = None,
        auto_fetch: bool = False,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if model_paths is None:
            manager = ModelWeightsManager()
            if auto_fetch:
                self.model_paths = manager.fetch_all()
            else:
                self.model_paths = {k: manager.get_model_path(k) for k in manager.registry}
        else:
            self.model_paths = {k: Path(v) for k, v in model_paths.items()}

        self._models: Dict[str, torch.nn.Module] = {}
        self._backbones: Dict[str, torch.nn.Module] = {}
        self._clip_preprocess = None

    @staticmethod
    def _coerce_enum_value(field_type: Any, raw_value: Any) -> Any:
        """Convert serialized JSON primitives back into Enum instances when annotated."""
        if raw_value is None:
            return None

        origin = get_origin(field_type)
        candidate_types = get_args(field_type) if origin is not None else (field_type,)

        for candidate in candidate_types:
            if isinstance(candidate, type) and issubclass(candidate, Enum):
                if isinstance(raw_value, candidate):
                    return raw_value
                # Handle integer-backed IntEnums serialized as int or numeric string
                try:
                    return candidate(int(raw_value))
                except (ValueError, TypeError):
                    return candidate(raw_value)
        return raw_value

    @classmethod
    def _build_ilm_config(cls, config_path: Path) -> ILMConfiguration:
        """Reconstruct a valid ILMConfiguration dataclass from a Hugging Face config.json."""
        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)

        # Strip leading underscores from private dataclass property backing fields
        normalized_dict = {k.lstrip("_"): v for k, v in raw_dict.items()}

        sig = inspect.signature(ILMConfiguration.__init__)
        valid_params = {p for p in sig.parameters if p != "self"}
        type_hints = get_type_hints(ILMConfiguration)

        filtered_kwargs: Dict[str, Any] = {}
        for key, value in normalized_dict.items():
            if key in valid_params:
                field_type = type_hints.get(key)
                filtered_kwargs[key] = (
                    cls._coerce_enum_value(field_type, value) if field_type is not None else value
                )

        # Ensure any dataclass fields with Enum defaults remain properly typed
        if dataclasses.is_dataclass(ILMConfiguration):
            for dc_field in dataclasses.fields(ILMConfiguration):
                if dc_field.name in filtered_kwargs:
                    filtered_kwargs[dc_field.name] = cls._coerce_enum_value(
                        dc_field.type, filtered_kwargs[dc_field.name]
                    )

        return ILMConfiguration(**filtered_kwargs)

    @staticmethod
    def _load_local_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
        """Load weights from .safetensors or .bin inside a local Hugging Face snapshot folder."""
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
            if not path.exists():
                raise FileNotFoundError(
                    f"Model directory missing at '{path}'. Run ModelWeightsManager.fetch_all() first."
                )

            model: Optional[torch.nn.Module] = None

            # 1. Preferred path: PyTorchModelHubMixin classmethod on the ConfigILM class
            if hasattr(ILMModel, "from_pretrained"):
                try:
                    model = ILMModel.from_pretrained(str(path))
                except Exception:
                    model = None

            # 2. Deterministic fallback: reconstruct ILMConfiguration + load safetensors
            if model is None:
                config_file = path / "config.json"
                if not config_file.is_file():
                    raise FileNotFoundError(f"Missing config.json in '{path}'.")

                config = self._build_ilm_config(config_file)
                model = ILMModel(config)

                state_dict = self._load_local_state_dict(path)
                model_keys = set(model.state_dict().keys())

                # Align potential 'model.' prefix differences between checkpoint and module
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
        """Locate the underlying timm ConvNeXtV2 vision module inside ConfigILM."""
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
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"RemoteCLIP checkpoint missing at '{ckpt_path}'. Run ModelWeightsManager.fetch_all() first."
                )
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
    def _prepare_multiband_tensor(
        data: Union[np.ndarray, torch.Tensor],
        expected_channels: int,
        target_size: int = 120,
    ) -> torch.Tensor:
        """Convert (C, H, W) or (B, C, H, W) input to a resized (B, C, 120, 120) float32 tensor."""
        tensor = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}.")

        if tensor.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got {tensor.shape[1]}."
            )

        if tensor.shape[-2:] != (target_size, target_size):
            tensor = F.interpolate(
                tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        return tensor

    def _forward_sentinel_backbone(self, key: str, x: torch.Tensor, l2_normalize: bool) -> np.ndarray:
        """Run forward pass through the resolved ConvNeXtV2 backbone and return 1024D pre-logits."""
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
    def extract_sentinel1(
        self,
        sar_patch: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract 1024D embedding for Sentinel-1 GRD backscatter [VV_dB, VH_dB].
        """
        x = self._prepare_multiband_tensor(sar_patch, expected_channels=2)
        if normalize:
            x = (x - torch.from_numpy(BEN_V2_S1_MEAN)) / torch.from_numpy(BEN_V2_S1_STD)
        return self._forward_sentinel_backbone("sentinel_1", x, l2_normalize=l2_normalize)

    @torch.inference_mode()
    def extract_sentinel2(
        self,
        ms_patch: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract 1024D embedding for Sentinel-2 L2A 10-band reflectance
        [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12].
        """
        x = self._prepare_multiband_tensor(ms_patch, expected_channels=10)
        if normalize:
            x = (x - torch.from_numpy(BEN_V2_S2_MEAN)) / torch.from_numpy(BEN_V2_S2_STD)
        return self._forward_sentinel_backbone("sentinel_2", x, l2_normalize=l2_normalize)

    @torch.inference_mode()
    def extract_optical_clip(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract 768D RemoteCLIP ViT-L-14 embedding from an RGB optical patch
        (Mars HiRISE, Moon LROC NAC, or Earth RGB).
        """
        model = self._load_remoteclip_model()

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            arr = image
            if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
                arr = np.clip(arr * 255.0, 0.0, 255.0)
            arr_u8 = arr.astype(np.uint8)
            if arr_u8.ndim == 2:
                pil_img = Image.fromarray(arr_u8).convert("RGB")
            else:
                pil_img = Image.fromarray(arr_u8[..., :3]).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type for RemoteCLIP: {type(image)}")

        tensor = self._clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        embeddings = model.encode_image(tensor)

        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

    def populate_bundle_embeddings(
        self,
        bundle: SiteDataBundle,
        aligner: Optional[LatentSpaceAligner] = None,
    ) -> SiteDataBundle:
        """
        Populate available neural embeddings on a SiteDataBundle in-place.
        If an aligner is provided for an Earth bundle, projects Sentinel embeddings into 768D space.
        """
        if bundle.rgb_image is not None:
            bundle.embeddings["clip_768"] = self.extract_optical_clip(bundle.rgb_image)[0]

        if bundle.site.body == PlanetBody.EARTH:
            if bundle.sar_backscatter is not None:
                bundle.embeddings["s1_1024"] = self.extract_sentinel1(bundle.sar_backscatter)[0]

            if bundle.s2_multispectral is not None:
                bundle.embeddings["s2_1024"] = self.extract_sentinel2(bundle.s2_multispectral)[0]

            if aligner is not None and aligner.input_dim is not None:
                if aligner.input_dim == 2048 and "s1_1024" in bundle.embeddings and "s2_1024" in bundle.embeddings:
                    fused = np.concatenate([bundle.embeddings["s1_1024"], bundle.embeddings["s2_1024"]])
                    bundle.embeddings["earth_aligned_768"] = aligner.project_earth_to_target_space(fused)[0]
                elif aligner.input_dim == 1024 and "s2_1024" in bundle.embeddings:
                    bundle.embeddings["earth_aligned_768"] = aligner.project_earth_to_target_space(
                        bundle.embeddings["s2_1024"]
                    )[0]

        return bundle