from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from configilm import ConfigILM
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
        self._clip_preprocess = None

    def _load_sentinel_model(self, key: str) -> torch.nn.Module:
        if key not in self._models:
            path = self.model_paths[key]
            if not path.exists():
                raise FileNotFoundError(
                    f"Model directory missing at '{path}'. Run ModelWeightsManager.fetch_all() first."
                )
            model = ConfigILM.from_pretrained(str(path))
            model.to(self.device)
            model.eval()
            self._models[key] = model
        return self._models[key]

    def _load_remoteclip_model(self) -> torch.nn.Module:
        key = "mars_remoteclip"
        if key not in self._models:
            ckpt_path = self.model_paths[key]
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"RemoteCLIP checkpoint missing at '{ckpt_path}'. Run ModelWeightsManager.fetch_all() first."
                )
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
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
        model = self._load_sentinel_model("sentinel_1")
        x = self._prepare_multiband_tensor(sar_patch, expected_channels=2)

        if normalize:
            mean = torch.from_numpy(BEN_V2_S1_MEAN)
            std = torch.from_numpy(BEN_V2_S1_STD)
            x = (x - mean) / std

        x = x.to(self.device)
        backbone = model.vision_encoder
        features = backbone.forward_features(x)
        embeddings = backbone.forward_head(features, pre_logits=True)

        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

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
        model = self._load_sentinel_model("sentinel_2")
        x = self._prepare_multiband_tensor(ms_patch, expected_channels=10)

        if normalize:
            mean = torch.from_numpy(BEN_V2_S2_MEAN)
            std = torch.from_numpy(BEN_V2_S2_STD)
            x = (x - mean) / std

        x = x.to(self.device)
        backbone = model.vision_encoder
        features = backbone.forward_features(x)
        embeddings = backbone.forward_head(features, pre_logits=True)

        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

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
            if image.ndim == 2:
                pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
            else:
                pil_img = Image.fromarray(image[..., :3].astype(np.uint8)).convert("RGB")
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