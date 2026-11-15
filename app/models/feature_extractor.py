from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

from configilm import ConfigILM


class MultiModalFeatureExtractor:
    """
    Loads locally stored Sentinel-1, Sentinel-2 (BIFOLD BigEarthNet v2.0),
    and Mars (RemoteCLIP ViT-L-14) models and extracts feature embeddings.
    """

    def __init__(
        self,
        model_paths: Dict[str, Path],
        device: str | None = None,
    ):
        # Automatically select CUDA if available, otherwise use CPU.
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model_paths = {k: Path(v) for k, v in model_paths.items()}

        # Internal cache for lazy model loading.
        # Only the requested model is loaded into VRAM.
        self._models: Dict[str, torch.nn.Module] = {}
        self._clip_preprocess = None

    def _load_sentinel_model(self, key: str) -> torch.nn.Module:
        """Load a Sentinel model if it is not already cached."""

        if key not in self._models:
            path = self.model_paths[key]

            if not path.exists():
                raise FileNotFoundError(
                    f"Model directory does not exist: {path}"
                )

            model = ConfigILM.from_pretrained(str(path))
            model.to(self.device)
            model.eval()

            self._models[key] = model

        return self._models[key]

    def _load_remoteclip_model(self) -> torch.nn.Module:
        """Load the RemoteCLIP model and its preprocessing pipeline."""

        key = "mars_remoteclip"

        if key not in self._models:
            ckpt_path = self.model_paths[key]

            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"Model checkpoint does not exist: {ckpt_path}"
                )

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14"
            )

            state_dict = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=True,
            )

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
        """
        Convert a (C, H, W) or (B, C, H, W) array/tensor into a
        PyTorch tensor resized to the native BigEarthNet v2.0
        spatial resolution (120x120).

        Args:
            data: Input multispectral or SAR data.
            expected_channels: Expected number of input channels.
            target_size: Target spatial resolution.

        Returns:
            A float32 tensor with shape (B, C, target_size, target_size).
        """

        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        else:
            tensor = data.float()

        # Add a batch dimension if the input is a single sample.
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(
                f"Invalid input dimensions: {tensor.shape}. "
                "Expected a 3D or 4D tensor."
            )

        # Validate the number of input channels.
        if tensor.shape[1] != expected_channels:
            raise ValueError(
                f"Invalid number of channels: received {tensor.shape[1]}, "
                f"expected {expected_channels}."
            )

        # Resize the input if its spatial dimensions do not match
        # the expected model resolution.
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
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract a 1024-dimensional embedding from a Sentinel-1 GRD image.

        Expected input:
            A tensor or NumPy array with 2 channels [VV, VH],
            normalized using z-score normalization.

        Args:
            sar_patch: Input Sentinel-1 image patch.
            l2_normalize: Whether to apply L2 normalization to the embedding.

        Returns:
            NumPy array containing the extracted embedding.
        """

        model = self._load_sentinel_model("sentinel_1")

        x = self._prepare_multiband_tensor(
            sar_patch,
            expected_channels=2,
        ).to(self.device)

        # Extract pre-logits features from the timm ConvNeXtV2 backbone.
        backbone = model.vision_encoder

        features = backbone.forward_features(x)
        embeddings = backbone.forward_head(features, pre_logits=True)

        # Optionally normalize the embeddings to unit length.
        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

    @torch.inference_mode()
    def extract_sentinel2(
        self,
        ms_patch: Union[np.ndarray, torch.Tensor],
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract a 1024-dimensional embedding from a Sentinel-2 L2A image.

        Expected input:
            A tensor or NumPy array with 10 spectral bands:
            [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12].

        Args:
            ms_patch: Input Sentinel-2 multispectral image patch.
            l2_normalize: Whether to apply L2 normalization to the embedding.

        Returns:
            NumPy array containing the extracted embedding.
        """

        model = self._load_sentinel_model("sentinel_2")

        x = self._prepare_multiband_tensor(
            ms_patch,
            expected_channels=10,
        ).to(self.device)

        backbone = model.vision_encoder

        features = backbone.forward_features(x)
        embeddings = backbone.forward_head(features, pre_logits=True)

        # Optionally normalize the embeddings to unit length.
        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

    @torch.inference_mode()
    def extract_mars(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract a 768-dimensional embedding from an optical or orbital
        Mars image (e.g., HiRISE) using RemoteCLIP ViT-L-14.

        Args:
            image: Input image as a file path, PIL image, or NumPy array.
            l2_normalize: Whether to apply L2 normalization to the embedding.

        Returns:
            NumPy array containing the extracted embedding.
        """

        model = self._load_remoteclip_model()

        # Convert the input into an RGB PIL image.
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")

        elif isinstance(image, np.ndarray):
            # Assume HWC format with uint8 values in the [0, 255] range.
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                pil_img = Image.fromarray(
                    image[..., :3].astype(np.uint8)
                ).convert("RGB")

        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")

        else:
            raise TypeError(
                f"Unsupported image input type: {type(image)}"
            )

        # Apply the preprocessing pipeline required by RemoteCLIP.
        tensor = self._clip_preprocess(pil_img).unsqueeze(0).to(self.device)

        # Extract image embeddings using the CLIP image encoder.
        embeddings = model.encode_image(tensor)

        # Optionally normalize the embeddings to unit length.
        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()


if __name__ == "__main__":

    # Resolve the directory containing this script.
    # Model paths are relative to the script location, not the working directory.
    SCRIPT_DIR = Path(__file__).resolve().parent
    WEIGHTS_DIR = SCRIPT_DIR / "weights"

    # Define paths to the locally downloaded model weights.
    # These paths should match the directory structure created
    # by ModelWeightsManager.
    paths = {
        "sentinel_1": WEIGHTS_DIR / "convnextv2_base-s1-v0.2.0",
        "sentinel_2": WEIGHTS_DIR / "convnextv2_base-s2-v0.2.0",
        "mars_remoteclip": (
            WEIGHTS_DIR / "RemoteCLIP" / "RemoteCLIP-ViT-L-14.pt"
        ),
    }

    # Initialize the multimodal feature extractor.
    extractor = MultiModalFeatureExtractor(model_paths=paths)

    print(f"Using device: {extractor.device}")

    # ---------------------------------------------------------
    # 1. Sentinel-1 feature extraction (VV, VH - 2 channels)
    # ---------------------------------------------------------

    dummy_s1 = np.random.randn(2, 120, 120).astype(np.float32)

    emb_s1 = extractor.extract_sentinel1(dummy_s1)

    print(f"Sentinel-1 embedding shape: {emb_s1.shape}")  # (1, 1024)

    # ---------------------------------------------------------
    # 2. Sentinel-2 feature extraction (10 spectral bands)
    # ---------------------------------------------------------

    dummy_s2 = np.random.randn(10, 120, 120).astype(np.float32)

    emb_s2 = extractor.extract_sentinel2(dummy_s2)

    print(f"Sentinel-2 embedding shape: {emb_s2.shape}")  # (1, 1024)

    # ---------------------------------------------------------
    # 3. Mars feature extraction (HiRISE RGB image)
    # ---------------------------------------------------------

    dummy_mars = np.random.randint(
        0,
        255,
        (512, 512, 3),
        dtype=np.uint8,
    )

    emb_mars = extractor.extract_mars(dummy_mars)

    print(f"Mars RemoteCLIP embedding shape: {emb_mars.shape}")  # (1, 768)
