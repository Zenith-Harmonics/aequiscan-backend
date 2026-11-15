from pathlib import Path
from typing import Dict
from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


class ModelWeightsManager:
    """
    Manages local verification and Hugging Face Hub downloading for:
      - BIFOLD BigEarthNet v2.0 ConvNeXtV2-Base (Sentinel-1 SAR & Sentinel-2 Multispectral)
      - RemoteCLIP ViT-L-14 (High-resolution optical imagery for Mars/Moon & Earth RGB)
    """

    def __init__(self, base_dir: str | Path = DEFAULT_WEIGHTS_DIR) -> None:
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
        """Check whether the requested model artifacts exist and are non-empty on disk."""
        if model_key not in self.registry:
            raise ValueError(f"Unknown model key '{model_key}'. Valid keys: {list(self.registry.keys())}")

        spec = self.registry[model_key]
        target_dir: Path = spec["target_dir"]

        if not target_dir.exists():
            return False

        if spec["type"] == "file":
            file_path = target_dir / spec["filename"]
            return file_path.is_file() and file_path.stat().st_size > 0

        if spec["type"] == "snapshot":
            marker_path = target_dir / spec["marker_file"]
            has_weights = any(target_dir.glob("*.safetensors")) or any(target_dir.glob("*.bin"))
            return marker_path.is_file() and has_weights

        return False

    def get_model_path(self, model_key: str) -> Path:
        """Return the expected local path for a model without triggering a download."""
        if model_key not in self.registry:
            raise ValueError(f"Unknown model key '{model_key}'.")
        spec = self.registry[model_key]
        target_dir: Path = spec["target_dir"]
        return (target_dir / spec["filename"]) if spec["type"] == "file" else target_dir

    def fetch_model(
        self,
        model_key: str,
        force_download: bool = False,
        local_files_only: bool = False,
    ) -> Path:
        """Download model weights if missing locally and return the resolved local path."""
        if model_key not in self.registry:
            raise ValueError(f"Unknown model key '{model_key}'.")

        spec = self.registry[model_key]
        target_dir: Path = spec["target_dir"]

        if not force_download and self.is_downloaded(model_key):
            return self.get_model_path(model_key)

        if local_files_only:
            raise FileNotFoundError(
                f"Model '{model_key}' not found in '{target_dir}' and local_files_only=True."
            )

        target_dir.mkdir(parents=True, exist_ok=True)

        if spec["type"] == "file":
            downloaded = hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                local_dir=target_dir,
                force_download=force_download,
            )
            return Path(downloaded)

        if spec["type"] == "snapshot":
            snapshot = snapshot_download(
                repo_id=spec["repo_id"],
                local_dir=target_dir,
                force_download=force_download,
            )
            return Path(snapshot)

        raise RuntimeError(f"Unsupported download type for '{model_key}'.")

    def fetch_all(
        self,
        force_download: bool = False,
        local_files_only: bool = False,
    ) -> Dict[str, Path]:
        """Verify and fetch all configured models, returning a mapping of keys to local paths."""
        return {
            key: self.fetch_model(
                key,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            for key in self.registry
        }