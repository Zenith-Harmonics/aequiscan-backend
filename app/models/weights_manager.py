from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_download, snapshot_download


# Resolve the directory containing this script.
# All model weights will be stored relative to this location,
# regardless of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"


class ModelWeightsManager:
    """Manages downloading, storing, and checking model weights."""

    def __init__(self, base_dir: str | Path = WEIGHTS_DIR):
        # Initialize the base directory for all model weights.
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Registry containing the configuration of all supported models.
        self.registry = {
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
        """Check whether a model has already been downloaded locally."""

        if model_key not in self.registry:
            raise ValueError(
                f"Unknown model: '{model_key}'. "
                f"Available models: {list(self.registry.keys())}"
            )

        spec = self.registry[model_key]
        target_dir: Path = spec["target_dir"]

        # The target directory must exist.
        if not target_dir.exists():
            return False

        # For single-file models, check whether the file exists and is not empty.
        if spec["type"] == "file":
            file_path = target_dir / spec["filename"]
            return file_path.is_file() and file_path.stat().st_size > 0

        # For snapshot models, check for the configuration file
        # and at least one model weights file.
        if spec["type"] == "snapshot":
            marker_path = target_dir / spec["marker_file"]

            has_weights = (
                any(target_dir.glob("*.safetensors"))
                or any(target_dir.glob("*.bin"))
            )

            return marker_path.is_file() and has_weights

        return False

    def fetch_model(
        self,
        model_key: str,
        force_download: bool = False,
    ) -> Path:
        """
        Download a model if it is not already available locally.

        Args:
            model_key: Identifier of the model to download.
            force_download: Whether to force a new download.

        Returns:
            Path to the downloaded model or model directory.
        """

        if model_key not in self.registry:
            raise ValueError(f"Unknown model: '{model_key}'.")

        spec = self.registry[model_key]
        target_dir: Path = spec["target_dir"]

        # Skip downloading if the model is already available locally.
        if not force_download and self.is_downloaded(model_key):
            print(f"[CACHE HIT] '{model_key}' is already available at: {target_dir}")

            if spec["type"] == "file":
                return target_dir / spec["filename"]

            return target_dir

        print(
            f"[DOWNLOADING] Downloading '{model_key}' "
            f"from {spec['repo_id']}..."
        )

        # Ensure the target directory exists before downloading.
        target_dir.mkdir(parents=True, exist_ok=True)

        # Download a single model file.
        if spec["type"] == "file":
            downloaded_path = hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                local_dir=target_dir,
                force_download=force_download,
            )

            return Path(downloaded_path)

        # Download the entire model repository snapshot.
        if spec["type"] == "snapshot":
            snapshot_path = snapshot_download(
                repo_id=spec["repo_id"],
                local_dir=target_dir,
                force_download=force_download,
            )

            return Path(snapshot_path)

        raise RuntimeError(
            f"Unsupported download type for model: {model_key}"
        )

    def fetch_all(self, force_download: bool = False) -> Dict[str, Path]:
        """
        Download all registered models.

        Existing models are skipped unless force_download is enabled.

        Returns:
            Dictionary mapping model identifiers to their local paths.
        """

        model_paths = {}

        for model_key in self.registry:
            model_paths[model_key] = self.fetch_model(
                model_key,
                force_download=force_download,
            )

        return model_paths


if __name__ == "__main__":

    # Initialize the manager using the default weights directory.
    # The directory is relative to this script, not the current working directory.
    manager = ModelWeightsManager()

    print(f"Weights directory: {manager.base_dir}")

    # Check whether Sentinel-1 weights are already available.
    print(
        "Initial Sentinel-1 status:",
        manager.is_downloaded("sentinel_1"),
    )

    # Download all registered models.
    # Models that are already available locally will be skipped.
    model_paths = manager.fetch_all()

    # Display the final paths to all downloaded models.
    print("\nFinal model paths:")

    for model_name, model_path in model_paths.items():
        print(f" - {model_name}: {model_path}")
