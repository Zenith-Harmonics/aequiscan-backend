import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


class DiskCacheManager:
    """
    Deterministic, atomic disk cache for raster arrays (.npz) and metadata/embeddings (.json).
    Prevents redundant downloads from Planetary Computer, ASF, and NASA PDS APIs.
    """

    def __init__(self, cache_dir: str | Path = "./.cache_planetary_analog") -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.arrays_dir = self.cache_dir / "arrays"
        self.meta_dir = self.cache_dir / "meta"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def has_arrays(self, key: str) -> bool:
        return (self.arrays_dir / f"{key}.npz").is_file()

    def save_arrays(self, key: str, **arrays: np.ndarray) -> Path:
        target_path = self.arrays_dir / f"{key}.npz"
        tmp_path = self.arrays_dir / f"{key}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, target_path)
        return target_path

    def load_arrays(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        target_path = self.arrays_dir / f"{key}.npz"
        if not target_path.is_file():
            return None
        with np.load(target_path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def has_json(self, key: str) -> bool:
        return (self.meta_dir / f"{key}.json").is_file()

    def save_json(self, key: str, payload: Dict[str, Any]) -> Path:
        target_path = self.meta_dir / f"{key}.json"
        tmp_path = self.meta_dir / f"{key}.{os.getpid()}.tmp.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, target_path)
        return target_path

    def load_json(self, key: str) -> Optional[Dict[str, Any]]:
        target_path = self.meta_dir / f"{key}.json"
        if not target_path.is_file():
            return None
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)