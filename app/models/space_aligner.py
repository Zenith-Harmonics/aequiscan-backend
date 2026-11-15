from pathlib import Path
from typing import Optional
import joblib
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge


class LatentSpaceAligner:
    """
    Projects Earth Sentinel-1 / Sentinel-2 embeddings (1024D or fused 2048D)
    into the 768D RemoteCLIP latent space trained on co-registered terrestrial analogs.
    """

    def __init__(
        self,
        method: str = "ridge",
        alpha: float = 10.0,
        n_components: int = 256,
    ) -> None:
        self.method = method
        self.input_dim: Optional[int] = None
        self.target_dim: int = 768

        if method == "ridge":
            self.model = Ridge(alpha=alpha, fit_intercept=True)
        elif method == "cca":
            self.model = CCA(n_components=n_components, max_iter=1000)
        else:
            raise ValueError("Alignment method must be either 'ridge' or 'cca'.")

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.clip(norms, a_min=1e-8, a_max=None)

    def fit(
        self,
        sentinel_embeddings: np.ndarray,
        remoteclip_embeddings: np.ndarray,
    ) -> "LatentSpaceAligner":
        """
        Fit projection weights on N co-registered Earth analog patches:
          - sentinel_embeddings: (N, 1024) or (N, 2048) for concatenated [S1 || S2]
          - remoteclip_embeddings: (N, 768) extracted from optical RGB of the same sites
        """
        x = self._l2_normalize(sentinel_embeddings)
        y = self._l2_normalize(remoteclip_embeddings)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {x.shape[0]}, Y has {y.shape[0]}.")
        if y.shape[1] != self.target_dim:
            raise ValueError(f"Expected RemoteCLIP target dim {self.target_dim}, got {y.shape[1]}.")

        self.input_dim = int(x.shape[1])
        self.model.fit(x, y)
        return self

    def project_earth_to_target_space(self, sentinel_embeddings: np.ndarray) -> np.ndarray:
        """Project Earth Sentinel embeddings into the shared 768D (or CCA) search space."""
        if self.input_dim is None:
            raise RuntimeError("LatentSpaceAligner must be fitted or loaded before projection.")

        x = self._l2_normalize(sentinel_embeddings)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.input_dim}, got {x.shape[1]}."
            )

        if self.method == "ridge":
            projected = self.model.predict(x)
        else:
            projected, _ = self.model.transform(x, np.zeros((x.shape[0], self.target_dim)))
        return self._l2_normalize(projected)

    def transform_target_embedding(self, planetary_clip_embeddings: np.ndarray) -> np.ndarray:
        """Prepare Mars/Moon RemoteCLIP embeddings for cross-domain similarity."""
        y = self._l2_normalize(planetary_clip_embeddings)
        if self.method == "ridge":
            return y
        if self.input_dim is None:
            raise RuntimeError("LatentSpaceAligner must be fitted before CCA transformation.")
        _, y_c = self.model.transform(np.zeros((y.shape[0], self.input_dim)), y)
        return self._l2_normalize(y_c)

    def compute_similarity_matrix(
        self,
        earth_sentinel_emb: np.ndarray,
        planetary_clip_emb: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity matrix (N_earth, N_target) in aligned space."""
        earth_proj = self.project_earth_to_target_space(earth_sentinel_emb)
        target_proj = self.transform_target_embedding(planetary_clip_emb)
        return earth_proj @ target_proj.T

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "LatentSpaceAligner":
        return joblib.load(Path(path))