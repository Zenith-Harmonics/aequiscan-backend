import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import CCA


class LatentSpaceAligner:
    """
    Aligns the Sentinel-1 / Sentinel-2 latent space (1024D)
    with the RemoteCLIP latent space (768D) using co-registered
    image pairs collected on Earth.
    """

    def __init__(self, method: str = "ridge", alpha: float = 10.0, n_components: int = 256):
        """
        method: 'ridge' (direct projection from 1024D to 768D)
                or 'cca' (shared latent space with n_components dimensions).
        alpha: L2 regularization parameter for Ridge regression
               (helps prevent overfitting on small datasets).
        """
        self.method = method
        if method == "ridge":
            self.model = Ridge(alpha=alpha, fit_intercept=True)
        elif method == "cca":
            self.model = CCA(n_components=n_components, max_iter=1000)
        else:
            raise ValueError("Method must be either 'ridge' or 'cca'.")

    @staticmethod
    def _l2_norm(x: np.ndarray) -> np.ndarray:
        """Applies L2 normalization to the input embeddings."""
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(norms, a_min=1e-8, a_max=None)

    def fit(self, sentinel_embeddings: np.ndarray, remoteclip_embeddings: np.ndarray):
        """
        Trains the projection using N co-registered Earth locations.

        Args:
            sentinel_embeddings: Array of shape (N, 1024), or (N, 2048)
                                 if Sentinel-1 and Sentinel-2 embeddings
                                 are concatenated.
            remoteclip_embeddings: Array of shape (N, 768), extracted
                                   from RGB images of the same locations.
        """
        X = self._l2_norm(sentinel_embeddings)
        Y = self._l2_norm(remoteclip_embeddings)
        self.model.fit(X, Y)
        return self

    def project_earth_to_mars_space(self, sentinel_embeddings: np.ndarray) -> np.ndarray:
        """Projects a Sentinel-1/2 embedding into the Mars search space."""
        X = self._l2_norm(sentinel_embeddings)
        if self.method == "ridge":
            projected = self.model.predict(X)
        else:
            projected, _ = self.model.transform(X, np.zeros((X.shape[0], 768)))
        return self._l2_norm(projected)

    def transform_mars_embedding(self, mars_embeddings: np.ndarray) -> np.ndarray:
        """Prepares HiRISE/Mars embeddings for similarity comparison."""
        Y = self._l2_norm(mars_embeddings)
        if self.method == "ridge":
            return Y

        # In CCA mode, Mars embeddings are also projected
        # into the shared latent subspace.
        _, y_c = self.model.transform(
            np.zeros((Y.shape[0], self.model.n_features_in_)), Y
        )
        return self._l2_norm(y_c)

    def compute_similarity(
        self,
        earth_sentinel_emb: np.ndarray,
        mars_clip_emb: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the cosine similarity matrix of shape (N_earth, N_mars)
        between Sentinel-1/2 patches from Earth and HiRISE patches from Mars.

        Args:
            earth_sentinel_emb: Sentinel-1/2 embeddings from Earth.
            mars_clip_emb: RemoteCLIP embeddings extracted from Mars imagery.

        Returns:
            A cosine similarity matrix of shape (N_earth, N_mars).
        """
        earth_proj = self.project_earth_to_mars_space(earth_sentinel_emb)
        mars_proj = self.transform_mars_embedding(mars_clip_emb)
        return earth_proj @ mars_proj.T

    def save(self, path: str | Path):
        """Saves the trained aligner to disk."""
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "LatentSpaceAligner":
        """Loads a previously saved aligner from disk."""
        return joblib.load(Path(path))
