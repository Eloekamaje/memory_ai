"""
episode_splitter_fast.py — EpisodeSplitter avec silhouette GPU.

Bottleneck identifié : sklearn.silhouette_score = O(n²) CPU.
Ici on remplace par un calcul de similarité cosinus par bloc sur GPU
via PyTorch, puis silhouette analytique.

Device-agnostique : fonctionne sur CUDA (éval) et CPU (inference).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from episode_splitter import EpisodeSplitter, SplitConfig
from models import Artifact, Episode


def _silhouette_cosine_gpu(embeddings: np.ndarray,
                           labels: np.ndarray,
                           device: str) -> float:
    """
    Silhouette score avec distance cosinus, accéléré GPU.

    Complexité : O(n²) en mémoire mais sur GPU → bien plus rapide
    que sklearn CPU pour n < 5000.

    Returns
    -------
    silhouette moyen (float) — -1 si calcul impossible
    """
    n = len(embeddings)
    if n < 2:
        return -1.0

    X = torch.from_numpy(embeddings.astype(np.float32)).to(device)
    X_norm = F.normalize(X, dim=1)                  # (n, d) normalisé

    # Distance cosinus = 1 - similarité cosinus
    cos_sim = X_norm @ X_norm.T                     # (n, n)
    dist = (1.0 - cos_sim).clamp(min=0.0)           # (n, n)

    labels_t = torch.tensor(labels, device=device)
    unique = labels_t.unique()
    n_clusters = len(unique)

    if n_clusters < 2:
        return -1.0

    a = torch.zeros(n, device=device)               # dist intra-cluster moyenne
    b = torch.full((n,), float('inf'), device=device)  # min dist inter-cluster

    for lbl in unique:
        mask = labels_t == lbl
        cluster_dist = dist[:, mask]                # (n, cluster_size)
        cluster_size = mask.sum().item()

        # Intra-cluster : moyenne des distances aux autres membres
        if cluster_size > 1:
            a[mask] = cluster_dist[mask].sum(dim=1) / (cluster_size - 1)
        else:
            a[mask] = 0.0

        # Inter-cluster : moyenne des distances aux membres de CE cluster
        inter_mean = cluster_dist[~mask].mean(dim=1)  # (n - cluster_size,)
        b[~mask] = torch.min(b[~mask], inter_mean)

    # Éviter inf quand un cluster = 1 seul point
    b = b.clamp(max=2.0)

    denom = torch.max(a, b)
    s = torch.where(denom > 0, (b - a) / denom, torch.zeros_like(a))

    return float(s.mean().item())


class EpisodeSplitterFast(EpisodeSplitter):
    """
    EpisodeSplitter avec silhouette GPU.

    Même interface que EpisodeSplitter — drop-in replacement.
    """

    def __init__(self, config: Optional[SplitConfig] = None,
                 device: Optional[str] = None):
        super().__init__(config)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def _find_best_k(
        self, ep_embeddings: np.ndarray,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Même logique que EpisodeSplitter._find_best_k mais avec
        silhouette GPU au lieu de sklearn (O(n²) CPU).
        """
        from sklearn.cluster import AgglomerativeClustering

        n = len(ep_embeddings)
        max_k = min(self.config.max_splits, n // self.config.min_sub_size)
        max_k = max(max_k, 2)

        best_k = 1
        best_labels = np.zeros(n, dtype=int)
        best_sil = -1.0

        for k in range(2, max_k + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=k,
                    metric="cosine",
                    linkage="average",
                )
                labels = clustering.fit_predict(ep_embeddings)

                sizes = np.bincount(labels)
                if np.min(sizes) < self.config.min_sub_size:
                    continue

                # Silhouette GPU — remplace sklearn.silhouette_score
                sil = _silhouette_cosine_gpu(ep_embeddings, labels, self.device)

                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels

            except Exception:
                continue

        return best_k, best_labels, best_sil
