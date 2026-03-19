"""
episode_splitter_fast.py — EpisodeSplitter O(n·k) sur GPU/CPU.

Remplacement algorithmique du silhouette O(n²) par Calinski-Harabasz O(n·k).

Calinski-Harabasz (Variance Ratio Criterion) :
    CH(k) = [BGSS / (k-1)] / [WGSS / (n-k)]

    BGSS = Between-Group Sum of Squares  (variance inter-cluster)
    WGSS = Within-Group Sum of Squares   (variance intra-cluster)

    Complexité : O(n·k) — pas de distances pairwise, pas de matrice n×n.
    Plus CH est élevé, meilleur est le clustering.

Sur embeddings normalisés (cosinus) : les distances euclidiennes sont
proportionnelles aux distances cosinus → CH s'applique directement.

Device-agnostique : cuda pour l'éval, cpu pour l'inference.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from episode_splitter import EpisodeSplitter, SplitConfig
from models import Artifact, Episode


def _calinski_harabasz_gpu(embeddings: np.ndarray,
                           labels: np.ndarray,
                           device: str) -> float:
    """
    Calinski-Harabasz Index — O(n·k), GPU/CPU.

    Paramètres
    ----------
    embeddings : (n, d) float32
    labels     : (n,)  int — cluster ids 0..k-1
    device     : 'cuda' ou 'cpu'

    Retourne
    --------
    score CH (float) — plus élevé = meilleur clustering
    0.0 si calcul impossible (n == k ou k < 2)
    """
    n, _ = embeddings.shape
    unique = np.unique(labels)
    k = len(unique)

    if k < 2 or n <= k:
        return 0.0

    X = F.normalize(
        torch.from_numpy(embeddings.astype(np.float32)).to(device),
        dim=1
    )  # (n, d) — normalisé pour distance cosinus ≡ distance euclidienne

    overall_centroid = X.mean(dim=0)  # (d,)

    bgss = torch.tensor(0.0, device=device)  # Between-Group Sum of Squares
    wgss = torch.tensor(0.0, device=device)  # Within-Group Sum of Squares

    labels_t = torch.from_numpy(labels.astype(np.int64)).to(device)

    for lbl in unique:
        mask = labels_t == int(lbl)
        cluster = X[mask]          # (nk, d)
        nk = cluster.shape[0]

        if nk == 0:
            continue

        centroid_k = cluster.mean(dim=0)  # (d,)

        # Inter : nk · ||centroid_k − centroid_global||²
        bgss = bgss + nk * ((centroid_k - overall_centroid) ** 2).sum()

        # Intra : Σ ||x − centroid_k||²
        wgss = wgss + ((cluster - centroid_k.unsqueeze(0)) ** 2).sum()

    if wgss.item() < 1e-10:
        return 0.0

    ch = (bgss / (k - 1)) / (wgss / (n - k))
    return float(ch.item())


class EpisodeSplitterFast(EpisodeSplitter):
    """
    EpisodeSplitter avec Calinski-Harabasz O(n·k) au lieu de silhouette O(n²).

    Drop-in replacement — même interface que EpisodeSplitter.
    """

    def __init__(self, config: Optional[SplitConfig] = None,
                 device: Optional[str] = None,
                 quality_threshold: float = 0.10):
        """
        Parameters
        ----------
        config            : SplitConfig — silhouette_threshold ignoré ici
        device            : 'cuda' | 'cpu' | None (auto-détecté)
        quality_threshold : seuil sur le gap relatif CH normalisé [0, 1].
                            Un split est validé si (CH_best - CH_2nd) / CH_best
                            dépasse ce seuil. 0.10 = au moins 10% d'écart.
        """
        super().__init__(config)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.quality_threshold = quality_threshold  # remplace silhouette_threshold

    def _find_best_k(
        self, ep_embeddings: np.ndarray,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Trouve le meilleur k via Calinski-Harabasz O(n·k) sur GPU.

        Retourne (best_k, labels, gap_relatif_normalisé)
        Le gap est comparé à self.quality_threshold (pas silhouette_threshold).
        """
        from sklearn.cluster import AgglomerativeClustering

        n = len(ep_embeddings)
        max_k = min(self.config.max_splits, n // self.config.min_sub_size)
        max_k = max(max_k, 2)

        best_k = 1
        best_labels = np.zeros(n, dtype=int)
        best_ch = -1.0
        ch_scores = []

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
                    ch_scores.append(-1.0)
                    continue

                ch = _calinski_harabasz_gpu(ep_embeddings, labels, self.device)
                ch_scores.append(ch)

                if ch > best_ch:
                    best_ch = ch
                    best_k = k
                    best_labels = labels

            except Exception:
                ch_scores.append(-1.0)
                continue

        # Gap relatif normalisé [0, 1] — comparé à self.quality_threshold.
        # CH n'a pas de borne supérieure → on mesure l'écart relatif
        # entre le meilleur k et le 2e meilleur : grand écart = split justifié.
        if best_ch <= 0:
            return 1, np.zeros(n, dtype=int), 0.0

        valid = [s for s in ch_scores if s > 0]
        # Score normalisé : 1.0 si le meilleur k est clairement meilleur
        # Utilise l'écart relatif entre le best et le 2e meilleur
        if len(valid) >= 2:
            sorted_valid = sorted(valid, reverse=True)
            gap = (sorted_valid[0] - sorted_valid[1]) / (sorted_valid[0] + 1e-8)
            normalized = gap  # 0..1 — grand gap = split clairement justifié
        else:
            normalized = 0.0

        return best_k, best_labels, normalized

    def split(self, episodes, artifacts, embeddings, verbose=False):
        """Override : utilise quality_threshold au lieu de silhouette_threshold."""
        # Patch temporaire du config pour que le parent utilise notre seuil
        original = self.config.silhouette_threshold
        self.config.silhouette_threshold = self.quality_threshold
        result = super().split(episodes, artifacts, embeddings, verbose=verbose)
        self.config.silhouette_threshold = original
        return result
