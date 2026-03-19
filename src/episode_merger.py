"""
episode_merger.py — Post-merge pass pour réduire la sur-fragmentation.

Fusionne les micro-épisodes consécutifs qui vérifient simultanément :
    1. taille(B)  ≤ min_size                    (B est un fragment court)
    2. sim(A, B)  ≥ merge_sim                   (sémantiquement proches)
    3. gap(A, B)  ≤ merge_gap_minutes            (temporellement contigus)

Principe : les faux positifs Stage 1 qui ont échappé à Stage 2 créent
des micro-épisodes immédiatement suivis de la suite de l'épisode parent.
Ce post-traitement les absorbe sans toucher aux épisodes bien délimités.

Paramètres tunables par Optuna :
    min_size           : taille max d'un épisode mergeable (défaut 3)
    merge_sim          : similarité cosinus min pour merger (défaut 0.6)
    merge_gap_minutes  : gap max entre A et B pour merger (défaut 60)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from datetime import timedelta
from typing import List, Optional

from models import Episode


class EpisodeMerger:
    """
    Post-merge pass O(n_episodes) — appliqué après segmentation.

    Usage
    -----
        merger = EpisodeMerger(min_size=3, merge_sim=0.6, merge_gap_minutes=60)
        episodes = merger.merge(episodes, embeddings)
    """

    def __init__(self,
                 min_size: int = 3,
                 merge_sim: float = 0.60,
                 merge_gap_minutes: float = 60.0,
                 device: Optional[str] = None):
        """
        Parameters
        ----------
        min_size          : nb max d'artefacts dans B pour être mergeable
        merge_sim         : similarité cosinus min entre centroïdes A et B
        merge_gap_minutes : gap temporel max entre fin de A et début de B
        device            : 'cuda' | 'cpu' | None (auto)
        """
        self.min_size          = min_size
        self.merge_sim         = merge_sim
        self.merge_gap_minutes = timedelta(minutes=merge_gap_minutes)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    # ------------------------------------------------------------------
    # Merge condition
    # ------------------------------------------------------------------

    def _should_merge(self, ep_a: Episode, ep_b: Episode) -> bool:
        """True si B doit être fusionné dans A."""
        # 1. B doit être un micro-épisode
        if len(ep_b.artifact_indices) > self.min_size:
            return False

        # 2. Les deux doivent avoir un centroïde valide
        if ep_a.centroid is None or ep_b.centroid is None:
            return False

        # 3. Similarité sémantique
        a = torch.from_numpy(ep_a.centroid.astype(np.float32)).to(self.device)
        b = torch.from_numpy(ep_b.centroid.astype(np.float32)).to(self.device)
        sim = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        if sim < self.merge_sim:
            return False

        # 4. Contigüité temporelle
        if ep_a.time_interval is None or ep_b.time_interval is None:
            return False
        gap = ep_b.time_interval[0] - ep_a.time_interval[1]
        if gap > self.merge_gap_minutes:
            return False

        return True

    # ------------------------------------------------------------------
    # Merge operation
    # ------------------------------------------------------------------

    @staticmethod
    def _do_merge(ep_a: Episode, ep_b: Episode) -> Episode:
        """
        Fusionne ep_b dans ep_a. Retourne ep_a modifié.

        Centroïde : moyenne pondérée par taille (pas EMA — opération one-shot).
        Temporel  : étend l'intervalle de ep_a jusqu'à la fin de ep_b.
        Entités   : union pondérée.
        """
        n_a = len(ep_a.artifact_indices)
        n_b = len(ep_b.artifact_indices)

        # Artefacts
        ep_a.artifact_indices = sorted(ep_a.artifact_indices + ep_b.artifact_indices)

        # Centroïde — moyenne pondérée par taille
        if ep_a.centroid is not None and ep_b.centroid is not None:
            ep_a.centroid = (n_a * ep_a.centroid + n_b * ep_b.centroid) / (n_a + n_b)

        # Goal centroid
        if ep_a.goal_centroid is not None and ep_b.goal_centroid is not None:
            ep_a.goal_centroid = (n_a * ep_a.goal_centroid + n_b * ep_b.goal_centroid) / (n_a + n_b)

        # Intervalle temporel
        t_start = min(ep_a.time_interval[0], ep_b.time_interval[0])
        t_end   = max(ep_a.time_interval[1], ep_b.time_interval[1])
        ep_a.time_interval = (t_start, t_end)

        # Entités — union pondérée par taille relative
        for ent, w in ep_b.entity_weights.items():
            if ent in ep_a.entity_weights:
                ep_a.entity_weights[ent] = max(ep_a.entity_weights[ent], w)
            else:
                ep_a.entity_weights[ent] = w * n_b / (n_a + n_b)

        return ep_a

    # ------------------------------------------------------------------
    # Main pass
    # ------------------------------------------------------------------

    def merge(self, episodes: List[Episode]) -> List[Episode]:
        """
        Passe itérative de fusion O(n_episodes).

        Parcourt les épisodes dans l'ordre chronologique.
        Dès qu'une fusion est faite, reprend depuis le début de la paire.
        S'arrête quand aucune fusion n'est possible.
        """
        # Trier par ordre chronologique (start_time)
        eps = sorted(
            [ep for ep in episodes if ep.time_interval is not None],
            key=lambda e: e.time_interval[0]
        )

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(eps) - 1:
                if self._should_merge(eps[i], eps[i + 1]):
                    eps[i] = self._do_merge(eps[i], eps[i + 1])
                    eps.pop(i + 1)
                    changed = True
                    # Ne pas incrémenter i — re-tester la nouvelle paire (i, i+1)
                else:
                    i += 1

        return eps
