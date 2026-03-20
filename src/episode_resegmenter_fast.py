"""
episode_resegmenter_fast.py — EpisodeResegmenterFast (E-step vectorisé).

Optimisation vs EpisodeResegmenter :
  - E-step : scoring batch torch via segmenter._attach_scores_batch()
             au lieu d'un appel Python par candidat
  - Fallback automatique sur attach_score() si le segmenteur ne supporte
    pas _attach_scores_batch (compatibilité EpisodeSegmenter de base)

Usage :
    reseg = EpisodeResegmenterFast(max_iter=3, window_minutes=480)
    eps   = reseg.resegment(episodes, artifacts, embeddings, segmenter)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from models import Artifact, Episode
from episode_resegmenter import EpisodeResegmenter


class EpisodeResegmenterFast(EpisodeResegmenter):
    """
    Version vectorisée de EpisodeResegmenter.

    L'E-step appelle segmenter._attach_scores_batch(artifact, emb, candidates)
    pour scorer tous les épisodes candidats en une seule passe torch,
    au lieu d'une boucle Python sur attach_score().

    Si le segmenteur ne dispose pas de _attach_scores_batch (ex. EpisodeSegmenter
    de base sans GPU), repli transparent sur la version scalaire.
    """

    # ------------------------------------------------------------------
    # E-step vectorisé
    # ------------------------------------------------------------------

    def _e_step(self,
                eps: List[Episode],
                artifacts: List[Artifact],
                embeddings: np.ndarray,
                segmenter) -> tuple:
        """
        E-step avec scoring batch torch.

        Pour chaque artifact :
            1. Récupère les épisodes candidats dans la fenêtre temporelle
            2. Appelle _attach_scores_batch → vecteur (k,) en un seul appel
            3. Réaffecte à l'épisode de score max si différent du courant

        Returns
        -------
        new_asgn : Dict[int, Episode]
        changed  : int
        """
        # Mapping courant
        old_asgn: Dict[int, Episode] = {}
        for ep in eps:
            for idx in ep.artifact_indices:
                old_asgn[idx] = ep

        new_asgn = dict(old_asgn)
        changed  = 0

        use_batch = hasattr(segmenter, '_attach_scores_batch')

        for i, artifact in enumerate(artifacts):
            current = old_asgn.get(i)
            if current is None:
                continue
            if len(current.artifact_indices) <= self.min_ep_size:
                continue

            candidates = self._temporal_candidates(artifact, eps)
            if not candidates:
                continue

            emb = embeddings[i]

            # ── Batch scoring ────────────────────────────────────────
            if use_batch:
                scores = segmenter._attach_scores_batch(artifact, emb, candidates)
            else:
                scores = np.array([
                    segmenter.attach_score(artifact, emb, ep)
                    for ep in candidates
                ])

            best_idx = int(np.argmax(scores))
            best_ep  = candidates[best_idx]

            # current est toujours dans candidates (fenêtre ⊇ timestamp artifact)
            # → la comparaison max(scores) vs current_score est implicite
            if best_ep is not current:
                new_asgn[i] = best_ep
                changed += 1

        return new_asgn, changed
