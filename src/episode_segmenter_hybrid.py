"""
episode_segmenter_hybrid.py — HybridEpisodeSegmenter (Stage 1 + 2 + 3).

Architecture en 3 stages :
    Stage 1 — BoundaryDetector  : P(frontière) en O(n) batch GPU
    Stage 2 — AttachScore       : identité épisodique (seulement aux frontières)
    Stage 3 — Lifecycle         : EMA, DORMANT, ACTIVE (inchangé)

Gain : ~18x moins d'appels AttachScore vs EpisodeSegmenterFast.

Usage
-----
    from boundary_detector import BoundaryDetector, extract_features, extract_labels
    from episode_segmenter_hybrid import HybridEpisodeSegmenter

    # 1. Entraîner le boundary detector (sur tune)
    detector = BoundaryDetector()
    X_tune = extract_features(emb_tune, arts_tune)
    y_tune = extract_labels(y_true_tune, len(arts_tune))
    detector.fit(X_tune, y_tune)
    detector.optimize_threshold(X_tune, y_tune)
    detector.save('boundary_detector.pt')

    # 2. Segmenter (sur n'importe quel corpus)
    seg = HybridEpisodeSegmenter(detector=detector, **params)
    episodes = seg.segment(artifacts, embeddings)
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from boundary_detector import BoundaryDetector, extract_features
from episode_algorithm_fast import EpisodeSegmenterFast
from models import Artifact, Episode


class HybridEpisodeSegmenter(EpisodeSegmenterFast):
    """
    EpisodeSegmenter hybride en 3 stages.

    Stage 1 : BoundaryDetector filtre O(n) → seules les ~5% frontières
              détectées passent au Stage 2.
    Stage 2 : AttachScore vectorisé (hérité de EpisodeSegmenterFast).
    Stage 3 : Lifecycle (EMA, DORMANT/ACTIVE) — hérité de EpisodeSegmenter.

    Pour les messages non-frontières : rattachement direct à l'épisode
    courant, sans aucun calcul de score.
    """

    def __init__(self,
                 detector: BoundaryDetector,
                 device: Optional[str] = None,
                 **kwargs):
        """
        Parameters
        ----------
        detector : BoundaryDetector entraîné (Stage 1)
        device   : 'cuda' | 'cpu' | None (auto)
        **kwargs : tous les paramètres de EpisodeSegmenter
                   (attach_threshold, ema_alpha, dormancy_minutes, …)
        """
        super().__init__(device=device, **kwargs)
        self.detector = detector

    # ------------------------------------------------------------------
    # Stage 1 helper
    # ------------------------------------------------------------------

    def _predict_boundaries(self,
                            artifacts: List[Artifact],
                            embeddings: np.ndarray) -> np.ndarray:
        """
        Retourne un masque booléen (n,) : True = frontière détectée.

        Le premier message est toujours une frontière (crée le 1er épisode).
        """
        X = extract_features(embeddings, artifacts)
        boundary_indices = self.detector.predict(X)
        mask = np.zeros(len(artifacts), dtype=bool)
        mask[boundary_indices] = True
        mask[0] = True  # premier message toujours frontière
        return mask

    # ------------------------------------------------------------------
    # Segment — override principal
    # ------------------------------------------------------------------

    def segment(self,
                artifacts: List[Artifact],
                embeddings: np.ndarray) -> List[Episode]:
        """
        Segmente le flux en 3 stages.

        Stage 1 : boundaries = BoundaryDetector.predict()  — O(n) batch
        Stage 2 : pour chaque frontière → AttachScore vectorisé
        Stage 3 : lifecycle (aging, EMA, DORMANT/ACTIVE) à chaque message
        """
        episodes: List[Episode] = []
        episode_counter = 0
        self.reactivation_count = 0
        self._idx_init()

        # ── Stage 1 : toutes les frontières en un seul appel ──────────
        boundary_mask = self._predict_boundaries(artifacts, embeddings)

        n_boundaries = int(boundary_mask.sum())
        n_total = len(artifacts)

        current_episode: Optional[Episode] = None

        for i, artifact in enumerate(artifacts):
            emb = embeddings[i]

            # Lifecycle : aging (toujours appliqué)
            self._apply_aging(artifact.timestamp, episodes)

            if boundary_mask[i]:
                # ── Stage 2 : AttachScore seulement ici ───────────────
                candidates = self._get_candidates(artifact, episodes)

                if candidates:
                    scores = self._attach_scores_batch(artifact, emb, candidates)
                    best_idx   = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    best_ep    = candidates[best_idx]
                else:
                    best_score = -1.0
                    best_ep    = None

                # hard break check
                if (best_ep is not None
                        and self.hard_break_minutes is not None
                        and best_ep.time_interval is not None):
                    _, t_end = best_ep.time_interval
                    if artifact.timestamp - t_end > self.hard_break_minutes:
                        best_score = -1.0
                        best_ep    = None

                if best_score >= self.attach_threshold and best_ep is not None:
                    # Réactivation d'un épisode existant
                    self._update_episode(best_ep, artifact, i, emb)
                    current_episode = best_ep
                else:
                    # Nouvel épisode
                    episode_counter += 1
                    new_ep = self._create_episode(episode_counter, artifact, i, emb)
                    episodes.append(new_ep)
                    self._idx_add(new_ep)
                    current_episode = new_ep

            else:
                # ── Continuation : rattachement direct (pas de scoring) ─
                if current_episode is not None:
                    self._update_episode(current_episode, artifact, i, emb)
                else:
                    # Cas de bord : pas d'épisode courant (ne devrait pas arriver
                    # car mask[0]=True, mais sécurité)
                    episode_counter += 1
                    new_ep = self._create_episode(episode_counter, artifact, i, emb)
                    episodes.append(new_ep)
                    self._idx_add(new_ep)
                    current_episode = new_ep

        return episodes

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def boundary_stats(self,
                       artifacts: List[Artifact],
                       embeddings: np.ndarray) -> dict:
        """
        Retourne les statistiques du Stage 1 (pour diagnostic).

        Returns
        -------
        dict avec :
            n_total        : nombre total de messages
            n_boundaries   : frontières détectées
            boundary_rate  : taux de frontières
            threshold_used : seuil de classification utilisé
        """
        mask = self._predict_boundaries(artifacts, embeddings)
        n_total = len(artifacts)
        n_boundaries = int(mask.sum())
        return {
            'n_total':       n_total,
            'n_boundaries':  n_boundaries,
            'boundary_rate': n_boundaries / max(n_total, 1),
            'threshold_used': self.detector.threshold,
        }
