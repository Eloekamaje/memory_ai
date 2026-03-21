"""
episode_segmenter_hybrid.py — HybridEpisodeSegmenter (Stage 1 + 2 + 3).

Architecture en 3 stages :
    Stage 1 — BoundaryDetector  : P(frontière) en O(n) batch GPU
    Stage 2 — AttachScore       : identité épisodique (seulement aux frontières)
    Stage 3 — Lifecycle         : EMA, DORMANT, ACTIVE (inchangé)

Calibration probabiliste (boundary_k) :
    Le seuil de Stage 2 est modulé par la confiance du Stage 1 :

        threshold_eff = attach_threshold + boundary_k × (1 - P_boundary)

    P_boundary = 1.0 → threshold normal     (frontière certaine)
    P_boundary = 0.5 → threshold + k×0.5   (incertain → barre plus haute)
    P_boundary = 0.3 → threshold + k×0.7   (probable faux positif → difficile de fragmenter)

    Cela réduit la sur-fragmentation sans re-tuning et sans dépendre
    de la distribution des données d'entraînement.

Gain : ~8x moins d'appels AttachScore vs EpisodeSegmenterFast.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from boundary_detector import BoundaryDetector, extract_features
from episode_algorithm_fast import EpisodeSegmenterFast
from models import Artifact, Episode


class HybridEpisodeSegmenter(EpisodeSegmenterFast):
    """
    EpisodeSegmenter hybride en 3 stages avec calibration probabiliste.

    Stage 1 : BoundaryDetector → P(frontière) pour chaque message
    Stage 2 : AttachScore avec seuil adaptatif selon P(frontière)
    Stage 3 : Lifecycle (EMA, DORMANT/ACTIVE) — inchangé

    Le paramètre `boundary_k` contrôle combien l'incertitude du Stage 1
    durcit le seuil Stage 2. k=0 = comportement binaire original.
    """

    def __init__(self,
                 detector: BoundaryDetector,
                 boundary_k: float = 0.3,
                 bd_threshold: Optional[float] = None,
                 device: Optional[str] = None,
                 **kwargs):
        """
        Parameters
        ----------
        detector     : BoundaryDetector entraîné (Stage 1)
        boundary_k   : facteur de calibration probabiliste [0, 1].
                       0   = seuil fixe (pas de calibration)
                       0.3 = défaut recommandé
                       1.0 = calibration maximale
        bd_threshold : seuil de classification Stage 1 [0, 1].
                       None = utilise detector.threshold (défaut entraîné).
                       Tunable par Optuna — permet de co-optimiser précision/rappel
                       Stage 1 directement sur ARI, sans contrainte MIN_RECALL.
        device       : 'cuda' | 'cpu' | None (auto)
        **kwargs     : tous les paramètres de EpisodeSegmenter
                       (attach_threshold, ema_alpha, dormancy_minutes, …)
        """
        super().__init__(device=device, **kwargs)
        self.detector     = detector
        self.boundary_k   = boundary_k
        self.bd_threshold = bd_threshold  # None → utilise detector.threshold

    # ------------------------------------------------------------------
    # Stage 1 — probabilités continues
    # ------------------------------------------------------------------

    def _boundary_probs(self,
                        artifacts: List[Artifact],
                        embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retourne (mask, probs) :
            mask  : (n,) bool  — True si message est candidat frontière
            probs : (n,) float — P(frontière) pour chaque message

        Le premier message est toujours frontière (prob=1.0).
        """
        # TCNBoundaryDetector expose predict_proba_sequence (contexte séquentiel).
        # MLP/BoundaryDetector utilise predict_proba(X) — rétro-compatible.
        if hasattr(self.detector, 'predict_proba_sequence'):
            probs = self.detector.predict_proba_sequence(embeddings, artifacts)
        else:
            X     = extract_features(embeddings, artifacts)
            probs = self.detector.predict_proba(X)       # (n,) float

        thr   = self.bd_threshold if self.bd_threshold is not None else self.detector.threshold
        mask  = probs >= thr
        mask[0]   = True
        probs[0]  = 1.0  # premier message — certitude maximale

        return mask, probs

    # ------------------------------------------------------------------
    # Segment — override principal
    # ------------------------------------------------------------------

    def _best_candidate(self,
                        artifact: Artifact,
                        emb: np.ndarray,
                        episodes: List[Episode],
                        threshold_eff: float) -> Optional[Episode]:
        """
        Retourne l'épisode candidat si son score dépasse threshold_eff, None sinon.
        Applique aussi le hard break check.
        """
        candidates = self._get_candidates(artifact, episodes)
        if not candidates:
            return None

        scores     = self._attach_scores_batch(artifact, emb, candidates)
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_ep    = candidates[best_idx]

        if (self.hard_break_minutes is not None
                and best_ep.time_interval is not None):
            _, t_end = best_ep.time_interval
            if artifact.timestamp - t_end > self.hard_break_minutes:
                return None

        return best_ep if best_score >= threshold_eff else None

    def _new_episode(self,
                     counter: int,
                     artifact: Artifact,
                     idx: int,
                     emb: np.ndarray,
                     episodes: List[Episode]) -> Tuple[int, Episode]:
        """Crée un nouvel épisode et l'enregistre dans l'index."""
        counter += 1
        ep = self._create_episode(counter, artifact, idx, emb)
        episodes.append(ep)
        self._idx_add(ep)
        return counter, ep

    def segment(self,
                artifacts: List[Artifact],
                embeddings: np.ndarray) -> List[Episode]:
        """
        Segmente le flux en 3 stages avec calibration probabiliste.

        Pour chaque message candidat frontière :
            threshold_eff = attach_threshold + boundary_k × (1 - P_boundary)

        Faux positifs Stage 1 (P faible) → seuil Stage 2 élevé → rattachés
        à l'épisode existant plutôt que fragmentés inutilement.
        """
        episodes: List[Episode] = []
        episode_counter = 0
        self.reactivation_count = 0
        self._idx_init()

        boundary_mask, boundary_probs = self._boundary_probs(artifacts, embeddings)
        current_episode: Optional[Episode] = None

        for i, artifact in enumerate(artifacts):
            emb = embeddings[i]
            self._apply_aging(artifact.timestamp, episodes)

            if boundary_mask[i]:
                # Stage 2 — seuil adaptatif selon confiance Stage 1 + pénalité création
                base_thr = self._creation_threshold(episodes)
                threshold_eff = base_thr + self.boundary_k * (1.0 - float(boundary_probs[i]))
                best_ep = self._best_candidate(artifact, emb, episodes, threshold_eff)
                if best_ep is not None:
                    self._update_episode(best_ep, artifact, i, emb)
                    current_episode = best_ep
                else:
                    episode_counter, current_episode = self._new_episode(
                        episode_counter, artifact, i, emb, episodes)
            else:
                # Continuation directe — pas de scoring
                if current_episode is None:
                    episode_counter, current_episode = self._new_episode(
                        episode_counter, artifact, i, emb, episodes)
                else:
                    self._update_episode(current_episode, artifact, i, emb)

        return episodes

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def boundary_stats(self,
                       artifacts: List[Artifact],
                       embeddings: np.ndarray) -> dict:
        """Statistiques Stage 1 + calibration pour diagnostic."""
        mask, probs = self._boundary_probs(artifacts, embeddings)
        n_total      = len(artifacts)
        n_boundaries = int(mask.sum())
        p_at_boundaries = probs[mask]
        return {
            'n_total':           n_total,
            'n_boundaries':      n_boundaries,
            'boundary_rate':     n_boundaries / max(n_total, 1),
            'threshold_used':    self.bd_threshold if self.bd_threshold is not None else self.detector.threshold,
            'boundary_k':        self.boundary_k,
            'mean_prob':         float(p_at_boundaries.mean()),
            'p25_prob':          float(np.percentile(p_at_boundaries, 25)),
            'effective_thr_p25': self.attach_threshold + self.boundary_k * (1 - float(np.percentile(p_at_boundaries, 25))),
            'effective_thr_p75': self.attach_threshold + self.boundary_k * (1 - float(np.percentile(p_at_boundaries, 75))),
        }
