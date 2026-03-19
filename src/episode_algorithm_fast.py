"""
episode_algorithm_fast.py — EpisodeSegmenter vectorisé (GPU/CPU).

Optimisations vs episode_algorithm.py :
  - AttachScore vectorisé sur tous les candidats en un seul appel torch
  - Pas de sklearn.cosine_similarity en boucle (1×1 → catastrophique)
  - Device-agnostique : 'cuda' si disponible, 'cpu' sinon
  - Compatible CPU pour l'inference (même interface que EpisodeSegmenter)

Usage :
    from episode_algorithm_fast import EpisodeSegmenterFast
    seg = EpisodeSegmenterFast(device='cuda', **params)
    # Même API que EpisodeSegmenter
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional

from episode_algorithm import EpisodeSegmenter
from models import Artifact, Episode, EpisodeState


class EpisodeSegmenterFast(EpisodeSegmenter):
    """
    Sous-classe de EpisodeSegmenter avec AttachScore vectorisé.

    Le bottleneck identifié : sklearn.cosine_similarity appelé une paire
    à la fois dans une boucle Python sur les candidats. Sur 8954 messages
    avec ~k candidats actifs, ça représente des millions d'appels sklearn.

    Ici : on empile les centroïdes candidats en matrice (k, d) et on fait
    une seule opération F.cosine_similarity — 10-100x plus rapide.
    """

    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    # ------------------------------------------------------------------
    # AttachScore vectorisé — remplace la boucle for ep in candidates
    # ------------------------------------------------------------------

    def _attach_scores_batch(self,
                             artifact: Artifact,
                             emb: np.ndarray,
                             candidates: List[Episode]) -> np.ndarray:
        """
        Calcule l'AttachScore pour tous les candidats en une seule passe.

        Returns
        -------
        scores : np.ndarray (k,) — AttachScore pour chaque candidat
        """
        k = len(candidates)
        if k == 0:
            return np.array([])

        # ── Semantic similarity — dual centroid ─────────────────────
        # sem = max(sim(emb, centroid_ema), sim(emb, centroid_init))
        # centroid_ema  : dérive avec l'épisode (contexte récent)
        # centroid_init : gelé au 1er message (identité fondatrice)
        centroids_ema = np.stack([
            ep.centroid if ep.centroid is not None else np.zeros_like(emb)
            for ep in candidates
        ])  # (k, d)
        emb_t           = torch.from_numpy(emb.astype(np.float32)).to(self.device)             # (d,)
        centroids_ema_t = torch.from_numpy(centroids_ema.astype(np.float32)).to(self.device)  # (k, d)
        emb_exp = emb_t.unsqueeze(0).expand(k, -1)                                            # (k, d)

        sim_ema = F.cosine_similarity(emb_exp, centroids_ema_t, dim=1).clamp(min=0.0)  # (k,)

        if self.use_dual_centroid:
            centroids_init = np.stack([
                getattr(ep, 'centroid_init', None) if getattr(ep, 'centroid_init', None) is not None
                else (ep.centroid if ep.centroid is not None else np.zeros_like(emb))
                for ep in candidates
            ])  # (k, d)
            centroids_init_t = torch.from_numpy(centroids_init.astype(np.float32)).to(self.device)
            sim_init = F.cosine_similarity(emb_exp, centroids_init_t, dim=1).clamp(min=0.0)
            sem = torch.maximum(sim_ema, sim_init).cpu().numpy()
        else:
            sem = sim_ema.cpu().numpy()

        # Masque : épisodes sans centroïde → sem = 0
        no_centroid = np.array([ep.centroid is None for ep in candidates])
        sem[no_centroid] = 0.0

        # ── Temporal proximity ──────────────────────────────────────
        t_gaps = np.array([
            abs((artifact.timestamp - ep.time_interval[1]).total_seconds() / 60.0)
            if ep.time_interval else 0.0
            for ep in candidates
        ])
        temp = np.exp(-self._lambda * t_gaps)  # (k,)

        # ── Goal similarity ─────────────────────────────────────────
        if artifact.goal_vector is not None:
            goal_vecs = np.stack([
                ep.goal_centroid if ep.goal_centroid is not None
                else np.zeros_like(artifact.goal_vector)
                for ep in candidates
            ])  # (k, g)
            has_goal = np.array([ep.goal_centroid is not None for ep in candidates], dtype=np.float32)

            gv_t  = torch.from_numpy(artifact.goal_vector.astype(np.float32)).to(self.device)  # (g,)
            gvec_t = torch.from_numpy(goal_vecs.astype(np.float32)).to(self.device)              # (k, g)

            goal = F.cosine_similarity(
                gv_t.unsqueeze(0).expand(k, -1),
                gvec_t,
                dim=1
            ).clamp(min=0.0).cpu().numpy() * has_goal  # (k,)
        else:
            goal = np.zeros(k, dtype=np.float32)

        # ── Entity overlap ──────────────────────────────────────────
        # Set operations — pas vectorisable simplement, reste Python
        # mais représente un temps << semantic similarity
        ent = np.array([self._entity_overlap(artifact, ep) for ep in candidates])

        # ── Age penalty ─────────────────────────────────────────────
        # Dépend du state (DORMANT/ACTIVE) — reste Python
        age = np.array([self._age_penalty(artifact, ep) for ep in candidates])

        # ── Score final ─────────────────────────────────────────────
        scores = (self.alpha * sem
                  + self.beta * ent
                  + self.gamma * temp
                  + self.delta * goal
                  - self.rho * age)

        return np.maximum(scores, 0.0)

    # ------------------------------------------------------------------
    # Override de la boucle de scoring dans segment()
    # ------------------------------------------------------------------

    def segment(self, artifacts: List[Artifact],
                embeddings: np.ndarray) -> List[Episode]:
        """
        Segmente le flux — même logique que EpisodeSegmenter.segment()
        mais avec AttachScore vectorisé sur les candidats.
        """
        episodes: List[Episode] = []
        episode_counter = 0
        self.reactivation_count = 0
        self._idx_init()

        for i, artifact in enumerate(artifacts):
            emb = embeddings[i]

            self._apply_aging(artifact.timestamp, episodes)
            candidates = self._get_candidates(artifact, episodes)

            if candidates:
                # Vectorisé — un seul appel torch pour tous les candidats
                scores = self._attach_scores_batch(artifact, emb, candidates)
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                best_episode = candidates[best_idx]
            else:
                best_score = -1.0
                best_episode = None

            # hard break
            if (best_episode is not None
                    and self.hard_break_minutes is not None
                    and best_episode.time_interval is not None):
                _, t_end = best_episode.time_interval
                if artifact.timestamp - t_end > self.hard_break_minutes:
                    best_score = -1.0
                    best_episode = None

            create_thr = self._creation_threshold(episodes)
            if best_score >= create_thr and best_episode is not None:
                self._update_episode(best_episode, artifact, i, emb)
            else:
                episode_counter += 1
                new_ep = self._create_episode(episode_counter, artifact, i, emb)
                episodes.append(new_ep)
                self._idx_add(new_ep)

        return episodes
