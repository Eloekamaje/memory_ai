"""
episode_resegmenter.py — Re-segmentation EM-style (post-processing greedy).

Principe
--------
La segmentation greedy commet des erreurs locales en cascade :
un artifact mal assigné prive son épisode réel → dérive du centroïde →
les suivants ne peuvent plus s'attacher → fragmentation.

Ce module corrige ça en itérant :
    E-step : réaffecte chaque artifact à l'épisode qui maximise AttachScore
             parmi les épisodes dans sa fenêtre temporelle (candidats locaux).
    M-step : recompute centroïdes (mean), time_intervals, entity_weights
             à partir des nouvelles assignations.

Typiquement converge en 2-3 itérations.

Usage
-----
    reseg = EpisodeResegmenter(max_iter=3, window_minutes=480, min_ep_size=2)
    episodes = reseg.resegment(episodes, artifacts, embeddings, segmenter)
"""

from __future__ import annotations

import numpy as np
from datetime import timedelta
from typing import Dict, List, Optional

from models import Artifact, Episode, EpisodeState


class EpisodeResegmenter:
    """
    Post-processing EM-style après segmentation greedy.

    Parameters
    ----------
    max_iter       : nombre max d'itérations E+M. Arrêt anticipé si convergence.
    window_minutes : fenêtre temporelle pour les épisodes candidats d'un artifact.
                     Doit correspondre à recent_window du segmenteur (défaut 480 min).
    min_ep_size    : taille min d'un épisode pour qu'on puisse lui retirer un artifact.
                     Protège les épisodes d'1 artifact contre le vidage.
    """

    def __init__(self,
                 max_iter: int = 3,
                 window_minutes: float = 480.0,
                 min_ep_size: int = 2):
        self.max_iter    = max_iter
        self.window      = timedelta(minutes=window_minutes)
        self.min_ep_size = min_ep_size

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def resegment(self,
                  episodes: List[Episode],
                  artifacts: List[Artifact],
                  embeddings: np.ndarray,
                  segmenter) -> List[Episode]:
        """
        Re-segmente les épisodes avec des itérations EM.

        Parameters
        ----------
        episodes   : List[Episode] — sortie du segmenteur greedy
        artifacts  : List[Artifact] — flux original (même ordre)
        embeddings : np.ndarray (n, d)
        segmenter  : EpisodeSegmenter — fournit attach_score()

        Returns
        -------
        List[Episode] — épisodes re-segmentés (sans épisodes vides)
        """
        eps = [ep for ep in episodes if ep.artifact_indices]
        if not eps:
            return eps

        for iteration in range(self.max_iter):
            # ── E-step ───────────────────────────────────────────────
            new_asgn, changed = self._e_step(eps, artifacts, embeddings, segmenter)

            # ── M-step ───────────────────────────────────────────────
            eps = self._m_step(eps, new_asgn, artifacts, embeddings)

            if changed == 0:
                break

        return eps

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(self,
                eps: List[Episode],
                artifacts: List[Artifact],
                embeddings: np.ndarray,
                segmenter) -> tuple:
        """
        Pour chaque artifact, choisit l'épisode qui maximise AttachScore
        parmi les épisodes dans la fenêtre temporelle.

        Returns
        -------
        new_asgn : Dict[int, Episode]  — nouvelle assignation artifact_idx → ep
        changed  : int                 — nombre de réassignations
        """
        # Mapping courant artifact_idx → episode
        old_asgn: Dict[int, Episode] = {}
        for ep in eps:
            for idx in ep.artifact_indices:
                old_asgn[idx] = ep

        new_asgn = dict(old_asgn)
        changed = 0

        for i, artifact in enumerate(artifacts):
            current = old_asgn.get(i)
            if current is None:
                continue

            # Ne pas vider les petits épisodes
            if len(current.artifact_indices) <= self.min_ep_size:
                continue

            candidates = self._temporal_candidates(artifact, eps)
            if not candidates:
                continue

            emb = embeddings[i]
            best_ep    = current
            best_score = segmenter.attach_score(artifact, emb, current)

            for ep in candidates:
                if ep is current:
                    continue
                score = segmenter.attach_score(artifact, emb, ep)
                if score > best_score:
                    best_score = score
                    best_ep    = ep

            if best_ep is not current:
                new_asgn[i] = best_ep
                changed += 1

        return new_asgn, changed

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _m_step(self,
                eps: List[Episode],
                new_asgn: Dict[int, Episode],
                artifacts: List[Artifact],
                embeddings: np.ndarray) -> List[Episode]:
        """
        Reconstruit artifact_indices depuis new_asgn,
        puis recompute centroid, time_interval, entity_weights pour chaque épisode.
        """
        # Reconstruire artifact_indices
        ep_to_indices: Dict[int, List[int]] = {id(ep): [] for ep in eps}
        for idx, ep in new_asgn.items():
            if id(ep) in ep_to_indices:
                ep_to_indices[id(ep)].append(idx)

        for ep in eps:
            ep.artifact_indices = sorted(ep_to_indices.get(id(ep), []))

        # Supprimer les épisodes vides
        eps = [ep for ep in eps if ep.artifact_indices]

        # Recompute représentations
        self._recompute_all(eps, artifacts, embeddings)

        return eps

    # ------------------------------------------------------------------
    # Recompute représentations (centroid mean, time_interval, entities)
    # ------------------------------------------------------------------

    def _recompute_all(self,
                       eps: List[Episode],
                       artifacts: List[Artifact],
                       embeddings: np.ndarray) -> None:
        """
        Recompute centroid (mean), goal_centroid, time_interval, entity_weights
        à partir des artifact_indices courants.

        On utilise la moyenne arithmétique (pas EMA) car les artifacts ne sont
        plus dans l'ordre séquentiel d'origine après réaffectation.
        """
        for ep in eps:
            indices = ep.artifact_indices
            if not indices:
                continue

            # ── Centroïde sémantique ─────────────────────────────────
            vecs = embeddings[indices]               # (k, d)
            ep.centroid = vecs.mean(axis=0)

            # centroid_init = premier artifact (ordre chronologique)
            t_sorted = sorted(indices, key=lambda i: artifacts[i].timestamp)
            ep.centroid_init = embeddings[t_sorted[0]].copy()

            # ── Goal centroid ────────────────────────────────────────
            goal_vecs = [
                artifacts[i].goal_vector for i in indices
                if artifacts[i].goal_vector is not None
            ]
            ep.goal_centroid = (
                np.mean(goal_vecs, axis=0).astype(np.float32)
                if goal_vecs else None
            )

            # ── Intervalle temporel ──────────────────────────────────
            timestamps = [artifacts[i].timestamp for i in indices]
            ep.time_interval = (min(timestamps), max(timestamps))

            # ── Entités — compte brut ────────────────────────────────
            ep.entity_weights = {}
            for i in indices:
                for ent in artifacts[i].entities:
                    key = ent.lower().strip()
                    ep.entity_weights[key] = ep.entity_weights.get(key, 0.0) + 1.0

    # ------------------------------------------------------------------
    # Candidats temporels
    # ------------------------------------------------------------------

    def _temporal_candidates(self,
                             artifact: Artifact,
                             episodes: List[Episode]) -> List[Episode]:
        """
        Épisodes dont l'intervalle temporel chevauche la fenêtre autour
        du timestamp de l'artifact.

        t - window ≤ ep.time_interval[1]  (fin de l'épisode dans la fenêtre)
        t + window ≥ ep.time_interval[0]  (début de l'épisode dans la fenêtre)
        """
        t = artifact.timestamp
        lo = t - self.window
        hi = t + self.window
        return [
            ep for ep in episodes
            if ep.time_interval is not None
            and ep.time_interval[1] >= lo
            and ep.time_interval[0] <= hi
        ]
