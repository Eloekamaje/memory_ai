"""
episode_splitter.py — Splitting post-segmentation des méga-épisodes (Phase D).

Détecte les épisodes à faible cohésion et les découpe en sous-épisodes
via clustering hiérarchique (Agglomerative) sur les embeddings.

Critères de splitting (§3 du framework) :
  1. Cohésion sémantique < seuil  →  le contenu est trop hétérogène
  2. Taille > seuil               →  trop d'artefacts concentrés
  3. Span temporel > seuil        →  trop étalé dans le temps

Algorithme :
  Pour chaque épisode candidat :
    a) Extraire les embeddings des artefacts
    b) Clustering hiérarchique (Ward) avec distance-based cut
    c) Choisir k via silhouette score (2 ≤ k ≤ max_splits)
    d) Reconstruire les sous-épisodes avec centroïdes, entités, intervalles

Le splitter ne touche PAS au segmenter original — c'est une passe
post-traitement indépendante.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from models import Artifact, Episode, EpisodeState


# ================================================================== #
# Configuration                                                        #
# ================================================================== #

class SplitConfig:
    """Paramètres du splitting post-segmentation."""

    def __init__(
        self,
        min_cohesion: float = 0.55,
        min_size_to_split: int = 8,
        max_span_hours: float = 168.0,     # 7 jours
        max_splits: int = 6,
        min_sub_size: int = 3,
        silhouette_threshold: float = 0.10,
    ):
        """
        Parameters
        ----------
        min_cohesion :
            Seuil de cohésion sémantique en-dessous duquel un épisode
            est candidat au splitting.
        min_size_to_split :
            Nombre minimum d'artefacts pour qu'un épisode soit splittable.
        max_span_hours :
            Si le span temporel dépasse ce seuil, c'est un signal de split.
        max_splits :
            Nombre maximum de sous-épisodes issus d'un split.
        min_sub_size :
            Taille minimale d'un sous-épisode (sinon → absorption).
        silhouette_threshold :
            Score silhouette minimum pour valider un split.
        """
        self.min_cohesion = min_cohesion
        self.min_size_to_split = min_size_to_split
        self.max_span_hours = max_span_hours
        self.max_splits = max_splits
        self.min_sub_size = min_sub_size
        self.silhouette_threshold = silhouette_threshold


# ================================================================== #
# Episode Splitter                                                     #
# ================================================================== #

class EpisodeSplitter:
    """
    Découpe les épisodes surdimensionnés ou incohérents en sous-épisodes.
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()
        # statistiques
        self.split_count = 0
        self.candidates_found = 0
        self.episodes_before = 0
        self.episodes_after = 0

    # ============================================================== #
    # Détection des candidats                                          #
    # ============================================================== #

    def _semantic_cohesion(
        self, indices: List[int], embeddings: np.ndarray,
    ) -> float:
        """Cohésion sémantique intra-épisode : mean(sim(a_i, centroïde))."""
        if len(indices) < 2:
            return 1.0
        ep_embs = embeddings[indices]
        centroid = np.mean(ep_embs, axis=0).reshape(1, -1)
        sims = cosine_similarity(ep_embs, centroid).flatten()
        return float(np.mean(sims))

    def _temporal_span_hours(self, episode: Episode) -> float:
        """Span temporel de l'épisode en heures."""
        if episode.time_interval is None:
            return 0.0
        t_start, t_end = episode.time_interval
        return (t_end - t_start).total_seconds() / 3600.0

    def _is_candidate(
        self, episode: Episode, embeddings: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """
        Vérifie si un épisode est candidat au splitting.

        Returns (is_candidate, cohesion, span_hours)
        """
        n = len(episode.artifact_indices)
        if n < self.config.min_size_to_split:
            return False, 1.0, 0.0

        cohesion = self._semantic_cohesion(episode.artifact_indices, embeddings)
        span = self._temporal_span_hours(episode)

        # Candidat si : cohésion faible OU taille × span excessive
        is_low_cohesion = cohesion < self.config.min_cohesion
        is_long_span = span > self.config.max_span_hours

        return (is_low_cohesion or is_long_span), cohesion, span

    # ============================================================== #
    # Clustering interne                                               #
    # ============================================================== #

    def _find_best_k(
        self, ep_embeddings: np.ndarray,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Trouve le meilleur nombre de clusters via silhouette score.

        Returns (best_k, labels, best_silhouette)
        """
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

                # vérifier que chaque cluster a assez d'éléments
                sizes = np.bincount(labels)
                if np.min(sizes) < self.config.min_sub_size:
                    continue

                sil = silhouette_score(ep_embeddings, labels, metric="cosine")
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels
            except Exception:
                continue

        return best_k, best_labels, best_sil

    # ============================================================== #
    # Construction des sous-épisodes                                   #
    # ============================================================== #

    def _build_sub_episodes(
        self,
        parent: Episode,
        labels: np.ndarray,
        artifacts: List[Artifact],
        embeddings: np.ndarray,
        start_counter: int,
    ) -> List[Episode]:
        """
        Construit les sous-épisodes à partir du clustering interne.
        """
        n_clusters = int(labels.max()) + 1
        sub_episodes: List[Episode] = []

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            sub_art_indices = [
                parent.artifact_indices[i]
                for i in range(len(parent.artifact_indices))
                if mask[i]
            ]

            if not sub_art_indices:
                continue

            # Centroïde
            sub_embs = embeddings[sub_art_indices]
            centroid = np.mean(sub_embs, axis=0)

            # Goal centroid
            goal_vecs = [
                artifacts[idx].goal_vector
                for idx in sub_art_indices
                if artifacts[idx].goal_vector is not None
            ]
            goal_centroid = np.mean(goal_vecs, axis=0) if goal_vecs else None

            # Intervalle temporel
            timestamps = [artifacts[idx].timestamp for idx in sub_art_indices]
            time_interval = (min(timestamps), max(timestamps))

            # Entités pondérées
            entity_weights: Dict[str, float] = {}
            for idx in sub_art_indices:
                art = artifacts[idx]
                if art.entities:
                    for ent in art.entities:
                        key = ent.lower().strip()
                        entity_weights[key] = entity_weights.get(key, 0.0) + 1.0

            # Memory score proportionnel
            ratio = len(sub_art_indices) / len(parent.artifact_indices)
            mem_score = parent.memory_score * ratio

            sub_ep = Episode(
                id=f"{parent.id}_S{cluster_id + 1:02d}",
                artifact_indices=sorted(sub_art_indices),
                centroid=centroid,
                goal_centroid=goal_centroid,
                entity_weights=entity_weights,
                time_interval=time_interval,
                memory_score=mem_score,
                state=parent.state,
            )
            sub_episodes.append(sub_ep)

        return sub_episodes

    # ============================================================== #
    # Pipeline principal                                               #
    # ============================================================== #

    def split(
        self,
        episodes: List[Episode],
        artifacts: List[Artifact],
        embeddings: np.ndarray,
        verbose: bool = True,
    ) -> List[Episode]:
        """
        Applique le splitting sur les épisodes candidats.

        Parameters
        ----------
        episodes : épisodes issus de segment() + consolidate()
        artifacts : liste d'artefacts enrichis
        embeddings : matrice (n, dim)
        verbose : afficher les détails

        Returns
        -------
        Liste d'épisodes (certains découpés, les autres inchangés).
        """
        self.episodes_before = len(episodes)
        self.split_count = 0
        self.candidates_found = 0

        result: List[Episode] = []
        counter = 0

        for episode in episodes:
            is_cand, cohesion, span = self._is_candidate(episode, embeddings)

            if not is_cand:
                result.append(episode)
                continue

            self.candidates_found += 1

            if verbose:
                print(f"    ✂ Candidate: {episode.id} | "
                      f"{len(episode.artifact_indices)} arts | "
                      f"coh={cohesion:.3f} | span={span:.0f}h")

            # Clustering interne
            ep_embs = embeddings[episode.artifact_indices]
            best_k, labels, sil = self._find_best_k(ep_embs)

            if best_k <= 1 or sil < self.config.silhouette_threshold:
                if verbose:
                    print(f"      → Pas de split valide "
                          f"(k={best_k}, sil={sil:.3f})")
                result.append(episode)
                continue

            # Split !
            sub_episodes = self._build_sub_episodes(
                parent=episode,
                labels=labels,
                artifacts=artifacts,
                embeddings=embeddings,
                start_counter=counter,
            )

            self.split_count += 1
            counter += len(sub_episodes)

            if verbose:
                print(f"      → Split en {len(sub_episodes)} sous-épisodes "
                      f"(sil={sil:.3f})")
                for sub in sub_episodes:
                    sub_coh = self._semantic_cohesion(
                        sub.artifact_indices, embeddings,
                    )
                    ents = sorted(
                        sub.entity_weights.keys(),
                        key=lambda k: sub.entity_weights.get(k, 0),
                        reverse=True,
                    )[:4]
                    print(f"        {sub.id}: {len(sub.artifact_indices)} arts "
                          f"| coh={sub_coh:.3f} | {', '.join(ents)}")

            result.extend(sub_episodes)

        self.episodes_after = len(result)

        if verbose and self.split_count > 0:
            print(f"    Splitting: {self.episodes_before} → "
                  f"{self.episodes_after} épisodes "
                  f"({self.candidates_found} candidats, "
                  f"{self.split_count} splits)")

        return result

    # ============================================================== #
    # Analyse : pourquoi un épisode devrait être splitté              #
    # ============================================================== #

    def diagnose(
        self,
        episode: Episode,
        artifacts: List[Artifact],
        embeddings: np.ndarray,
    ) -> str:
        """
        Diagnostic détaillé d'un épisode : cohésion, span, structure interne.
        """
        lines = []
        n = len(episode.artifact_indices)
        cohesion = self._semantic_cohesion(episode.artifact_indices, embeddings)
        span = self._temporal_span_hours(episode)

        lines.append(f"📊 Diagnostic: {episode.id}")
        lines.append(f"   Artefacts: {n}")
        lines.append(f"   Cohésion sémantique: {cohesion:.4f}"
                      f"  {'⚠ FAIBLE' if cohesion < self.config.min_cohesion else '✓'}")
        lines.append(f"   Span temporel: {span:.1f}h"
                      f"  {'⚠ LONG' if span > self.config.max_span_hours else '✓'}")

        # Entités dominantes
        top_ents = sorted(
            episode.entity_weights.items(),
            key=lambda x: x[1], reverse=True,
        )[:8]
        lines.append(f"   Entités: {', '.join(f'{e}({w:.0f})' for e, w in top_ents)}")

        # Structure interne
        if n >= self.config.min_size_to_split:
            ep_embs = embeddings[episode.artifact_indices]
            best_k, labels, sil = self._find_best_k(ep_embs)
            lines.append(f"   Clustering interne: k={best_k}, silhouette={sil:.4f}")

            if best_k > 1:
                for c in range(best_k):
                    mask = labels == c
                    c_indices = [
                        episode.artifact_indices[i]
                        for i in range(n) if mask[i]
                    ]
                    c_coh = self._semantic_cohesion(c_indices, embeddings)
                    c_ents: Dict[str, float] = {}
                    for idx in c_indices:
                        art = artifacts[idx]
                        if art.entities:
                            for e in art.entities:
                                k = e.lower().strip()
                                c_ents[k] = c_ents.get(k, 0.0) + 1.0
                    top_c = sorted(c_ents.items(), key=lambda x: x[1], reverse=True)[:4]
                    ent_str = ', '.join(f'{e}' for e, _ in top_c)
                    lines.append(
                        f"     Cluster {c}: {len(c_indices)} arts, "
                        f"coh={c_coh:.3f}, [{ent_str}]"
                    )

        return "\n".join(lines)
