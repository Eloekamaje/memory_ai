"""
episode_algorithm.py — Segmentation épisodique multi-critères (V3).

AttachScore(a, e) = α·Sem(a,e) + β·Ent(a,e) + γ·Temp(a,e) + δ·Goal(a,e)
                    − ρ·AgePenalty(e)

V2 : Aging (ACTIVE→DORMANT), Réactivation (DORMANT→ACTIVE),
     Goal similarity comme 4e signal.

V3 (Challenger fixes) :
  - EMA centroid : remplace moyenne arithmétique (biais 1/n aux grands épisodes)
  - Entity overlap : normalise par artifact_entities, pas union (évite effondrement)
  - AgePenalty sur ACTIVE : pénalise les épisodes actifs depuis > active_penalty_hours
  - hard_break_minutes=720 dans build_pipeline : sessions de 12h par défaut
"""

import numpy as np
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple

from sklearn.metrics.pairwise import cosine_similarity

from models import Artifact, Episode, EpisodeState


class EpisodeSegmenter:

    def __init__(self,
                 time_threshold_minutes: float = 120,
                 attach_threshold: float = 0.3,
                 alpha: float = 0.55,
                 beta: float = 0.15,
                 gamma: float = 0.10,
                 delta: float = 0.20,
                 rho: float = 0.05,
                 recent_window_minutes: float = 480,
                 dormancy_minutes: float = 1440,
                 allow_reactivation: bool = True,
                 hard_break_minutes: float = 0,
                 # V3 Challenger fixes
                 ema_alpha: float = 0.80,
                 active_penalty_hours: float = 24.0):
        """
        Parameters
        ----------
        time_threshold_minutes :
            Au-delà de ce gap, la composante temporelle tend vers 0.
        attach_threshold :
            Score minimum pour rattacher un artefact à un épisode existant.
        alpha, beta, gamma, delta :
            Poids des composantes Sem, Ent, Temp, Goal dans l'AttachScore.
        rho :
            Poids de la pénalité d'âge pour épisodes dormants.
        recent_window_minutes :
            Fenêtre pour les épisodes "récents" candidats.
        dormancy_minutes :
            Inactivité (min) avant passage ACTIVE → DORMANT.
        hard_break_minutes :
            Si > 0 : gap forcé — aucun épisode ACTIVE dont le dernier
            message date de plus de hard_break_minutes ne peut recevoir
            un nouveau message. Cela crée des "sessions" naturelles.
            0 = désactivé (comportement par défaut).
        ema_alpha : (V3)
            Poids du nouvel artefact dans la mise à jour EMA du centroïde.
            centroid ← (1 - ema_alpha) · centroid + ema_alpha · embedding.
            0.80 = le centroïde reflète le contexte récent (dernier msg pèse
            80%, avant-dernier 16%, etc.). Calibré sur synthetic_200 :
            meilleur ARI=0.5179 vs arithmetic mean (1.0) → ARI=0.4598.
        active_penalty_hours : (V3)
            Durée d'inactivité (heures) après laquelle un épisode ACTIVE
            commence à accumuler une pénalité d'âge (évite les épisodes
            immortels qui absorbent n'importe quel thème).
        """

        self.time_threshold = timedelta(minutes=time_threshold_minutes)
        self.attach_threshold = attach_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rho = rho
        self.recent_window = timedelta(minutes=recent_window_minutes)
        self.dormancy_threshold = timedelta(minutes=dormancy_minutes)
        self.allow_reactivation = allow_reactivation
        self.hard_break_minutes = timedelta(minutes=hard_break_minutes) if hard_break_minutes > 0 else None
        # V3
        self.ema_alpha = ema_alpha
        self.active_penalty_hours = active_penalty_hours

        # lambda pour la décroissance temporelle exponentielle
        # calibré pour que exp(-λ * time_threshold) ≈ 0.1
        self._lambda = 2.3 / time_threshold_minutes

        # statistiques de réactivation (pour analyse)
        self.reactivation_count = 0

    # ------------------------------------------------------------------
    # Composantes du score d'attachement
    # ------------------------------------------------------------------

    def _semantic_similarity(self, embedding: np.ndarray,
                             episode: Episode) -> float:
        """
        Similarité cosinus dual-centroid.

        max(sim(emb, centroid_ema), sim(emb, centroid_init))

        centroid_ema  : reflète le contexte récent (dérive avec l'épisode)
        centroid_init : gelé au 1er message — identité fondatrice de l'épisode

        Un épisode long dont le sujet revient à son origine reste attachable
        même si son EMA centroid a dérivé vers un sujet intermédiaire.
        """
        if episode.centroid is None:
            return 0.0

        sim_ema = float(cosine_similarity(
            embedding.reshape(1, -1),
            episode.centroid.reshape(1, -1)
        )[0][0])

        if episode.centroid_init is not None:
            sim_init = float(cosine_similarity(
                embedding.reshape(1, -1),
                episode.centroid_init.reshape(1, -1)
            )[0][0])
            return max(sim_ema, sim_init, 0.0)

        return max(sim_ema, 0.0)

    def _entity_overlap(self, artifact: Artifact,
                        episode: Episode) -> float:
        """Compatibilité entités entre l'artefact et l'épisode."""

        if not artifact.entities or not episode.entity_weights:
            return 0.0

        artifact_entities = set(e.lower().strip() for e in artifact.entities)
        episode_entities = set(episode.entity_weights.keys())

        if not artifact_entities:
            return 0.0

        intersection = artifact_entities & episode_entities

        if not intersection:
            return 0.0

        # V3 fix : normalise par len(artifact_entities), pas union.
        # Jaccard s'effondre quand l'épisode accumule beaucoup d'entités
        # (union >> artifact_entities) → score artificiellement bas.
        # La precision depuis l'artefact mesure "quelle fraction des entités
        # de cet artefact sont déjà dans l'épisode ?" — stable quelle que
        # soit la taille de l'épisode.
        weighted_sum = sum(
            episode.entity_weights.get(e, 0.0)
            for e in intersection
        )

        max_weight = max(episode.entity_weights.values()) if episode.entity_weights else 1.0

        return weighted_sum / (len(artifact_entities) * max_weight) if max_weight > 0 else 0.0

    def _temporal_proximity(self, artifact: Artifact,
                            episode: Episode) -> float:
        """Proximité temporelle : exp(-λ * distance)."""

        if episode.time_interval is None:
            return 0.0

        _, t_end = episode.time_interval

        gap = (artifact.timestamp - t_end).total_seconds() / 60.0

        if gap < 0:
            gap = abs(gap)

        return float(np.exp(-self._lambda * gap))

    # ------------------------------------------------------------------
    # AttachScore
    # ------------------------------------------------------------------

    def _goal_similarity(self, artifact: Artifact,
                         episode: Episode) -> float:
        """Similarité cosinus entre goal_vector de l'artefact et goal_centroid de l'épisode."""

        if artifact.goal_vector is None or episode.goal_centroid is None:
            return 0.0

        sim = cosine_similarity(
            artifact.goal_vector.reshape(1, -1),
            episode.goal_centroid.reshape(1, -1)
        )[0][0]

        return float(max(sim, 0.0))

    def _age_penalty(self, artifact: Artifact, episode: Episode) -> float:
        """
        Pénalité d'âge.

        DORMANT : log(1 + elapsed_hours) — pénalité forte, croissante.
        ACTIVE   : log(1 + max(0, elapsed_hours - active_penalty_hours)) * 0.3
                   — pénalité douce après la fenêtre d'inactivité autorisée.
                   Évite les épisodes ACTIVE immortels qui absorbent n'importe
                   quel sujet par inertie thématique (flaw 4, Challenger V3).

        (V3) : avant, les épisodes ACTIVE retournaient toujours 0 →
               aucune pression temporelle → sur-attachement sur données réelles.
        """

        if episode.state == EpisodeState.ARCHIVED:
            return 0.0

        if episode.time_interval is None:
            return 0.0

        _, t_end = episode.time_interval
        elapsed_hours = (artifact.timestamp - t_end).total_seconds() / 3600.0

        if elapsed_hours < 0:
            elapsed_hours = 0.0

        if episode.state == EpisodeState.DORMANT:
            return float(np.log1p(elapsed_hours))

        # ACTIVE — pénalité douce après active_penalty_hours d'inactivité
        excess = elapsed_hours - self.active_penalty_hours
        if excess <= 0:
            return 0.0
        return float(np.log1p(excess) * 0.3)

    def attach_score(self, artifact: Artifact, embedding: np.ndarray,
                     episode: Episode) -> float:
        """
        AttachScore(a, e) = α·Sem + β·Ent + γ·Temp + δ·Goal − ρ·AgePenalty
        """

        sem = self._semantic_similarity(embedding, episode)
        ent = self._entity_overlap(artifact, episode)
        temp = self._temporal_proximity(artifact, episode)
        goal = self._goal_similarity(artifact, episode)
        age = self._age_penalty(artifact, episode)

        score = (self.alpha * sem
                 + self.beta * ent
                 + self.gamma * temp
                 + self.delta * goal
                 - self.rho * age)

        return max(score, 0.0)

    # ------------------------------------------------------------------
    # Index incrémental (V3 — Challenger fix 5/5)
    # ------------------------------------------------------------------

    def _idx_init(self) -> None:
        """Initialise les index en début de segment()."""
        self._ep_by_id: Dict[str, Episode] = {}
        # entité (lower) → ensemble d'episode_ids qui la contiennent
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)

    def _idx_add(self, episode: Episode) -> None:
        """Enregistre un nouvel épisode dans les index."""
        self._ep_by_id[episode.id] = episode
        for ent in episode.entity_weights:
            self._entity_index[ent].add(episode.id)

    def _idx_add_entities(self, ep_id: str, entities: List[str]) -> None:
        """Ajoute de nouvelles entités à l'index après mise à jour d'un épisode."""
        for ent in entities:
            self._entity_index[ent.lower().strip()].add(ep_id)

    # ------------------------------------------------------------------
    # Sélection de candidats
    # ------------------------------------------------------------------

    def _is_eligible(self, ep: Episode) -> bool:
        """Filtre rapide : exclut ARCHIVED et DORMANT non réactivables."""
        if ep.state == EpisodeState.ARCHIVED:
            return False
        if ep.state == EpisodeState.DORMANT and not self.allow_reactivation:
            return False
        return True

    def _recent_candidates(self, episodes: List[Episode], cutoff) -> Set[str]:
        """
        Épisodes dont t_end >= cutoff — scan complet O(n).

        Note : le break serait O(k) mais incorrect sur les épisodes réactivés
        (leur t_end est mis à jour mais leur position dans la liste reste
        ancienne → le break les manquerait). L'entity index couvre le gain
        de performance principal ; ici on garde la correction.
        """
        ids: Set[str] = set()
        for ep in episodes:
            if ep.time_interval is None:
                continue
            _, t_end = ep.time_interval
            if t_end >= cutoff and self._is_eligible(ep):
                ids.add(ep.id)
        return ids

    def _entity_candidates(self, artifact_entities: Set[str],
                           known: Set[str]) -> Set[str]:
        """
        Épisodes partageant au moins une entité avec l'artefact — O(|entities|).
        Exclut les ids déjà dans `known` pour éviter les doublons.
        """
        ids: Set[str] = set()
        for ent in artifact_entities:
            for ep_id in self._entity_index.get(ent, ()):
                if ep_id in known or ep_id in ids:
                    continue
                ep = self._ep_by_id.get(ep_id)
                if ep is not None and self._is_eligible(ep):
                    ids.add(ep_id)
        return ids

    def _get_candidates(self, artifact: Artifact,
                        episodes: List[Episode]) -> List[Episode]:
        """
        Candidats = Recent ∪ EntityMatched — V3 indexé.

        Avant (V2) : O(n_episodes) par artefact.
        Après (V3) : O(k + |artifact_entities|)  k = épisodes dans la fenêtre.
        """
        cutoff = artifact.timestamp - self.recent_window
        artifact_entities = {e.lower().strip() for e in artifact.entities} if artifact.entities else set()

        recent = self._recent_candidates(episodes, cutoff)
        by_entity = self._entity_candidates(artifact_entities, recent)
        all_ids = recent | by_entity

        return [self._ep_by_id[ep_id] for ep_id in all_ids
                if ep_id in self._ep_by_id]

    # ------------------------------------------------------------------
    # Mise à jour d'un épisode après insertion
    # ------------------------------------------------------------------

    def _update_centroid(self, episode: Episode, embedding: np.ndarray) -> None:
        """EMA centroid (V3) : centroid ← (1-α)·c + α·emb."""
        if episode.centroid is None:
            episode.centroid = embedding.copy()
        else:
            a = self.ema_alpha
            episode.centroid = (1.0 - a) * episode.centroid + a * embedding

    def _update_goal_centroid(self, episode: Episode, artifact: Artifact) -> None:
        """EMA goal_centroid — symétrique au centroid sémantique."""
        if artifact.goal_vector is None:
            return
        if episode.goal_centroid is None:
            episode.goal_centroid = artifact.goal_vector.copy()
        else:
            a = self.ema_alpha
            episode.goal_centroid = (1.0 - a) * episode.goal_centroid + a * artifact.goal_vector

    def _update_time_interval(self, episode: Episode, timestamp) -> None:
        """Étend l'intervalle temporel de l'épisode."""
        if episode.time_interval is None:
            episode.time_interval = (timestamp, timestamp)
        else:
            t_start, t_end = episode.time_interval
            episode.time_interval = (min(t_start, timestamp), max(t_end, timestamp))

    def _update_entity_weights(self, episode: Episode, artifact: Artifact) -> None:
        """Met à jour entity_weights et l'index inversé pour les nouvelles entités."""
        if not artifact.entities:
            return
        new_for_index = []
        for entity in artifact.entities:
            key = entity.lower().strip()
            if key not in episode.entity_weights:
                new_for_index.append(key)
            episode.entity_weights[key] = episode.entity_weights.get(key, 0.0) + 1.0
        if new_for_index:
            self._idx_add_entities(episode.id, new_for_index)

    def _update_episode(self, episode: Episode, artifact: Artifact,
                        artifact_index: int, embedding: np.ndarray):
        """Met à jour l'épisode après insertion d'un nouvel artefact."""
        episode.artifact_indices.append(artifact_index)
        self._update_centroid(episode, embedding)
        self._update_goal_centroid(episode, artifact)
        self._update_time_interval(episode, artifact.timestamp)
        self._update_entity_weights(episode, artifact)
        episode.memory_score += artifact.importance * 0.1
        if episode.state == EpisodeState.DORMANT and self.allow_reactivation:
            episode.state = EpisodeState.ACTIVE
            self.reactivation_count += 1

    # ------------------------------------------------------------------
    # V2 — Transitions d'état (aging)
    # ------------------------------------------------------------------

    def _apply_aging(self, current_time, episodes: List[Episode]):
        """
        Vérifie chaque épisode ACTIVE :
        si inactif depuis > dormancy_threshold → passe en DORMANT.
        """

        for episode in episodes:
            if episode.state != EpisodeState.ACTIVE:
                continue

            if episode.time_interval is None:
                continue

            _, t_end = episode.time_interval
            inactivity = current_time - t_end

            if inactivity > self.dormancy_threshold:
                episode.state = EpisodeState.DORMANT

    # ------------------------------------------------------------------
    # Segmentation principale
    # ------------------------------------------------------------------

    def segment(self, artifacts: List[Artifact],
                embeddings: np.ndarray) -> List[Episode]:
        """
        Segmente un flux d'artefacts en épisodes (V2 avec aging).

        Parameters
        ----------
        artifacts : liste d'Artifact triés par timestamp
        embeddings : np.ndarray de shape (n, dim)

        Returns
        -------
        episodes : liste d'Episode
        """

        episodes: List[Episode] = []
        episode_counter = 0
        self.reactivation_count = 0
        self._idx_init()   # V3 — index incrémental

        for i, artifact in enumerate(artifacts):

            emb = embeddings[i]

            # V2 — aging : mettre à jour les états AVANT le rattachement
            self._apply_aging(artifact.timestamp, episodes)

            # récupérer les candidats
            candidates = self._get_candidates(artifact, episodes)

            # calculer les scores
            best_score = -1.0
            best_episode: Optional[Episode] = None

            for ep in candidates:
                score = self.attach_score(artifact, emb, ep)
                if score > best_score:
                    best_score = score
                    best_episode = ep

            # hard break : vérifier si le meilleur candidat est trop ancien
            if (best_episode is not None
                    and self.hard_break_minutes is not None
                    and best_episode.time_interval is not None):
                _, t_end = best_episode.time_interval
                gap = artifact.timestamp - t_end
                if gap > self.hard_break_minutes:
                    best_score = -1.0     # forcer un nouvel épisode
                    best_episode = None

            # décision : rattacher ou créer
            if best_score >= self.attach_threshold and best_episode is not None:
                self._update_episode(best_episode, artifact, i, emb)
            else:
                episode_counter += 1
                new_episode = self._create_episode(
                    episode_counter, artifact, i, emb
                )
                episodes.append(new_episode)
                self._idx_add(new_episode)   # V3 — enregistrer dans l'index

        return episodes

    def _create_episode(self, counter: int, artifact: Artifact,
                        index: int, embedding: np.ndarray) -> Episode:
        """Crée un nouvel épisode à partir d'un artefact fondateur."""

        new_episode = Episode(
            id=f"EP_{counter:04d}",
            artifact_indices=[index],
            centroid=embedding.copy(),
            centroid_init=embedding.copy(),   # gelé — identité fondatrice
            goal_centroid=artifact.goal_vector.copy() if artifact.goal_vector is not None else None,
            time_interval=(artifact.timestamp, artifact.timestamp),
            entity_weights={},
            state=EpisodeState.ACTIVE,
        )

        if artifact.entities:
            for entity in artifact.entities:
                key = entity.lower().strip()
                new_episode.entity_weights[key] = 1.0

        return new_episode

    # ------------------------------------------------------------------
    # V2+ — Consolidation post-segmentation
    # ------------------------------------------------------------------

    def consolidate(self, episodes: List[Episode],
                    sem_threshold: float = 0.70,
                    entity_threshold: float = 0.30,
                    max_gap_hours: float = 72.0) -> List[Episode]:
        """
        Fusionne les épisodes dormants sémantiquement + thématiquement proches
        ET temporellement adjacents.

        Deux épisodes sont fusionnés si :
          cosine(centroid_i, centroid_j) >= sem_threshold
          AND entity_overlap >= entity_threshold
          AND gap temporel entre eux <= max_gap_hours

        Parameters
        ----------
        episodes : liste d'Episode (sortie de segment())
        sem_threshold : seuil de similarité cosinus entre centroïdes
        entity_threshold : seuil de Jaccard sur les entités
        max_gap_hours : gap temporel max entre fin d'un et début de l'autre
        """

        if len(episodes) < 2:
            return episodes

        # Index des épisodes dormants candidats à la fusion
        dormant_ids = [
            i for i, ep in enumerate(episodes)
            if ep.state == EpisodeState.DORMANT and len(ep.artifact_indices) > 0
        ]

        if len(dormant_ids) < 2:
            return episodes

        # Calcul de toutes les paires (similarité, entity_overlap)
        merge_pairs = []
        for idx_a in range(len(dormant_ids)):
            for idx_b in range(idx_a + 1, len(dormant_ids)):
                i, j = dormant_ids[idx_a], dormant_ids[idx_b]
                ep_i, ep_j = episodes[i], episodes[j]

                # Similarité sémantique des centroïdes
                if ep_i.centroid is None or ep_j.centroid is None:
                    continue
                sem_sim = cosine_similarity(
                    ep_i.centroid.reshape(1, -1),
                    ep_j.centroid.reshape(1, -1)
                )[0][0]

                if sem_sim < sem_threshold:
                    continue

                # Contrainte temporelle : gap entre les épisodes
                if ep_i.time_interval and ep_j.time_interval:
                    # gap = min distance entre les intervalles
                    end_i, start_j = ep_i.time_interval[1], ep_j.time_interval[0]
                    end_j, start_i = ep_j.time_interval[1], ep_i.time_interval[0]
                    gap_hours = min(
                        abs((start_j - end_i).total_seconds()),
                        abs((start_i - end_j).total_seconds())
                    ) / 3600.0
                    if gap_hours > max_gap_hours:
                        continue

                # Jaccard entités
                ent_i = set(ep_i.entity_weights.keys())
                ent_j = set(ep_j.entity_weights.keys())
                if ent_i or ent_j:
                    ent_overlap = len(ent_i & ent_j) / len(ent_i | ent_j)
                else:
                    ent_overlap = 0.0

                if ent_overlap < entity_threshold:
                    continue

                merge_pairs.append((sem_sim + ent_overlap, i, j))

        # Tri par score combiné décroissant
        merge_pairs.sort(key=lambda x: x[0], reverse=True)

        # Union-Find pour les fusions
        parent = list(range(len(episodes)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merge_count = 0
        for _, i, j in merge_pairs:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
                merge_count += 1

        if merge_count == 0:
            return episodes

        # Regrouper les épisodes par racine
        from collections import defaultdict
        groups = defaultdict(list)
        for idx in range(len(episodes)):
            groups[find(idx)].append(idx)

        # Construire les épisodes consolidés
        consolidated = []
        for root, members in groups.items():
            if len(members) == 1:
                consolidated.append(episodes[members[0]])
                continue

            # Fusion : on garde l'ID du plus gros épisode
            member_episodes = [episodes[m] for m in members]
            member_episodes.sort(key=lambda e: len(e.artifact_indices), reverse=True)
            base = member_episodes[0]

            # Fusionner les indices d'artefacts
            all_indices = []
            for ep in member_episodes:
                all_indices.extend(ep.artifact_indices)
            all_indices.sort()

            # Nouveau centroïde = moyenne pondérée par taille
            total_arts = sum(len(ep.artifact_indices) for ep in member_episodes)
            new_centroid = np.zeros_like(base.centroid)
            for ep in member_episodes:
                w = len(ep.artifact_indices) / total_arts
                new_centroid += w * ep.centroid

            # Nouveau goal_centroid
            new_goal = None
            goal_eps = [ep for ep in member_episodes if ep.goal_centroid is not None]
            if goal_eps:
                total_g = sum(len(ep.artifact_indices) for ep in goal_eps)
                new_goal = np.zeros_like(goal_eps[0].goal_centroid)
                for ep in goal_eps:
                    w = len(ep.artifact_indices) / total_g
                    new_goal += w * ep.goal_centroid

            # Fusionner entités
            merged_weights = {}
            for ep in member_episodes:
                for ent, w in ep.entity_weights.items():
                    merged_weights[ent] = merged_weights.get(ent, 0.0) + w

            # Fusionner intervalles
            all_starts = [ep.time_interval[0] for ep in member_episodes if ep.time_interval]
            all_ends = [ep.time_interval[1] for ep in member_episodes if ep.time_interval]

            merged_ep = Episode(
                id=base.id,
                artifact_indices=all_indices,
                centroid=new_centroid,
                goal_centroid=new_goal,
                entity_weights=merged_weights,
                time_interval=(min(all_starts), max(all_ends)) if all_starts else None,
                memory_score=sum(ep.memory_score for ep in member_episodes),
                state=EpisodeState.CONSOLIDATED,
            )
            consolidated.append(merged_ep)

        self.consolidation_merges = merge_count
        return consolidated