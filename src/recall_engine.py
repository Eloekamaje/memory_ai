"""
recall_engine.py — Moteur de rappel épisodique (Phase C — §10 du framework).

Étant donné une requête en langage naturel, retrouve les épisodes
les plus pertinents de la mémoire, classés par score de rappel :

    RecallScore(q, e) = α·Sem(q,e) + β·Ent(q,e) + γ·Recency(e) + δ·Mem(e)

Fonctionnalités :
  • recall(query)         — recherche hybride (sémantique + entités + récence)
  • recall_by_entity(name)— recherche par entité spécifique
  • recall_by_time(start, end) — recherche par fenêtre temporelle
  • explain(result)       — explication lisible du rappel
  • Expansion via MemoryGraph : épisodes reliés par entités-pont
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models import Artifact, Episode, EpisodeState


# ================================================================== #
# Résultat de rappel                                                   #
# ================================================================== #

@dataclass
class RecallResult:
    """Résultat d'un rappel pour un épisode donné."""

    episode_id: str
    relevance_score: float = 0.0       # score global combiné

    # --- sous-scores ---
    semantic_score: float = 0.0        # sim(query_emb, centroïde)
    entity_score: float = 0.0          # overlap entités requête ↔ épisode
    recency_score: float = 0.0         # fraîcheur temporelle
    memory_score: float = 0.0          # saillance mémorielle normalisée

    # --- détails ---
    matching_entities: List[str] = field(default_factory=list)
    episode_state: str = "ACTIVE"
    artifact_count: int = 0
    time_interval: Optional[Tuple[datetime, datetime]] = None

    # --- expansion via graphe ---
    related_episodes: List[Tuple[str, float]] = field(default_factory=list)

    # --- artefacts les plus pertinents dans l'épisode ---
    top_artifacts: List[Tuple[int, str, float]] = field(default_factory=list)
    # (index, content_preview, similarity)


# ================================================================== #
# Recall Engine                                                        #
# ================================================================== #

class RecallEngine:
    """
    Moteur de rappel multi-critères sur la mémoire épisodique.

    Parameters
    ----------
    episodes : épisodes segmentés (sortie de EpisodeSegmenter)
    artifacts : liste d'artefacts enrichis (entités)
    embeddings : matrice (n, dim) — embeddings des artefacts
    embed_fn : fonction d'embedding texte → np.ndarray
    memory_graph : instance de MemoryGraph (optionnel, pour expansion)
    extractor : EntityExtractor (optionnel, pour extraire les entités de la query)
    resolver : EntityResolver (optionnel, pour résoudre les entités de la query)
    index : instance de MemoryIndex (optionnel, accélère recall via ANN)
    """

    def __init__(
        self,
        episodes: List[Episode],
        artifacts: List[Artifact],
        embeddings: np.ndarray,
        embed_fn: Callable,
        memory_graph=None,
        extractor=None,
        resolver=None,
        index=None,
        # --- poids du recall score ---
        alpha: float = 0.50,
        beta: float = 0.20,
        gamma: float = 0.15,
        delta: float = 0.15,
        # --- paramètres ---
        recency_half_life_hours: float = 168.0,   # 1 semaine
        top_artifacts_per_episode: int = 3,
        content_preview_length: int = 120,
    ):
        self.episodes = episodes
        self.artifacts = artifacts
        self.embeddings = embeddings
        self.embed_fn = embed_fn
        self.graph = memory_graph
        self.extractor = extractor
        self.resolver = resolver
        self.index = index

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.recency_half_life = recency_half_life_hours
        self.top_k_artifacts = top_artifacts_per_episode
        self.preview_len = content_preview_length

        # mapping optionnel episode_id → EpisodeSummary (ajouté post-init)
        self.summary_map: Dict[str, object] = {}

        # pré-calcul : normalisation du memory_score
        all_mem = [ep.memory_score for ep in episodes]
        self._max_memory = max(all_mem) if all_mem else 1.0

        # pré-calcul : timestamp le plus récent (référence pour la récence)
        all_ends = [
            ep.time_interval[1]
            for ep in episodes
            if ep.time_interval is not None
        ]
        self._reference_time = max(all_ends) if all_ends else datetime.now()

        # index inverse artifact_index → episode_id
        self._art_to_ep: Dict[int, str] = {}
        for ep in episodes:
            for idx in ep.artifact_indices:
                self._art_to_ep[idx] = ep.id

    # ============================================================== #
    # Composantes du score de rappel                                   #
    # ============================================================== #

    def _semantic_score(self, query_emb: np.ndarray, episode: Episode) -> float:
        """Similarité cosinus query ↔ centroïde de l'épisode."""
        if episode.centroid is None:
            return 0.0
        sim = cosine_similarity(
            query_emb.reshape(1, -1),
            episode.centroid.reshape(1, -1),
        )[0][0]
        return float(max(sim, 0.0))

    def _entity_score(
        self, query_entities: Set[str], episode: Episode,
    ) -> Tuple[float, List[str]]:
        """
        Overlap entités entre la requête et l'épisode.

        Returns (score, matching_entities)
        """
        if not query_entities or not episode.entity_weights:
            return 0.0, []

        ep_entities = set(episode.entity_weights.keys())
        matching = query_entities & ep_entities

        if not matching:
            return 0.0, []

        # Jaccard pondéré par les poids de l'épisode
        w_match = sum(episode.entity_weights.get(e, 0.0) for e in matching)
        w_total = sum(episode.entity_weights.values())

        score = w_match / w_total if w_total > 0 else 0.0
        return score, sorted(matching)

    def _recency_score(self, episode: Episode) -> float:
        """
        Score de récence : décroissance exponentielle.

        $$\\text{Recency}(e) = 2^{-\\Delta t / t_{1/2}}$$

        Où $\\Delta t$ est le temps écoulé depuis la fin de l'épisode
        et $t_{1/2}$ est la demi-vie en heures.
        """
        if episode.time_interval is None:
            return 0.0

        _, t_end = episode.time_interval
        delta_hours = (self._reference_time - t_end).total_seconds() / 3600.0

        if delta_hours < 0:
            delta_hours = 0.0

        return float(2.0 ** (-delta_hours / self.recency_half_life))

    def _memory_norm_score(self, episode: Episode) -> float:
        """Score de saillance mémorielle normalisé [0, 1]."""
        if self._max_memory <= 0:
            return 0.0
        return min(episode.memory_score / self._max_memory, 1.0)

    # ============================================================== #
    # Extraction des entités de la requête                             #
    # ============================================================== #

    def _extract_query_entities(self, query: str) -> Set[str]:
        """
        Extrait les entités de la requête.

        Stratégie multi-couche :
          1. Si extractor+resolver dispo → extraction NER complète
          2. Sinon → matching par mots-clés contre les entités connues
        """
        query_lower = query.lower().strip()

        # --- Stratégie 1 : NER si disponible ---
        if self.extractor is not None:
            # Créer un pseudo-artefact pour l'extraction
            pseudo = Artifact(
                id="__query__",
                timestamp=datetime.now(),
                author="query",
                content=query,
            )
            mentions = self.extractor.extract_batch([pseudo])

            if mentions and self.resolver is not None:
                # Résoudre via le resolver existant
                resolved = set()
                for m in mentions:
                    key = m.surface_form.lower().strip()
                    # chercher dans les entités connues du resolver
                    if hasattr(self.resolver, "lookup"):
                        ce = self.resolver.lookup(key)
                        if ce is not None:
                            resolved.add(ce.canonical_name.lower().strip())
                        else:
                            resolved.add(key)
                    else:
                        resolved.add(key)
                if resolved:
                    return resolved
            elif mentions:
                return {m.surface_form.lower().strip() for m in mentions}

        # --- Stratégie 2 : fuzzy keyword matching ---
        all_known = set()
        for ep in self.episodes:
            all_known.update(ep.entity_weights.keys())

        matched = set()
        query_words = set(query_lower.split())

        for entity_name in all_known:
            ent_lower = entity_name.lower()
            if ent_lower in query_lower or any(w in ent_lower for w in query_words if len(w) > 2):
                matched.add(entity_name)

        return matched

    # ============================================================== #
    # Top artefacts pertinents dans un épisode                         #
    # ============================================================== #

    def _top_episode_artifacts(
        self, query_emb: np.ndarray, episode: Episode,
    ) -> List[Tuple[int, str, float]]:
        """
        Retourne les artefacts les plus similaires à la requête
        au sein d'un épisode donné.
        """
        scored = []
        for idx in episode.artifact_indices:
            if idx >= len(self.embeddings):
                continue
            art_emb = self.embeddings[idx].reshape(1, -1)
            sim = float(cosine_similarity(query_emb, art_emb)[0][0])
            content = self.artifacts[idx].content
            preview = content[:self.preview_len].replace("\n", " ")
            if len(content) > self.preview_len:
                preview += "…"
            scored.append((idx, preview, sim))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:self.top_k_artifacts]

    # ============================================================== #
    # Expansion via Memory Graph                                       #
    # ============================================================== #

    def _graph_expansion(
        self, episode_id: str, min_similarity: float = 0.1, top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Épisodes reliés via entités partagées (Memory Graph)."""
        if self.graph is None:
            return []

        related = self.graph.related_episodes(
            episode_id,
            min_similarity=min_similarity,
            top_k=top_k,
        )
        return [(ep_id, sim) for ep_id, sim, _ in related]

    # ============================================================== #
    # Rappel principal                                                 #
    # ============================================================== #

    def recall(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
        min_score: float = 0.0,
        exclude_archived: bool = True,
    ) -> List[RecallResult]:
        """
        Recherche hybride dans la mémoire épisodique.

        $$\\text{Recall}(q, e) = \\alpha \\cdot \\text{Sem}(q,e)
                                + \\beta  \\cdot \\text{Ent}(q,e)
                                + \\gamma \\cdot \\text{Recency}(e)
                                + \\delta \\cdot \\text{Mem}(e)$$

        Parameters
        ----------
        query : texte de la requête en langage naturel
        top_k : nombre maximum de résultats
        use_graph : si True, ajoute les épisodes reliés via le graphe
        min_score : seuil minimum de pertinence
        exclude_archived : ignorer les épisodes archivés

        Returns
        -------
        Liste de RecallResult triés par relevance_score décroissant.
        """
        # 1. Embed la requête
        query_emb = self.embed_fn([query])
        if query_emb.ndim == 2:
            query_emb = query_emb[0]
        q = query_emb.reshape(1, -1)

        # 2. Extraire les entités de la requête
        query_entities = self._extract_query_entities(query)

        # 3. Filtrer les candidats via l'index ANN (si disponible)
        if self.index is not None:
            candidate_ids = self.index.candidates(
                query_emb, query_entities, top_k_sem=max(top_k * 4, 20)
            )
            episodes_to_score = [ep for ep in self.episodes if ep.id in candidate_ids]
        else:
            episodes_to_score = self.episodes

        # 4. Scorer les candidats
        results: List[RecallResult] = []

        for episode in episodes_to_score:
            if exclude_archived and episode.state == EpisodeState.ARCHIVED:
                continue

            sem = self._semantic_score(q, episode)
            ent, matching = self._entity_score(query_entities, episode)
            rec = self._recency_score(episode)
            mem = self._memory_norm_score(episode)

            score = (
                self.alpha * sem
                + self.beta * ent
                + self.gamma * rec
                + self.delta * mem
            )

            if score < min_score:
                continue

            # Top artefacts de cet épisode
            top_arts = self._top_episode_artifacts(q, episode)

            result = RecallResult(
                episode_id=episode.id,
                relevance_score=round(score, 4),
                semantic_score=round(sem, 4),
                entity_score=round(ent, 4),
                recency_score=round(rec, 4),
                memory_score=round(mem, 4),
                matching_entities=matching,
                episode_state=episode.state.value,
                artifact_count=len(episode.artifact_indices),
                time_interval=episode.time_interval,
            )

            # Artefacts pertinents
            result.top_artifacts = top_arts

            # Expansion graphe
            if use_graph:
                result.related_episodes = self._graph_expansion(episode.id)

            results.append(result)

        # 4. Trier par relevance décroissante
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        return results[:top_k]

    # ============================================================== #
    # Rappel par entité                                                #
    # ============================================================== #

    def recall_by_entity(
        self, entity_name: str, top_k: int = 5,
    ) -> List[RecallResult]:
        """
        Rappel direct par nom d'entité.

        Utilise le Memory Graph si disponible, sinon scan les épisodes.
        """
        key = entity_name.lower().strip()

        # Trouver les épisodes contenant cette entité
        matching_eps: Dict[str, float] = {}

        if self.graph is not None:
            # via le graphe (plus efficace)
            ep_ids = self.graph.entity_episodes(key)
            for eid in ep_ids:
                ent_data = self.graph.episode_entities(eid)
                matching_eps[eid] = ent_data.get(key, 1.0)
        else:
            # scan direct
            for ep in self.episodes:
                if key in ep.entity_weights:
                    matching_eps[ep.id] = ep.entity_weights[key]

        if not matching_eps:
            return []

        # Construire les résultats
        results: List[RecallResult] = []
        ep_dict = {ep.id: ep for ep in self.episodes}

        for eid, weight in matching_eps.items():
            ep = ep_dict.get(eid)
            if ep is None:
                continue

            rec = self._recency_score(ep)
            mem = self._memory_norm_score(ep)

            # Score : poids de l'entité + récence + mémoire
            max_w = max(matching_eps.values()) if matching_eps else 1.0
            ent_norm = weight / max_w if max_w > 0 else 0.0

            score = 0.50 * ent_norm + 0.30 * rec + 0.20 * mem

            result = RecallResult(
                episode_id=eid,
                relevance_score=round(score, 4),
                semantic_score=0.0,
                entity_score=round(ent_norm, 4),
                recency_score=round(rec, 4),
                memory_score=round(mem, 4),
                matching_entities=[key],
                episode_state=ep.state.value,
                artifact_count=len(ep.artifact_indices),
                time_interval=ep.time_interval,
            )

            if self.graph is not None:
                result.related_episodes = self._graph_expansion(eid)

            results.append(result)

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    # ============================================================== #
    # Rappel par fenêtre temporelle                                    #
    # ============================================================== #

    def recall_by_time(
        self,
        start: datetime,
        end: datetime,
        top_k: int = 10,
    ) -> List[RecallResult]:
        """
        Rappel des épisodes actifs dans la fenêtre [start, end].

        Un épisode est inclus s'il chevauche la fenêtre temporelle.
        """
        results: List[RecallResult] = []

        for ep in self.episodes:
            if ep.time_interval is None:
                continue

            ep_start, ep_end = ep.time_interval

            # test de chevauchement
            if ep_end < start or ep_start > end:
                continue

            # score = proportion de chevauchement + mémoire
            overlap_start = max(ep_start, start)
            overlap_end = min(ep_end, end)
            overlap_span = (overlap_end - overlap_start).total_seconds()
            ep_span = max((ep_end - ep_start).total_seconds(), 1.0)
            window_span = max((end - start).total_seconds(), 1.0)

            coverage = overlap_span / min(ep_span, window_span)
            mem = self._memory_norm_score(ep)

            score = 0.70 * min(coverage, 1.0) + 0.30 * mem

            result = RecallResult(
                episode_id=ep.id,
                relevance_score=round(score, 4),
                recency_score=round(coverage, 4),
                memory_score=round(mem, 4),
                episode_state=ep.state.value,
                artifact_count=len(ep.artifact_indices),
                time_interval=ep.time_interval,
            )

            # extraire les entités pour le contexte
            result.matching_entities = sorted(
                ep.entity_weights.keys(),
                key=lambda k: ep.entity_weights[k],
                reverse=True,
            )[:5]

            results.append(result)

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    # ============================================================== #
    # Explication lisible                                              #
    # ============================================================== #

    def explain(self, result: RecallResult, verbose: bool = False) -> str:
        """
        Explication humaine d'un résultat de rappel.

        Returns
        -------
        Texte formaté décrivant pourquoi cet épisode a été rappelé.
        """
        lines = []
        lines.append(f"🔍 {result.episode_id}  (score: {result.relevance_score:.3f})")

        # Ajouter le label si disponible
        s = self.summary_map.get(result.episode_id)
        if s and hasattr(s, 'label'):
            lines.append(f"   📌 {s.label}")

        lines.append(f"   État: {result.episode_state}  |  {result.artifact_count} artefacts")

        if result.time_interval:
            t0, t1 = result.time_interval
            lines.append(f"   Période: {t0.strftime('%Y-%m-%d %H:%M')} → {t1.strftime('%Y-%m-%d %H:%M')}")

        # Sous-scores
        bars = []
        if result.semantic_score > 0:
            bars.append(f"Sem={result.semantic_score:.3f}")
        if result.entity_score > 0:
            bars.append(f"Ent={result.entity_score:.3f}")
        if result.recency_score > 0:
            bars.append(f"Rec={result.recency_score:.3f}")
        if result.memory_score > 0:
            bars.append(f"Mem={result.memory_score:.3f}")
        if bars:
            lines.append(f"   Composantes: {' | '.join(bars)}")

        # Entités matchées
        if result.matching_entities:
            lines.append(f"   Entités: {', '.join(result.matching_entities)}")

        # Épisodes reliés
        if result.related_episodes:
            related = [f"{eid}({sim:.2f})" for eid, sim in result.related_episodes[:3]]
            lines.append(f"   Reliés: {' '.join(related)}")

        # Top artefacts
        if verbose and result.top_artifacts:
            lines.append("   ── Artefacts pertinents ──")
            for idx, preview, sim in result.top_artifacts:
                lines.append(f"      [{idx}] (sim={sim:.3f}) {preview}")

        return "\n".join(lines)

    # ============================================================== #
    # Recherche batch (multiple queries)                               #
    # ============================================================== #

    def recall_batch(
        self,
        queries: List[str],
        top_k: int = 3,
    ) -> Dict[str, List[RecallResult]]:
        """
        Rappel pour plusieurs requêtes à la fois.

        Returns
        -------
        Dict[query → List[RecallResult]]
        """
        return {q: self.recall(q, top_k=top_k) for q in queries}

    # ============================================================== #
    # Résumé de session (quelles sont les mémoires actives ?)          #
    # ============================================================== #

    def active_memories_summary(self, top_k: int = 10) -> str:
        """
        Résumé des mémoires les plus saillantes (sans requête).

        Classement basé sur : memory_score × recency.
        """
        scored = []
        for ep in self.episodes:
            if ep.state == EpisodeState.ARCHIVED:
                continue
            rec = self._recency_score(ep)
            mem = self._memory_norm_score(ep)
            salience = 0.60 * mem + 0.40 * rec
            scored.append((ep, salience))

        scored.sort(key=lambda x: x[1], reverse=True)

        lines = [
            "╔══════════════════════════════════════════════╗",
            "║         ACTIVE MEMORIES SUMMARY              ║",
            "╚══════════════════════════════════════════════╝",
        ]

        for ep, sal in scored[:top_k]:
            state_icon = {
                EpisodeState.ACTIVE: "🟢",
                EpisodeState.DORMANT: "💤",
                EpisodeState.CONSOLIDATED: "📦",
            }.get(ep.state, "❓")

            n = len(ep.artifact_indices)
            ents = sorted(
                ep.entity_weights.keys(),
                key=lambda k: ep.entity_weights[k],
                reverse=True,
            )[:4]

            t_str = ""
            if ep.time_interval:
                t0, t1 = ep.time_interval
                t_str = f"{t0.strftime('%m-%d %H:%M')}→{t1.strftime('%m-%d %H:%M')}"

            # Label du summarizer si disponible
            s = self.summary_map.get(ep.id)
            label = f" | 📌 {s.label}" if (s and hasattr(s, 'label')) else ""

            lines.append(
                f"  {state_icon} {ep.id} | sal={sal:.3f} | "
                f"{n} arts | {t_str}{label}"
            )

        return "\n".join(lines)
