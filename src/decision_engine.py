"""
decision_engine.py — Decision Engine v0 (règles déterministes).

Rôle dans l'architecture :
    Memory Engine = State
    Agent          = Policy
    Decision Engine = Bridge  ← ce module

Le Decision Engine transforme un état mémoire en actions concrètes.
Il rend le système proactif : sans lui, la mémoire reste passive.

Actions produites :
    COMPRESS  → épisode DORMANT trop vieux     → résumer (LLM ou extractif)
    ARCHIVE   → épisode CONSOLIDATED trop vieux → cold storage
    SURFACE   → épisode dormant proche d'un nouvel artefact → suggérer à l'agent
    LINK      → deux épisodes partagent ≥ N entités → les connecter
    FLAG      → épisode ouvert non résolu → rappel prioritaire

Usage minimal (post-build_pipeline) :
    engine = DecisionEngine()
    actions = engine.evaluate(episodes, new_artifact=artifact)
    for action in actions:
        print(action.summary())

Usage avec exécution des actions :
    executor = ActionExecutor(store, summarizer)
    engine = DecisionEngine()
    actions = engine.evaluate(episodes, new_artifact=artifact)
    executor.apply(actions, episodes, store)

Référence :
    ai-brain/07_visionary_hypotheses.md  — H0
    ai-brain/08_multi_channel_lifecycle.md — Decision Engine section
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import Artifact, Episode, EpisodeState


# ================================================================== #
# Types d'action                                                       #
# ================================================================== #

class ActionType(Enum):
    COMPRESS = "COMPRESS"   # DORMANT → CONSOLIDATED (résumer)
    ARCHIVE  = "ARCHIVE"    # CONSOLIDATED → ARCHIVED (cold storage)
    SURFACE  = "SURFACE"    # Suggérer proactivement à l'agent
    LINK     = "LINK"       # Connecter deux épisodes (entités partagées)
    FLAG     = "FLAG"       # Marquer comme ouvert / non résolu


@dataclass
class Action:
    """
    Action recommandée par le Decision Engine.

    Attributs
    ---------
    type        : type d'action
    episode     : épisode principal concerné
    episode_b   : deuxième épisode (LINK uniquement)
    score       : priorité 0→1 (plus haut = plus urgent)
    reason      : explication lisible par l'humain
    context     : données structurées supplémentaires (optionnel)
    """

    type:      ActionType
    episode:   Episode
    episode_b: Optional[Episode] = None  # pour LINK
    score:     float             = 0.5
    reason:    str               = ""
    context:   Dict              = field(default_factory=dict)

    def summary(self) -> str:
        ep_id  = self.episode.id
        ep_b   = f" ↔ {self.episode_b.id}" if self.episode_b else ""
        return (f"[{self.type.value}] {ep_id}{ep_b} "
                f"(score={self.score:.2f}) — {self.reason}")


# ================================================================== #
# Decision Engine                                                      #
# ================================================================== #

class DecisionEngine:
    """
    Moteur de décision basé sur des règles déterministes.

    Tous les seuils sont configurables à l'instanciation.
    Aucun GPU requis — fonctionne entièrement sur CPU.

    Paramètres
    ----------
    compress_after_days : int
        Jours d'inactivité avant compression (DORMANT → CONSOLIDATED).
        Défaut : 30 jours.
    archive_after_days : int
        Jours depuis compression avant archivage (CONSOLIDATED → ARCHIVED).
        Défaut : 90 jours supplémentaires.
    open_compress_multiplier : float
        Multiplicateur sur compress_after_days pour les épisodes ouverts.
        Défaut : 6× (un épisode ouvert résiste 6× plus longtemps à la compression).
    surface_min_similarity : float
        Similarité cosinus minimale pour déclencher un SURFACE.
        Défaut : 0.72.
    surface_min_importance : float
        memory_score minimal pour qu'un épisode soit surfacé.
        Défaut : 0.35.
    link_min_shared_entities : int
        Nombre minimal d'entités partagées pour un LINK.
        Défaut : 2.
    flag_open_keywords : List[str]
        Mots-clés dans les entités/labels qui signalent un épisode ouvert.
    """

    _OPEN_KEYWORDS = [
        "?", "à faire", "todo", "en cours", "prévu", "attente",
        "décision", "deadline", "projet", "bloquer", "bloqué",
    ]

    def __init__(
        self,
        compress_after_days:       int   = 30,
        archive_after_days:        int   = 90,
        open_compress_multiplier:  float = 6.0,
        surface_min_similarity:    float = 0.72,
        surface_min_importance:    float = 0.35,
        link_min_shared_entities:  int   = 2,
        flag_open_keywords:        Optional[List[str]] = None,
    ):
        self.compress_after_days      = compress_after_days
        self.archive_after_days       = archive_after_days
        self.open_compress_multiplier = open_compress_multiplier
        self.surface_min_similarity   = surface_min_similarity
        self.surface_min_importance   = surface_min_importance
        self.link_min_shared_entities = link_min_shared_entities
        self.flag_open_keywords       = flag_open_keywords or self._OPEN_KEYWORDS

    # ------------------------------------------------------------------ #
    # Point d'entrée principal                                             #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        episodes:     List[Episode],
        new_artifact: Optional[Artifact] = None,
        now:          Optional[datetime] = None,
    ) -> List[Action]:
        """
        Évalue l'état mémoire et retourne les actions recommandées.

        Paramètres
        ----------
        episodes     : liste de tous les épisodes connus
        new_artifact : nouvel artefact entrant (déclenche SURFACE + LINK)
        now          : timestamp de référence (défaut : datetime.now())

        Retourne
        --------
        List[Action] triée par score décroissant (priorité la plus haute en premier)
        """
        if now is None:
            now = datetime.now()

        actions: List[Action] = []

        for ep in episodes:
            # ── Cycle de vie ──────────────────────────────────────────
            if ep.state == EpisodeState.DORMANT:
                if self._should_compress(ep, now):
                    actions.append(self._make_compress(ep, now))

            elif ep.state == EpisodeState.CONSOLIDATED:
                if self._should_archive(ep, now):
                    actions.append(self._make_archive(ep, now))

            # ── FLAG : épisodes ouverts ───────────────────────────────
            if ep.state in (EpisodeState.ACTIVE, EpisodeState.DORMANT):
                if self._is_open(ep):
                    actions.append(self._make_flag(ep))

            # ── SURFACE : proactivité sur nouvel artefact ─────────────
            if (new_artifact is not None
                    and ep.state in (EpisodeState.ACTIVE, EpisodeState.DORMANT)
                    and ep.centroid is not None
                    and new_artifact.goal_vector is not None):
                sim = _cosine(new_artifact.goal_vector, ep.centroid)
                if self._should_surface(ep, sim):
                    actions.append(self._make_surface(ep, new_artifact, sim))

        # ── LINK : connexions entre épisodes ─────────────────────────
        actions += self._detect_links(episodes)

        # Tri par score décroissant
        actions.sort(key=lambda a: a.score, reverse=True)
        return actions

    # ------------------------------------------------------------------ #
    # Règles de décision                                                   #
    # ------------------------------------------------------------------ #

    def _should_compress(self, ep: Episode, now: datetime) -> bool:
        """DORMANT + inactif depuis N jours + peu important."""
        age = _age_days(ep, now)
        if age is None:
            return False
        threshold = self.compress_after_days
        if self._is_open(ep):
            threshold *= self.open_compress_multiplier
        return age >= threshold

    def _should_archive(self, ep: Episode, now: datetime) -> bool:
        """CONSOLIDATED + inactif depuis N jours supplémentaires."""
        age = _age_days(ep, now)
        if age is None:
            return False
        return age >= (self.compress_after_days + self.archive_after_days)

    def _should_surface(self, ep: Episode, similarity: float) -> bool:
        """Épisode suffisamment similaire et important pour être suggéré."""
        return (similarity >= self.surface_min_similarity
                and ep.memory_score >= self.surface_min_importance)

    def _is_open(self, ep: Episode) -> bool:
        """Heuristique : l'épisode contient des signaux de question ouverte."""
        entities_text = " ".join(ep.entity_weights.keys()).lower()
        return any(kw in entities_text for kw in self.flag_open_keywords)

    def _detect_links(self, episodes: List[Episode]) -> List[Action]:
        """
        Détecte les paires d'épisodes partageant ≥ link_min_shared_entities.
        Seuls les épisodes ACTIVE et DORMANT sont considérés.
        O(n²) — acceptable jusqu'à ~5 000 épisodes.
        """
        candidates = [
            ep for ep in episodes
            if ep.state in (EpisodeState.ACTIVE, EpisodeState.DORMANT)
            and ep.entity_weights
        ]
        actions = []
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                ep_a, ep_b = candidates[i], candidates[j]
                shared = set(ep_a.entity_weights) & set(ep_b.entity_weights)
                if len(shared) >= self.link_min_shared_entities:
                    # Score = min des memory_scores × ratio entités partagées
                    n_union = len(set(ep_a.entity_weights) | set(ep_b.entity_weights))
                    jaccard = len(shared) / max(n_union, 1)
                    score   = min(ep_a.memory_score, ep_b.memory_score) * jaccard
                    actions.append(Action(
                        type      = ActionType.LINK,
                        episode   = ep_a,
                        episode_b = ep_b,
                        score     = round(score, 4),
                        reason    = (f"{len(shared)} entités partagées : "
                                     f"{', '.join(list(shared)[:5])}"),
                        context   = {"shared_entities": list(shared)},
                    ))
        return actions

    # ------------------------------------------------------------------ #
    # Constructeurs d'actions                                              #
    # ------------------------------------------------------------------ #

    def _make_compress(self, ep: Episode, now: datetime) -> Action:
        age = _age_days(ep, now)
        score = _urgency(age, self.compress_after_days, ep.memory_score)
        return Action(
            type    = ActionType.COMPRESS,
            episode = ep,
            score   = score,
            reason  = f"Inactif depuis {age:.0f} jours (seuil={self.compress_after_days}j)",
            context = {"age_days": age, "memory_score": ep.memory_score},
        )

    def _make_archive(self, ep: Episode, now: datetime) -> Action:
        age   = _age_days(ep, now)
        total = self.compress_after_days + self.archive_after_days
        score = _urgency(age, total, ep.memory_score)
        return Action(
            type    = ActionType.ARCHIVE,
            episode = ep,
            score   = score,
            reason  = f"Compressé et inactif depuis {age:.0f} jours (seuil={total}j)",
            context = {"age_days": age, "memory_score": ep.memory_score},
        )

    def _make_surface(
        self, ep: Episode, artifact: Artifact, similarity: float
    ) -> Action:
        score = similarity * ep.memory_score
        return Action(
            type    = ActionType.SURFACE,
            episode = ep,
            score   = round(score, 4),
            reason  = (f"Similarité {similarity:.2f} avec le nouvel artefact "
                       f"(seuil={self.surface_min_similarity})"),
            context = {
                "similarity":    round(similarity, 4),
                "artifact_id":   artifact.id,
                "artifact_text": artifact.content[:120],
            },
        )

    def _make_flag(self, ep: Episode) -> Action:
        matched = [
            kw for kw in self.flag_open_keywords
            if kw in " ".join(ep.entity_weights.keys()).lower()
        ]
        return Action(
            type    = ActionType.FLAG,
            episode = ep,
            score   = ep.memory_score,
            reason  = f"Épisode ouvert détecté (signaux : {matched[:3]})",
            context = {"matched_keywords": matched},
        )

    # ------------------------------------------------------------------ #
    # Résumé lisible                                                        #
    # ------------------------------------------------------------------ #

    def report(
        self,
        episodes:     List[Episode],
        new_artifact: Optional[Artifact] = None,
        now:          Optional[datetime] = None,
    ) -> str:
        """Retourne un rapport textuel des actions recommandées."""
        actions = self.evaluate(episodes, new_artifact=new_artifact, now=now)
        if not actions:
            return "Aucune action recommandée."

        by_type: Dict[ActionType, List[Action]] = {}
        for a in actions:
            by_type.setdefault(a.type, []).append(a)

        lines = [f"Decision Engine — {len(actions)} actions recommandées\n"]
        for atype, acts in by_type.items():
            lines.append(f"  [{atype.value}] × {len(acts)}")
            for a in acts[:3]:   # top 3 par type
                lines.append(f"    • {a.summary()}")
            if len(acts) > 3:
                lines.append(f"    … +{len(acts) - 3} autres")
        return "\n".join(lines)


# ================================================================== #
# Helpers                                                              #
# ================================================================== #

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus robuste (retourne 0 si vecteurs nuls)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _age_days(ep: Episode, now: datetime) -> Optional[float]:
    """Âge en jours depuis la dernière activité (time_interval[1])."""
    if ep.time_interval is None:
        return None
    t_end = ep.time_interval[1]
    if t_end is None:
        return None
    # Normalise tz si besoin
    if t_end.tzinfo is not None and now.tzinfo is None:
        now = now.replace(tzinfo=t_end.tzinfo)
    elif t_end.tzinfo is None and now.tzinfo is not None:
        t_end = t_end.replace(tzinfo=now.tzinfo)
    return max(0.0, (now - t_end).total_seconds() / 86400.0)


def _urgency(age_days: Optional[float], threshold: float, importance: float) -> float:
    """
    Score d'urgence 0→1.
    Croît avec l'âge relatif, décroît avec l'importance
    (un épisode important résiste plus longtemps).
    """
    if age_days is None:
        return 0.0
    ratio     = age_days / max(threshold, 1.0)
    dampening = 1.0 - 0.4 * importance   # importance élevée → score réduit de 40%
    return round(min(1.0, ratio * dampening), 4)
