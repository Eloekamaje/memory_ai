"""
memory_store.py — Persistance et aging du MemoryState (Phase B4).

Responsabilités :
  1. save()          — sérialise épisodes + artefacts + embeddings + résumés
  2. load()          — recharge l'état complet pour reconstruire RecallEngine
  3. archive_old()   — CONSOLIDATED → ARCHIVED après N jours
  4. compress_archived() — vide artifact_indices des archivés
                          (centroïdes conservés pour le recall sémantique)

Structure sur disque :
  <directory>/
    episodes.json       # List[Episode] sérialisé (centroïdes en liste float)
    artifacts.jsonl     # 1 Artifact par ligne (JSON Lines)
    embeddings.npy      # np.ndarray (n_artifacts, dim)
    goal_vectors.npz    # sparse : arrays "data" (k, dim) + "indices" (k,)
    summaries.json      # Dict[episode_id → EpisodeSummary] (optionnel)

Choix techniques :
  - JSON + JSONL : inspectables, diffables, pas de dépendances
  - .npy / .npz  : format numpy natif, zéro overhead pour les matrices
  - Centroïdes dans episodes.json (pas dans .npy) : petites tailles,
    colocalisés avec leurs métadonnées
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from episode_summarizer import EpisodeSummary
from models import Artifact, Episode, EpisodeState


# ================================================================== #
# Helpers sérialisation                                               #
# ================================================================== #

def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _vec_to_list(v: Optional[np.ndarray]) -> Optional[list]:
    if v is None:
        return None
    return v.tolist()


def _list_to_vec(l: Optional[list]) -> Optional[np.ndarray]:
    if l is None:
        return None
    return np.array(l, dtype=np.float32)


# ================================================================== #
# Sérialisation Episode                                               #
# ================================================================== #

def _episode_to_dict(ep: Episode) -> dict:
    t_start, t_end = (None, None) if ep.time_interval is None else ep.time_interval
    return {
        "id": ep.id,
        "artifact_indices": ep.artifact_indices,
        "centroid": _vec_to_list(ep.centroid),
        "goal_centroid": _vec_to_list(ep.goal_centroid),
        "entity_weights": ep.entity_weights,
        "time_start": _dt_to_str(t_start),
        "time_end": _dt_to_str(t_end),
        "memory_score": ep.memory_score,
        "state": ep.state.value,
    }


def _dict_to_episode(d: dict) -> Episode:
    t_start = _str_to_dt(d.get("time_start"))
    t_end = _str_to_dt(d.get("time_end"))
    time_interval = (t_start, t_end) if (t_start and t_end) else None

    return Episode(
        id=d["id"],
        artifact_indices=d["artifact_indices"],
        centroid=_list_to_vec(d.get("centroid")),
        goal_centroid=_list_to_vec(d.get("goal_centroid")),
        entity_weights=d.get("entity_weights", {}),
        time_interval=time_interval,
        memory_score=d.get("memory_score", 1.0),
        state=EpisodeState(d.get("state", "ACTIVE")),
    )


# ================================================================== #
# Sérialisation Artifact                                              #
# ================================================================== #

def _artifact_to_dict(art: Artifact) -> dict:
    return {
        "id": art.id,
        "timestamp": _dt_to_str(art.timestamp),
        "author": art.author,
        "content": art.content,
        "entities": art.entities,
        "source": art.source,
        "artifact_type": art.artifact_type,
        "importance": art.importance,
        "true_episode_id": art.true_episode_id,
        # goal_vector stocké séparément dans goal_vectors.npz
    }


def _dict_to_artifact(d: dict) -> Artifact:
    return Artifact(
        id=d["id"],
        timestamp=_str_to_dt(d["timestamp"]),
        author=d["author"],
        content=d["content"],
        entities=d.get("entities", []),
        source=d.get("source", "unknown"),
        artifact_type=d.get("artifact_type", "message"),
        importance=d.get("importance", 0.5),
        true_episode_id=d.get("true_episode_id"),
        # goal_vector réinjecté après chargement
    )


# ================================================================== #
# Sérialisation EpisodeSummary                                        #
# ================================================================== #

def _summary_to_dict(s: EpisodeSummary) -> dict:
    return {
        "episode_id": s.episode_id,
        "label": s.label,
        "summary": s.summary,
        "keywords": s.keywords,
        "top_entities": s.top_entities,
        "representative_quotes": s.representative_quotes,
        "mood": s.mood,
        "time_label": s.time_label,
        "artifact_count": s.artifact_count,
        "duration_hours": s.duration_hours,
    }


def _dict_to_summary(d: dict) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id=d["episode_id"],
        label=d["label"],
        summary=d["summary"],
        keywords=d.get("keywords", []),
        top_entities=d.get("top_entities", []),
        representative_quotes=d.get("representative_quotes", []),
        mood=d.get("mood", "neutral"),
        time_label=d.get("time_label", ""),
        artifact_count=d.get("artifact_count", 0),
        duration_hours=d.get("duration_hours", 0.0),
    )


# ================================================================== #
# MemoryStore                                                          #
# ================================================================== #

class MemoryStore:
    """
    Persistance et aging du MemoryState.

    Usage typique :
        store = MemoryStore("memory/")

        # Sauvegarder après segmentation
        store.save(episodes, artifacts, embeddings, summaries)

        # Recharger au démarrage suivant
        state = store.load()
        episodes  = state["episodes"]
        artifacts = state["artifacts"]
        embeddings = state["embeddings"]
        summaries  = state["summaries"]

        # Aging périodique
        archived = store.archive_old(episodes, days_threshold=90)
        store.compress_archived(episodes)
        store.save(episodes, artifacts, embeddings, summaries)
    """

    def __init__(self, directory: str = "memory"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- #
    # Chemins                                                          #
    # -------------------------------------------------------------- #

    @property
    def _episodes_path(self) -> Path:
        return self.directory / "episodes.json"

    @property
    def _artifacts_path(self) -> Path:
        return self.directory / "artifacts.jsonl"

    @property
    def _embeddings_path(self) -> Path:
        return self.directory / "embeddings.npy"

    @property
    def _goal_vectors_path(self) -> Path:
        return self.directory / "goal_vectors.npz"

    @property
    def _summaries_path(self) -> Path:
        return self.directory / "summaries.json"

    @property
    def _entity_types_path(self) -> Path:
        return self.directory / "entity_types.json"

    # -------------------------------------------------------------- #
    # save()                                                           #
    # -------------------------------------------------------------- #

    def save(
        self,
        episodes: List[Episode],
        artifacts: List[Artifact],
        embeddings: np.ndarray,
        summaries: Optional[List[EpisodeSummary]] = None,
        entity_types: Optional[Dict[str, str]] = None,
    ) -> None:
        """Sérialise l'état complet sur disque."""

        # --- episodes.json ---
        ep_dicts = [_episode_to_dict(ep) for ep in episodes]
        with open(self._episodes_path, "w", encoding="utf-8") as f:
            json.dump(ep_dicts, f, ensure_ascii=False, indent=2)

        # --- artifacts.jsonl ---
        with open(self._artifacts_path, "w", encoding="utf-8") as f:
            for art in artifacts:
                line = json.dumps(_artifact_to_dict(art), ensure_ascii=False)
                f.write(line + "\n")

        # --- embeddings.npy ---
        np.save(self._embeddings_path, embeddings)

        # --- goal_vectors.npz (sparse : seuls les non-None sont stockés) ---
        gv_indices = []
        gv_data = []
        for i, art in enumerate(artifacts):
            if art.goal_vector is not None:
                gv_indices.append(i)
                gv_data.append(art.goal_vector)

        if gv_indices:
            np.savez(
                self._goal_vectors_path,
                indices=np.array(gv_indices, dtype=np.int32),
                data=np.array(gv_data, dtype=np.float32),
            )

        # --- summaries.json ---
        if summaries is not None:
            summary_dicts = [_summary_to_dict(s) for s in summaries]
            with open(self._summaries_path, "w", encoding="utf-8") as f:
                json.dump(summary_dicts, f, ensure_ascii=False, indent=2)

        # --- entity_types.json : {canonical_name → type_string} ---
        if entity_types is not None:
            with open(self._entity_types_path, "w", encoding="utf-8") as f:
                json.dump(entity_types, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------- #
    # load()                                                           #
    # -------------------------------------------------------------- #

    def load(self) -> Dict:
        """
        Recharge l'état complet.

        Returns
        -------
        {
            "episodes":   List[Episode],
            "artifacts":  List[Artifact],
            "embeddings": np.ndarray,
            "summaries":  Dict[str, EpisodeSummary],   # peut être vide
        }
        """
        if not self._episodes_path.exists():
            raise FileNotFoundError(f"Aucun état sauvegardé dans {self.directory}")

        # --- episodes ---
        with open(self._episodes_path, encoding="utf-8") as f:
            episodes = [_dict_to_episode(d) for d in json.load(f)]

        # --- artifacts ---
        artifacts: List[Artifact] = []
        with open(self._artifacts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    artifacts.append(_dict_to_artifact(json.loads(line)))

        # --- embeddings ---
        embeddings = np.load(self._embeddings_path)

        # --- goal_vectors (réinjection dans les Artifact) ---
        if self._goal_vectors_path.exists():
            gv = np.load(self._goal_vectors_path)
            for idx, vec in zip(gv["indices"], gv["data"]):
                if idx < len(artifacts):
                    artifacts[idx].goal_vector = vec

        # --- summaries ---
        summaries: Dict[str, EpisodeSummary] = {}
        if self._summaries_path.exists():
            with open(self._summaries_path, encoding="utf-8") as f:
                for d in json.load(f):
                    s = _dict_to_summary(d)
                    summaries[s.episode_id] = s

        # --- entity_types ---
        entity_types: Dict[str, str] = {}
        if self._entity_types_path.exists():
            with open(self._entity_types_path, encoding="utf-8") as f:
                entity_types = json.load(f)

        return {
            "episodes": episodes,
            "artifacts": artifacts,
            "embeddings": embeddings,
            "summaries": summaries,
            "entity_types": entity_types,
        }

    # -------------------------------------------------------------- #
    # archive_old()                                                    #
    # -------------------------------------------------------------- #

    def archive_old(
        self,
        episodes: List[Episode],
        days_threshold: int = 90,
        reference_time: Optional[datetime] = None,
    ) -> int:
        """
        Passe les épisodes CONSOLIDATED en ARCHIVED s'ils sont plus vieux
        que *days_threshold* jours.

        Seuls les épisodes CONSOLIDATED sont archivés (pas les DORMANT) :
        la consolidation est le signal que l'épisode est "fermé".

        Returns
        -------
        Nombre d'épisodes archivés.
        """
        if reference_time is None:
            reference_time = datetime.now()

        archived = 0
        for ep in episodes:
            if ep.state != EpisodeState.CONSOLIDATED:
                continue
            if ep.time_interval is None:
                continue
            _, t_end = ep.time_interval
            age_days = (reference_time - t_end).days
            if age_days >= days_threshold:
                ep.state = EpisodeState.ARCHIVED
                archived += 1

        return archived

    # -------------------------------------------------------------- #
    # compress_archived()                                              #
    # -------------------------------------------------------------- #

    def compress_archived(self, episodes: List[Episode]) -> int:
        """
        Vide les artifact_indices des épisodes ARCHIVED.

        Le centroïde est conservé : l'épisode reste searchable via
        recall sémantique mais ses artefacts bruts ne sont plus chargés.

        Returns
        -------
        Nombre d'épisodes compressés.
        """
        compressed = 0
        for ep in episodes:
            if ep.state == EpisodeState.ARCHIVED and ep.artifact_indices:
                ep.artifact_indices = []
                compressed += 1
        return compressed

    # -------------------------------------------------------------- #
    # État du store                                                    #
    # -------------------------------------------------------------- #

    def summary(self, episodes: List[Episode]) -> str:
        """Résumé textuel de l'état de la mémoire persistée."""
        from collections import Counter
        state_counts = Counter(ep.state.value for ep in episodes)

        size_mb = sum(
            p.stat().st_size for p in self.directory.iterdir()
            if p.is_file()
        ) / 1e6

        lines = [
            f"MemoryStore @ {self.directory}",
            f"  Taille disque : {size_mb:.2f} MB",
            f"  Episodes      : {len(episodes)}",
        ]
        for state in ["ACTIVE", "DORMANT", "CONSOLIDATED", "ARCHIVED"]:
            n = state_counts.get(state, 0)
            if n:
                lines.append(f"    {state:14s}: {n}")
        return "\n".join(lines)
