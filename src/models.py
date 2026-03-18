from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class Artifact:
    """Unité brute de mémoire : message, email, note, document, réunion…"""

    id: str
    timestamp: datetime
    author: str
    content: str

    # enrichissements
    entities: List[str] = field(default_factory=list)
    source: str = "unknown"            # whatsapp, email, csv, …
    artifact_type: str = "message"     # message, email, note, document, reunion, decision
    importance: float = 0.5            # 0→1, importance locale initiale

    # vecteur d'objectif latent (inféré par heuristiques ou LLM)
    goal_vector: Optional[np.ndarray] = field(default=None, repr=False)

    # vérité terrain (évaluation uniquement)
    true_episode_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

class EpisodeState(Enum):
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    CONSOLIDATED = "CONSOLIDATED"
    ARCHIVED = "ARCHIVED"


@dataclass
class Episode:
    """Épisode de mémoire : regroupement dynamique d'artefacts."""

    id: str
    artifact_indices: List[int] = field(default_factory=list)

    # représentation sémantique
    centroid: Optional[np.ndarray] = field(default=None, repr=False)
    goal_centroid: Optional[np.ndarray] = field(default=None, repr=False)

    # entités pondérées  {entité: poids}
    entity_weights: Dict[str, float] = field(default_factory=dict)

    # intervalle temporel [start, end]
    time_interval: Optional[Tuple[datetime, datetime]] = None

    # score de saillance mémorielle
    memory_score: float = 1.0

    # état courant
    state: EpisodeState = EpisodeState.ACTIVE
