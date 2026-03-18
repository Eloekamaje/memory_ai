"""
goal_heuristics.py — Inférence d'objectifs latents sans LLM.

Combine 3 signaux observables pour produire un goal_vector par artefact :
  1. Type d'artefact  → vecteur d'intention prédéfini
  2. Entités dominantes → pondération par fréquence
  3. Patterns lexicaux  → détection d'actions via regex

Le goal_vector résultant est un embedding de même dimension que les
embeddings sémantiques, obtenu par moyenne pondérée.
"""

import re
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# 1. Mapping type d'artefact → phrase d'intention
# ---------------------------------------------------------------------------

TYPE_INTENT_PHRASES: Dict[str, str] = {
    "message":   "informal communication exchange",
    "email":     "formal written communication",
    "note":      "personal note-taking and recording ideas",
    "document":  "creating or editing a formal document",
    "reunion":   "meeting coordination and group discussion",
    "decision":  "making a decision or resolving an issue",
    "task":      "planning and executing a task",
}

DEFAULT_INTENT_PHRASE = "general activity"


# ---------------------------------------------------------------------------
# 2. Patterns lexicaux → phrase d'action
# ---------------------------------------------------------------------------

ACTION_PATTERNS = [
    (re.compile(r"\b(demand|request|ask|demande|besoin)\b", re.IGNORECASE),
     "requesting information or action"),

    (re.compile(r"\b(send|envoy|transmit|forward)\b", re.IGNORECASE),
     "transmitting or sending something"),

    (re.compile(r"\b(prepar|creat|write|rédige|produi)\b", re.IGNORECASE),
     "producing or creating content"),

    (re.compile(r"\b(decid|décid|resolv|résol|chois|choose)\b", re.IGNORECASE),
     "making a decision or resolving"),

    (re.compile(r"\b(meet|réunion|rencontr|discuss)\b", re.IGNORECASE),
     "coordinating a meeting or discussion"),

    (re.compile(r"\b(pay|paie|budget|financ|cost|coût)\b", re.IGNORECASE),
     "financial planning or budget management"),

    (re.compile(r"\b(confirm|valide|approv|accept)\b", re.IGNORECASE),
     "confirming or approving something"),

    (re.compile(r"\b(correct|fix|modif|change|update)\b", re.IGNORECASE),
     "correcting or updating something"),
]


# ---------------------------------------------------------------------------
# 3. Calcul du goal_vector
# ---------------------------------------------------------------------------

def infer_goal_phrases(artifact) -> List[str]:
    """
    Retourne les phrases d'intention combinées pour un artefact.
    Pas d'embeddings ici — juste les phrases textuelles.
    """

    phrases = []

    # signal 1 : type d'artefact
    intent = TYPE_INTENT_PHRASES.get(artifact.artifact_type, DEFAULT_INTENT_PHRASE)
    phrases.append(intent)

    # signal 2 : patterns lexicaux détectés dans le contenu
    for pattern, action_phrase in ACTION_PATTERNS:
        if pattern.search(artifact.content):
            phrases.append(action_phrase)

    # signal 3 : entités dominantes (ajoutées comme contexte)
    if artifact.entities:
        entity_context = "related to " + ", ".join(artifact.entities[:5])
        phrases.append(entity_context)

    return phrases


def compute_goal_vectors(artifacts, embed_fn) -> np.ndarray:
    """
    Calcule les goal_vectors pour une liste d'artefacts.

    Parameters
    ----------
    artifacts : List[Artifact]
    embed_fn : callable
        Fonction d'embedding (textes → np.ndarray), ex: embed_texts

    Returns
    -------
    goal_vectors : np.ndarray de shape (n_artifacts, embedding_dim)
    """

    all_phrases = []
    phrase_counts = []

    for artifact in artifacts:
        phrases = infer_goal_phrases(artifact)
        all_phrases.extend(phrases)
        phrase_counts.append(len(phrases))

    # embed toutes les phrases en un seul batch
    if not all_phrases:
        # fallback : vecteurs nuls
        dim = 384  # dimension par défaut all-MiniLM-L6-v2
        return np.zeros((len(artifacts), dim))

    all_embeddings = embed_fn(all_phrases)

    # reconstruire un goal_vector par artefact (moyenne de ses phrases)
    goal_vectors = []
    offset = 0

    for count in phrase_counts:
        if count > 0:
            chunk = all_embeddings[offset: offset + count]
            goal_vectors.append(np.mean(chunk, axis=0))
        else:
            goal_vectors.append(np.zeros(all_embeddings.shape[1]))
        offset += count

    goal_vectors = np.array(goal_vectors)

    # assigner aux artefacts
    for i, artifact in enumerate(artifacts):
        artifact.goal_vector = goal_vectors[i]

    return goal_vectors
