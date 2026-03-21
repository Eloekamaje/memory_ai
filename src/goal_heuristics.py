"""
goal_heuristics.py — Inférence d'objectifs latents via zero-shot DeBERTa-v3-small.

Principe
--------
DeBERTa-v3-small (NLI) classifie chaque message contre un ensemble d'intents
prédéfinis, sans fine-tuning. Les scores softmax pondèrent les embeddings de
labels pour produire un goal_vector dans le même espace que mE5-base.

    goal_vector = Σ softmax(score_i) × embed("query: " + label_i)

Modèle : cross-encoder/nli-deberta-v3-small
  - ~135MB, ~2000 msgs/min GPU T4, ~500 msgs/min CPU
  - Entraîné sur SNLI + MultiNLI + ANLI → robuste sur les textes informels
  - Fonctionne sur FR (via transfert cross-lingual NLI)

Usage
-----
from goal_heuristics import compute_goal_vectors

goal_vecs = compute_goal_vectors(artifacts, embed_fn, device='cuda')
# goal_vecs : (n, 768) float32 — artifact.goal_vector assigné en place
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Intents — labels en anglais (NLI DeBERTa est anglais-dominant)
# ---------------------------------------------------------------------------

INTENT_LABELS: List[str] = [
    "requesting information or help",
    "planning and coordinating a meeting or event",
    "financial discussion or budget management",
    "task assignment or follow-up on action items",
    "sharing news or an update",
    "confirming or approving something",
    "making a decision or resolving a problem",
    "social chat or casual conversation",
    "technical problem solving or debugging",
    "producing sharing or reviewing a document",
]


# ---------------------------------------------------------------------------
# Zero-shot DeBERTa
# ---------------------------------------------------------------------------

def compute_goal_vectors(
    artifacts,
    embed_fn: Callable[[List[str]], np.ndarray],
    device: Optional[str] = None,
    batch_size: int = 64,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    verbose: bool = True,
) -> np.ndarray:
    """
    Calcule les goal_vectors via zero-shot DeBERTa et les assigne aux artefacts.

    Parameters
    ----------
    artifacts   : List[Artifact] — avec .content et .entities
    embed_fn    : callable (List[str] → np.ndarray) — même que le pipeline mE5
    device      : 'cuda' | 'cpu' | None (auto)
    batch_size  : messages traités par batch DeBERTa
    model_name  : modèle NLI HuggingFace
    verbose     : afficher la progression

    Returns
    -------
    goal_vectors : (n, dim) float32 — un vecteur par artefact
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError("pip install transformers")

    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"  Chargement {model_name} ...")

    classifier = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if device == "cuda" else -1,
    )

    # Pré-calculer les embeddings des labels (une seule fois)
    label_embeddings = embed_fn(["query: " + lbl for lbl in INTENT_LABELS])
    # label_embeddings : (n_labels, dim)

    n = len(artifacts)
    texts = [_prepare_text(a) for a in artifacts]
    goal_vectors = np.zeros((n, label_embeddings.shape[1]), dtype=np.float32)

    if verbose:
        print(f"  Classification zero-shot ({n} msgs, batch={batch_size}) ...")

    for start in range(0, n, batch_size):
        batch_texts = texts[start : start + batch_size]
        results = _run_classifier(classifier, batch_texts)
        for local_i, res in enumerate(results):
            goal_vectors[start + local_i] = _result_to_vector(res, label_embeddings)
        if verbose and (start + batch_size) % (batch_size * 10) == 0:
            print(f"    [{min(start + batch_size, n)}/{n}]")

    # Assigner aux artefacts
    for i, artifact in enumerate(artifacts):
        artifact.goal_vector = goal_vectors[i]

    if verbose:
        n_valid = int((np.linalg.norm(goal_vectors, axis=1) > 0).sum())
        print(f"  ✓ {n} goal vectors calculés  ({n_valid} non-nuls)")

    return goal_vectors


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _run_classifier(classifier, texts: List[str]) -> List[dict]:
    results = classifier(texts, candidate_labels=INTENT_LABELS, multi_label=False)
    return [results] if isinstance(results, dict) else results


def _result_to_vector(res: dict, label_embeddings: np.ndarray) -> np.ndarray:
    label_to_score = dict(zip(res["labels"], res["scores"]))
    scores = np.array([label_to_score[lbl] for lbl in INTENT_LABELS], dtype=np.float32)
    return _softmax(scores) @ label_embeddings

def _prepare_text(artifact) -> str:
    """Construit le texte d'entrée DeBERTa : contenu + entités dominantes."""
    text = artifact.content.strip() if artifact.content else ""
    if artifact.entities:
        # Ajouter les entités comme contexte (améliore la classification)
        ent_str = ", ".join(artifact.entities[:4])
        text = f"{text} [{ent_str}]"
    # Tronquer à 256 chars (DeBERTa supporte 512 tokens mais les msgs sont courts)
    return text[:256] if text else "."


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def infer_top_intent(artifact, classifier=None) -> str:
    """Retourne l'intent dominant d'un artefact (pour debug/inspection)."""
    if classifier is None:
        from transformers import pipeline as hf_pipeline
        classifier = hf_pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small",
        )
    text = _prepare_text(artifact)
    res = classifier(text, candidate_labels=INTENT_LABELS, multi_label=False)
    return res["labels"][0]


# ---------------------------------------------------------------------------
# Legacy — conservé pour compatibilité (utilisé par tests ou notebooks anciens)
# ---------------------------------------------------------------------------

_TYPE_INTENT_PHRASES = {
    "message":  "informal communication exchange",
    "email":    "formal written communication",
    "note":     "personal note-taking and recording ideas",
    "document": "creating or editing a formal document",
    "reunion":  "meeting coordination and group discussion",
    "decision": "making a decision or resolving an issue",
    "task":     "planning and executing a task",
}

_ACTION_PATTERNS = [
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


def infer_goal_phrases(artifact) -> List[str]:
    """Legacy heuristique — retourne les phrases d'intention sans DeBERTa."""
    phrases = [_TYPE_INTENT_PHRASES.get(artifact.artifact_type, "general activity")]
    for pattern, phrase in _ACTION_PATTERNS:
        if artifact.content and pattern.search(artifact.content):
            phrases.append(phrase)
    if artifact.entities:
        phrases.append("related to " + ", ".join(artifact.entities[:5]))
    return phrases
