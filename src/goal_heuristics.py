"""
goal_heuristics.py — Structure épisodique via zero-shot DeBERTa-v3-small.

Ancrage théorique
-----------------
Basé sur la mémoire épisodique humaine (Tulving 1972, Zacks Event Segmentation
Theory 2007, Conway 2000) : un épisode mémorable se distingue par son statut
de résolution, sa saillance émotionnelle et sa position dans le flux temporel.

Trois axes orthogonaux (9 labels) :
  AXE 1 — Résolution    : l'épisode est-il ouvert ou fermé ?
  AXE 2 — Saillance     : l'échange est-il marquant ou banal ?
  AXE 3 — Temporalité   : ouverture, continuation ou réactivation ?

goal_vector = Σ softmax(score_i) × embed("query: " + label_i)
  → vecteur 768d dans le même espace que mE5-base
  → axis_scores (n, 3) float32 séparément pour inspection et features TCN

Modèle : cross-encoder/nli-deberta-v3-small (~135MB)
  ~2000 msgs/min GPU T4 · ~500 msgs/min CPU

Usage
-----
from goal_heuristics import compute_goal_vectors, compute_episode_scores

goal_vecs   = compute_goal_vectors(artifacts, embed_fn, device='cuda')
axis_scores = compute_episode_scores(artifacts, device='cuda')
# axis_scores : (n, 3) — [resolution, salience, temporality] ∈ [0, 1]
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 3 axes · 9 labels — ancrés dans la mémoire épisodique humaine
# ---------------------------------------------------------------------------

# Axe 1 — Résolution : l'épisode avance-t-il vers une clôture ?
_AXE_RESOLUTION = [
    "open problem or unresolved question waiting for an answer",
    "decision reached or outcome confirmed and agreed upon",
    "task assigned with a clear owner and deadline",
]

# Axe 2 — Saillance : cet échange est-il mémorable ?
_AXE_SALIENCE = [
    "emotionally charged intense surprising or high-stakes exchange",
    "neutral factual routine coordination",
    "casual social bonding small talk with no concrete goal",
]

# Axe 3 — Temporalité épisodique : où en est-on dans le cycle de l'épisode ?
_AXE_TEMPORALITY = [
    "initiating a brand new topic or situation",
    "continuing or deepening an ongoing discussion",
    "reactivating or explicitly referencing a past episode",
]

INTENT_LABELS: List[str] = _AXE_RESOLUTION + _AXE_SALIENCE + _AXE_TEMPORALITY

# Indices des axes dans INTENT_LABELS (pour axis_scores)
AXIS_RESOLUTION  = slice(0, 3)
AXIS_SALIENCE    = slice(3, 6)
AXIS_TEMPORALITY = slice(6, 9)

AXIS_NAMES = ["resolution", "salience", "temporality"]


# ---------------------------------------------------------------------------
# goal_vector — représentation de la structure épisodique dans l'espace mE5
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
    Calcule les goal_vectors et les assigne aux artefacts.

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
    goal_vectors : (n, 768) float32 — artifact.goal_vector assigné en place
    """
    classifier, label_embeddings = _load_classifier(
        model_name, embed_fn, device, verbose
    )

    n = len(artifacts)
    texts = [_prepare_text(a) for a in artifacts]
    goal_vectors = np.zeros((n, label_embeddings.shape[1]), dtype=np.float32)

    if verbose:
        print(f"  goal_vector ({n} msgs, batch={batch_size}) ...")

    for start in range(0, n, batch_size):
        results = _run_classifier(classifier, texts[start : start + batch_size])
        for local_i, res in enumerate(results):
            goal_vectors[start + local_i] = _result_to_vector(res, label_embeddings)
        if verbose and (start + batch_size) % (batch_size * 10) == 0:
            print(f"    [{min(start + batch_size, n)}/{n}]")

    for i, artifact in enumerate(artifacts):
        artifact.goal_vector = goal_vectors[i]

    if verbose:
        print(f"  ✓ {n} goal vectors  shape={goal_vectors.shape}")

    return goal_vectors


# ---------------------------------------------------------------------------
# axis_scores — scores scalaires par axe (pour TCN features et inspection)
# ---------------------------------------------------------------------------

def compute_episode_scores(
    artifacts,
    device: Optional[str] = None,
    batch_size: int = 64,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    verbose: bool = True,
) -> np.ndarray:
    """
    Calcule les scores des 3 axes épisodiques pour chaque artefact.

    Chaque axe est un score P(dominant label in axis) ∈ [0, 1] :
      axis 0 — resolution  : 1.0 = décision/tâche fermée · 0.0 = problème ouvert
      axis 1 — salience    : 1.0 = échange intense/surprise · 0.0 = bavardage neutre
      axis 2 — temporality : 1.0 = nouveau sujet · 0.5 = continuation · 0.0 = réactivation

    Utilisable comme features supplémentaires pour le TCN (après boundary_detector).

    Returns
    -------
    axis_scores : (n, 3) float32
    """
    import torch
    from transformers import pipeline as hf_pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"  Chargement {model_name} ...")
    classifier = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if device == "cuda" else -1,
    )

    n = len(artifacts)
    texts = [_prepare_text(a) for a in artifacts]
    axis_scores = np.zeros((n, 3), dtype=np.float32)

    if verbose:
        print(f"  episode_scores ({n} msgs, batch={batch_size}) ...")

    for start in range(0, n, batch_size):
        results = _run_classifier(classifier, texts[start : start + batch_size])
        for local_i, res in enumerate(results):
            axis_scores[start + local_i] = _result_to_axis_scores(res)
        if verbose and (start + batch_size) % (batch_size * 10) == 0:
            print(f"    [{min(start + batch_size, n)}/{n}]")

    if verbose:
        for j, name in enumerate(AXIS_NAMES):
            print(f"  {name:15s} : moy={axis_scores[:, j].mean():.3f}  "
                  f"std={axis_scores[:, j].std():.3f}")

    return axis_scores


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _load_classifier(model_name, embed_fn, device, verbose):
    import torch
    from transformers import pipeline as hf_pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"  Chargement {model_name} ...")

    classifier = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if device == "cuda" else -1,
    )
    label_embeddings = embed_fn(["query: " + lbl for lbl in INTENT_LABELS])
    return classifier, label_embeddings


def _run_classifier(classifier, texts: List[str]) -> List[dict]:
    results = classifier(texts, candidate_labels=INTENT_LABELS, multi_label=False)
    return [results] if isinstance(results, dict) else results


def _result_to_scores_array(res: dict) -> np.ndarray:
    """Retourne les scores dans l'ordre de INTENT_LABELS."""
    label_to_score = dict(zip(res["labels"], res["scores"]))
    return np.array([label_to_score[lbl] for lbl in INTENT_LABELS], dtype=np.float32)


def _result_to_vector(res: dict, label_embeddings: np.ndarray) -> np.ndarray:
    scores = _result_to_scores_array(res)
    return _softmax(scores) @ label_embeddings


def _result_to_axis_scores(res: dict) -> np.ndarray:
    """
    Réduit les 9 scores en 3 scores d'axe.
    Score d'axe = max des scores dans l'axe (label dominant).
    """
    scores = _result_to_scores_array(res)
    return np.array([
        scores[AXIS_RESOLUTION].max(),
        scores[AXIS_SALIENCE].max(),
        scores[AXIS_TEMPORALITY].max(),
    ], dtype=np.float32)


def _prepare_text(artifact) -> str:
    """Texte d'entrée DeBERTa : contenu + entités comme contexte."""
    text = artifact.content.strip() if artifact.content else ""
    if artifact.entities:
        text = f"{text} [{', '.join(artifact.entities[:4])}]"
    return text[:256] if text else "."


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def describe_artifact_episode_structure(artifact) -> dict:
    """
    Retourne une description lisible de la structure épisodique d'un artefact.
    Nécessite que artifact.goal_vector soit déjà calculé.

    Returns
    -------
    dict avec 'top_label', 'axis_scores' (dict), 'text_preview'
    """
    if artifact.goal_vector is None:
        return {"error": "goal_vector non calculé — appeler compute_goal_vectors d'abord"}

    return {
        "text_preview": (artifact.content or "")[:80],
        "entities": artifact.entities[:5] if artifact.entities else [],
        "note": "Appeler compute_episode_scores() pour les axis_scores scalaires",
    }


# ---------------------------------------------------------------------------
# Legacy — conservé pour compatibilité
# ---------------------------------------------------------------------------

def infer_goal_phrases(artifact) -> List[str]:
    """Legacy heuristique regex — sans DeBERTa, pour usage offline."""
    _type_map = {
        "message": "informal communication exchange",
        "email": "formal written communication",
        "note": "personal note-taking",
        "document": "creating or editing a document",
        "reunion": "meeting coordination",
        "decision": "making a decision",
        "task": "planning a task",
    }
    _patterns = [
        (re.compile(r"\b(demand|request|ask|demande|besoin)\b", re.IGNORECASE),
         "open problem or unresolved question"),
        (re.compile(r"\b(décidé|décision|validé|confirmé|ok go)\b", re.IGNORECASE),
         "decision reached or outcome confirmed"),
        (re.compile(r"\b(chargé|responsable|deadline|pour lundi)\b", re.IGNORECASE),
         "task assigned with a clear owner"),
        (re.compile(r"\b(urgent|important|problème|bloqué|incroyable)\b", re.IGNORECASE),
         "emotionally charged or high-stakes exchange"),
        (re.compile(r"\b(tu te souviens|comme on avait|suite à|rappel)\b", re.IGNORECASE),
         "reactivating or referencing a past episode"),
    ]
    phrases = [_type_map.get(artifact.artifact_type, "general activity")]
    for pattern, phrase in _patterns:
        if artifact.content and pattern.search(artifact.content):
            phrases.append(phrase)
    if artifact.entities:
        phrases.append("related to " + ", ".join(artifact.entities[:4]))
    return phrases
