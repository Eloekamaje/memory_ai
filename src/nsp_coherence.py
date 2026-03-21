"""
nsp_coherence.py — Utterance-Pair Coherence Scoring via BERT NSP.

Inspiré de CSMSegmenter (SuperDialseg) et du paper :
    "Improving Unsupervised Dialogue Topic Segmentation with
     Utterance-Pair Coherence Scoring" (Jiang et al., 2023)

Principe
--------
Pour chaque paire consécutive (u_{t-1}, u_t), BERT NSP prédit
P(u_t est la suite naturelle de u_{t-1}).

    scores[t] = P(is_next | u_{t-1}, u_t)  — cohérence discursive
    scores[0] = 0.0                          — pas de prédécesseur

Interprétation :
    score élevé (≈ 1.0) → même topic, continuation naturelle
    score faible (≈ 0.0) → rupture discursive → frontière probable

Modèle recommandé (FR + EN) : 'bert-base-multilingual-cased'
    Pré-entraîné NSP sur 104 langues → zero-shot sans fine-tuning.

Usage
-----
from nsp_coherence import compute_nsp_scores

scores = compute_nsp_scores(
    utterances,
    device='cuda',
    backbone='bert-base-multilingual-cased',
)
# scores : (n,) float32, 0.0 = incoherent, 1.0 = coherent
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional


def compute_nsp_scores(
    utterances: List[str],
    device: Optional[str] = None,
    backbone: str = 'bert-base-multilingual-cased',
    batch_size: int = 64,
    max_length: int = 128,
    verbose: bool = True,
) -> np.ndarray:
    """
    Calcule les scores de cohérence NSP pour chaque message.

    Parameters
    ----------
    utterances  : liste de textes bruts (sans prefix mE5)
    device      : 'cuda' | 'cpu' | None (auto)
    backbone    : modèle BERT avec tête NSP pré-entraînée
                  'bert-base-multilingual-cased'  → FR+EN (recommandé)
                  'bert-base-uncased'             → EN uniquement
    batch_size  : nombre de paires par batch GPU
    max_length  : longueur max en tokens (truncation)

    Returns
    -------
    scores : (n,) float32
        scores[t] = P(u_t suit naturellement u_{t-1})
        scores[0] = 0.0 (pas de prédécesseur)
    """
    try:
        from transformers import BertForNextSentencePrediction, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print(f'  Chargement {backbone} ...')

    model     = BertForNextSentencePrediction.from_pretrained(backbone)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model.to(device).eval()

    n = len(utterances)
    scores = np.zeros(n, dtype=np.float32)

    if n < 2:
        return scores

    texts_a = utterances[:-1]   # u_{t-1}
    texts_b = utterances[1:]    # u_t

    pair_scores: List[float] = []

    if verbose:
        print(f'  Scoring {n-1} paires sur {device} (batch={batch_size}) ...')

    with torch.no_grad():
        for start in range(0, len(texts_a), batch_size):
            batch_a = texts_a[start:start + batch_size]
            batch_b = texts_b[start:start + batch_size]

            inputs = tokenizer(
                batch_a, batch_b,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # logits : (batch, 2)
            # classe 0 = IsNextSentence, classe 1 = NotNextSentence
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)[:, 0]  # P(is_next)
            pair_scores.extend(probs.cpu().float().numpy().tolist())

    scores[1:] = np.array(pair_scores, dtype=np.float32)

    if verbose:
        n_low = int((scores[1:] < 0.3).sum())
        print(f'  ✓ {n} scores calculés | '
              f'moy={scores[1:].mean():.3f} | '
              f'cohérence faible (<0.3) : {n_low} ({100*n_low/(n-1):.1f}%)')

    return scores


def nsp_boundary_predictions(
    scores: np.ndarray,
    cut_rate: float = 1.0,
) -> np.ndarray:
    """
    Convertit les scores de cohérence NSP en labels binaires de frontière.

    Utilise la méthode depth-score de TextTiling (Hearst, 1997)
    sur les scores d'incohérence (1 - scores NSP).

    Parameters
    ----------
    scores   : (n,) float32 — P(is_next) par message
    cut_rate : seuil = mean + cut_rate × std (1.0 = standard)

    Returns
    -------
    predictions : (n,) int — 1 = frontière, 0 = continuation
    """
    incoherence = 1.0 - scores  # faible cohérence = score élevé = frontière

    depth_scores = _depth_score_cal(incoherence[1:])  # (n-1,) scores aux positions 1..n-1
    mean = float(np.mean(depth_scores))
    std  = float(np.std(depth_scores))
    threshold = mean + cut_rate * std

    predictions = np.zeros(len(scores), dtype=int)
    for i, ds in enumerate(depth_scores):
        if ds > threshold:
            predictions[i + 1] = 1

    return predictions


def _depth_score_cal(scores: np.ndarray) -> List[float]:
    """
    Depth score à la TextTiling : profondeur de la vallée en position i.

        depth[i] = 0.5 × (left_peak + right_peak - 2 × scores[i])

    Port direct de SuperDialseg/utils_texttiling.py.
    """
    output = []
    n = len(scores)
    for i in range(n):
        lflag = scores[i]
        for j in range(i - 1, -1, -1):
            if scores[j] >= lflag:
                lflag = scores[j]
            else:
                break
        rflag = scores[i]
        for j in range(i + 1, n):
            if scores[j] >= rflag:
                rflag = scores[j]
            else:
                break
        output.append(0.5 * (lflag + rflag - 2 * scores[i]))
    return output


def save_nsp_scores(scores: np.ndarray, path: str) -> None:
    np.save(path, scores)
    print(f'✓ NSP scores sauvegardés → {path}  shape={scores.shape}')


def load_nsp_scores(path: str) -> np.ndarray:
    scores = np.load(path)
    print(f'✓ NSP scores chargés depuis {path}  shape={scores.shape}')
    return scores
