"""
context_embedder.py — Encodage contextuel fenêtré pour mE5-base.

Principe (H2.b) :
    "Ok" encodé seul          → embedding générique, peu discriminant
    "Ok" encodé avec contexte → embedding spécifique au sujet en cours

Fenêtre causale [t-window .. t] :
    - On regarde en arrière uniquement (causal) — cohérent avec la segmentation
    - On inclut l'auteur de chaque message → signal de continuité/rupture
    - Séparateur " | " → compatible avec mE5-base (512 tokens max)

Usage
-----
from context_embedder import build_context_texts

texts = build_context_texts(artifacts, window=4)
embeddings = model.encode(texts, ...)   # même pipeline qu'avant

Cache recommandé : group_embeddings_me5_ctx5.npy (window=4 → 5 messages)
"""

from __future__ import annotations
from typing import List


def build_context_texts(
    artifacts,
    window: int = 4,
    prefix: str = "passage: ",
    sep: str = " | ",
    include_author: bool = True,
) -> List[str]:
    """
    Construit les textes d'entrée mE5 avec fenêtre contextuelle causale.

    Pour chaque message t, concatène les `window` messages précédents + t :
        "[Alice] msg_{t-3} | [Bob] msg_{t-2} | [Alice] msg_{t-1} | [Alice] msg_t"

    Parameters
    ----------
    artifacts     : liste d'Artifact (avec .author et .content)
    window        : nombre de messages précédents à inclure (4 → fenêtre de 5)
    prefix        : préfixe requis par mE5 ("passage: " pour les documents)
    sep           : séparateur entre messages dans la fenêtre
    include_author: inclure [Author] devant chaque message

    Returns
    -------
    texts : List[str] — un texte contextuel par artifact, prêt pour model.encode()
    """
    texts = []
    for i, art in enumerate(artifacts):
        start       = max(0, i - window)
        window_arts = artifacts[start : i + 1]

        if include_author:
            parts = [f"[{a.author}] {a.content}" for a in window_arts]
        else:
            parts = [a.content for a in window_arts]

        context = sep.join(parts)
        texts.append(prefix + context)

    return texts
