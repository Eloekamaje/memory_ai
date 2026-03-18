"""
eval_on_gold.py — Évalue l'ARI du pipeline V3 sur le dataset gold réel.

Usage :
    python tools/eval_on_gold.py \
        --data data/group_anon.txt \
        --gold data/group_gold.json

Compare la segmentation V3 avec les frontières annotées manuellement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding_engine import embed_texts
from entity_extractor import EntityExtractor
from entity_resolver import EntityResolver, enrich_artifacts_with_entities
from episode_algorithm import EpisodeSegmenter
from episode_splitter import EpisodeSplitter, SplitConfig
from goal_heuristics import compute_goal_vectors
from parsers.whatsapp_parser import parse_whatsapp_chat


# ══════════════════════════════════════════════════════════════════════════════
# Chargement du gold
# ══════════════════════════════════════════════════════════════════════════════

def load_gold_labels(gold_path: str, n_artifacts: int) -> list:
    """
    Convertit les épisodes gold (boundaries) en vecteur de labels
    aligné sur les artefacts parsés.

    gold_episodes[i] contient l'episode_id gold du message i.
    """
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    labels = [None] * n_artifacts
    for ep in gold["episodes"]:
        ep_id = ep["episode_id"]
        for idx in range(ep["start_idx"], ep["end_idx"] + 1):
            if idx < n_artifacts:
                labels[idx] = ep_id

    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline V3 (segmentation seule, sans graph/summarizer)
# ══════════════════════════════════════════════════════════════════════════════

def run_segmentation(artifacts, embeddings, seg_kwargs: dict) -> list:
    """Lance EpisodeSegmenter + consolidation + EpisodeSplitter."""
    segmenter = EpisodeSegmenter(**seg_kwargs)
    episodes  = segmenter.segment(artifacts, embeddings)
    episodes  = segmenter.consolidate(episodes)

    splitter = EpisodeSplitter(SplitConfig(
        min_cohesion=0.65, min_size_to_split=8,
        max_span_hours=168.0, max_splits=6,
        min_sub_size=3, silhouette_threshold=0.10,
    ))
    episodes = splitter.split(episodes, artifacts, embeddings)
    return episodes


def build_pred_vector(artifacts, episodes) -> list:
    pred = [None] * len(artifacts)
    for ep in episodes:
        for idx in ep.artifact_indices:
            if idx < len(pred):
                pred[idx] = ep.id
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true: list, y_pred: list) -> dict:
    """ARI + NMI sur les messages avec label gold non-None."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not pairs:
        return {"ari": float("nan"), "nmi": float("nan"), "n_eval": 0}
    yt, yp = zip(*pairs)
    return {
        "ari":    adjusted_rand_score(yt, yp),
        "nmi":    normalized_mutual_info_score(yt, yp),
        "n_eval": len(pairs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/group_anon.txt")
    parser.add_argument("--gold", default="data/group_gold.json")
    parser.add_argument("--hard-break", type=float, default=720,
                        help="hard_break_minutes (0 = désactivé)")
    parser.add_argument("--skip-ner", action="store_true",
                        help="Sauter NER + goals (plus rapide, delta/rho désactivés)")
    args = parser.parse_args()

    print(f"\n  Données : {args.data}")
    print(f"  Gold    : {args.gold}")
    print()

    # ── 1. Parse ───────────────────────────────────────────────────────────
    print("  [1/5] Parsing ...")
    artifacts = parse_whatsapp_chat(args.data)
    print(f"        {len(artifacts)} messages")

    # ── 2. Gold labels ─────────────────────────────────────────────────────
    print("  [2/5] Chargement gold ...")
    y_true = load_gold_labels(args.gold, len(artifacts))
    n_gold = sum(1 for l in y_true if l is not None)
    n_ep_gold = len(set(l for l in y_true if l is not None))
    print(f"        {n_gold} messages labellisés, {n_ep_gold} épisodes gold")

    # ── 3. Embeddings ──────────────────────────────────────────────────────
    print("  [3/5] Embeddings ...")
    texts      = [a.content for a in artifacts]
    embeddings = embed_texts(texts).astype(np.float32)

    # ── 4. NER + goals ─────────────────────────────────────────────────────
    if args.skip_ner:
        print("  [4/5] NER + goals ... (ignoré via --skip-ner)")
    else:
        print("  [4/5] NER + goals ...")
        extractor = EntityExtractor()
        mentions  = extractor.extract_batch(artifacts)
        resolver  = EntityResolver(string_threshold=0.75)
        resolver.resolve(mentions)
        enrich_artifacts_with_entities(artifacts, resolver)

        gv = compute_goal_vectors(artifacts, embed_texts)
        for i, art in enumerate(artifacts):
            if gv is not None and i < len(gv):
                v = gv[i]
                art.goal_vector = v if not np.all(v == 0) else None

    # ── 5. Segmentation V3 ────────────────────────────────────────────────
    print("  [5/5] Segmentation V3 ...")
    seg_kwargs = dict(
        time_threshold_minutes=120,
        attach_threshold=0.30,
        alpha=0.45, beta=0.25, gamma=0.10, delta=0.20, rho=0.05,
        dormancy_minutes=1440,
        ema_alpha=0.80,
        active_penalty_hours=24.0,
        hard_break_minutes=args.hard_break,
        allow_reactivation=True,
    )
    episodes = run_segmentation(artifacts, embeddings, seg_kwargs)
    y_pred   = build_pred_vector(artifacts, episodes)

    n_ep_pred = len(episodes)
    print(f"        {n_ep_pred} épisodes prédits")

    # ── Résultats ──────────────────────────────────────────────────────────
    scores = evaluate(y_true, y_pred)

    print(f"""
  ╔══════════════════════════════════════════════════╗
  ║   ÉVALUATION V3 — données réelles (gold)         ║
  ╠══════════════════════════════════════════════════╣
  ║  ARI   : {scores['ari']:+.4f}                              ║
  ║  NMI   : {scores['nmi']:.4f}                               ║
  ║  Msgs  : {scores['n_eval']} évalués                       ║
  ║  Gold  : {n_ep_gold} épisodes                           ║
  ║  Pred  : {n_ep_pred} épisodes                           ║
  ╚══════════════════════════════════════════════════╝
""")

    if scores['ari'] < 0.20:
        print("  → ARI faible : le pipeline sur-fragmente ou sous-segmente")
        print(f"    Gold={n_ep_gold} épisodes vs Pred={n_ep_pred}")
        if n_ep_pred > n_ep_gold * 1.5:
            print("    Sur-fragmentation — réduire hard_break ou attach_threshold")
        elif n_ep_pred < n_ep_gold * 0.6:
            print("    Sous-segmentation — augmenter attach_threshold")
    elif scores['ari'] < 0.40:
        print("  → ARI moyen : base correcte, paramètres à affiner sur données réelles")
    else:
        print("  → ARI solide sur données réelles — bonne généralisation")


if __name__ == "__main__":
    main()
