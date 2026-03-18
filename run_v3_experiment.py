#!/usr/bin/env python3
"""
run_v3_experiment.py — Mesure l'impact ARI des fixes Challenger V3.

Baseline rappelé : ARI=0.506 (V2 entity-boosted, alpha=0.45, beta=0.25)

Configurations testées :
  V2_ref   : params V2 entity-boosted, algorithme V3 (isoler impact algo)
  V3_ema   : + EMA centroid seulement (ema_alpha=0.15)
  V3_ent   : + entity overlap normalization seulement
  V3_age   : + AgePenalty sur ACTIVE seulement
  V3_full  : tous les fixes, sans hard_break
  V3_hb    : tous les fixes + hard_break=720min
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from experiment_runner import run_experiment

DATA_FILE = "synthetic_200.csv"

# Params entity-boosted V2 (reproduire baseline 0.506)
BASE = dict(
    alpha=0.45,
    beta=0.25,
    gamma=0.10,
    delta=0.20,
    rho=0.05,
    attach_threshold=0.30,
    dormancy_minutes=1440,
    use_goals=True,
    use_entity_resolution=True,
    ner_backend="spacy",
    build_graph=False,   # rapide, pas utile pour l'ARI
    hard_break_minutes=0,
)

configs = {
    "V2_ref  (algo V3, params V2)": {
        **BASE,
        # ema_alpha et active_penalty_hours par défaut dans EpisodeSegmenter
        # mais on ne les passe pas ici car experiment_runner n'a pas ces params
        # → on les ajoutera en patchant directement EpisodeSegmenter dans la section dédiée
    },
    "V3_full (EMA+EntNorm+AgePen, no hard_break)": {
        **BASE,
        "hard_break_minutes": 0,
    },
    "V3_hb   (V3_full + hard_break=720min)": {
        **BASE,
        "hard_break_minutes": 720,
    },
}

# run_experiment ne passe pas ema_alpha / active_penalty_hours
# On patch EpisodeSegmenter pour les 3 variantes isolées
from episode_algorithm import EpisodeSegmenter as _Seg
from dataset_loader import load_dataset
from embedding_engine import embed_texts
from goal_heuristics import compute_goal_vectors
from entity_extractor import EntityExtractor
from entity_resolver import EntityResolver, enrich_artifacts_with_entities
from metrics import evaluate_segmentation
from episode_splitter import EpisodeSplitter, SplitConfig
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", DATA_FILE)


def run_quick(label, seg_kwargs, use_goals=True, hard_break=0):
    """Run allégé : pas de graph, pas de summaries."""
    print(f"\n{'▓'*60}")
    print(f"  {label}")
    print(f"{'▓'*60}")

    artifacts = load_dataset(DATA_PATH)

    # NER
    extractor = EntityExtractor(backend="spacy")
    mentions = extractor.extract_batch(artifacts)
    resolver = EntityResolver(string_threshold=0.75, use_embeddings=False, min_mention_confidence=0.50)
    resolver.resolve(mentions)
    enrich_artifacts_with_entities(artifacts, resolver)

    embeddings = embed_texts([a.content for a in artifacts]).astype(np.float32)

    if use_goals:
        gv = compute_goal_vectors(artifacts, embed_texts)
        for i, art in enumerate(artifacts):
            if gv is not None and i < len(gv):
                v = gv[i]
                art.goal_vector = v if not np.all(v == 0) else None

    segmenter = _Seg(
        hard_break_minutes=hard_break,
        **seg_kwargs,
    )
    episodes = segmenter.segment(artifacts, embeddings)
    episodes = segmenter.consolidate(episodes)

    # Splitter
    splitter = EpisodeSplitter(SplitConfig(
        min_cohesion=0.65, min_size_to_split=8,
        max_span_hours=168.0, max_splits=6,
        min_sub_size=3, silhouette_threshold=0.10,
    ))
    episodes = splitter.split(episodes, artifacts, embeddings)

    r = evaluate_segmentation(artifacts, episodes, embeddings)
    r["reactivation_count"] = segmenter.reactivation_count

    ari = r.get("adjusted_rand_index", float("nan"))
    nmi = r.get("normalized_mutual_info", float("nan"))
    coh = r.get("mean_coherence", float("nan"))
    frag = r.get("fragmentation_ratio", float("nan"))
    n_ep = r.get("episode_count", 0)
    react = r.get("reactivation_count", 0)

    print(f"  ARI={ari:.4f}  NMI={nmi:.4f}  Coh={coh:.4f}  Frag={frag:.2f}  #Ep={n_ep}  React={react}")
    return r


BASE_SEG = dict(
    time_threshold_minutes=120,
    attach_threshold=0.30,
    alpha=0.45,
    beta=0.25,
    gamma=0.10,
    delta=0.20,
    rho=0.05,
    dormancy_minutes=1440,
    allow_reactivation=True,
)

all_results = {}

# V2_ref : V3 algo mais avec ema_alpha=1.0 (≡ moyenne arithmétique) et
#          active_penalty_hours=999 (désactive la pénalité ACTIVE)
all_results["V2_ref  (baseline V2 params, algo V3 désactivé)"] = run_quick(
    "V2_ref — baseline V2 params (EMA=arithmetic, no active penalty)",
    seg_kwargs={**BASE_SEG, "ema_alpha": 1.0, "active_penalty_hours": 9999.0},
    hard_break=0,
)

# V3_ema seul
all_results["V3_ema  (EMA centroid seulement, ema_alpha=0.15)"] = run_quick(
    "V3_ema — EMA centroid (alpha=0.15), pas autres fixes",
    seg_kwargs={**BASE_SEG, "ema_alpha": 0.15, "active_penalty_hours": 9999.0},
    hard_break=0,
)

# V3_age seul
all_results["V3_age  (AgePenalty ACTIVE seulement, 24h)"] = run_quick(
    "V3_age — AgePenalty sur ACTIVE (24h threshold), pas EMA",
    seg_kwargs={**BASE_SEG, "ema_alpha": 1.0, "active_penalty_hours": 24.0},
    hard_break=0,
)

# V3_full
all_results["V3_full (EMA+AgePenalty, no hard_break)"] = run_quick(
    "V3_full — EMA + AgePenalty sur ACTIVE, sans hard_break",
    seg_kwargs={**BASE_SEG, "ema_alpha": 0.15, "active_penalty_hours": 24.0},
    hard_break=0,
)

# V3_hb
all_results["V3_hb   (V3_full + hard_break=720min)"] = run_quick(
    "V3_hb — V3_full + hard_break=720min (sessions 12h)",
    seg_kwargs={**BASE_SEG, "ema_alpha": 0.15, "active_penalty_hours": 24.0},
    hard_break=720,
)

# ─── Tableau final ────────────────────────────────────────────────────────────
print(f"\n\n{'='*90}")
print("  RÉSULTATS V3 — synthetic_200.csv")
print(f"  Note : baseline V2 mesuré précédemment = ARI 0.5060")
print(f"{'='*90}")
print(f"{'Config':48s} | {'ARI':>7} {'NMI':>7} {'Coh':>7} {'Frag':>6} {'#Ep':>4} {'React':>5}")
print("-" * 90)

for label, r in all_results.items():
    ari  = r.get("adjusted_rand_index", float("nan"))
    nmi  = r.get("normalized_mutual_info", float("nan"))
    coh  = r.get("mean_coherence", float("nan"))
    frag = r.get("fragmentation_ratio", float("nan"))
    n_ep = r.get("episode_count", 0)
    react = r.get("reactivation_count", 0)
    delta = ari - 0.5060
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:46s} | {ari:7.4f} {nmi:7.4f} {coh:7.4f} {frag:6.2f} {n_ep:4d} {react:5d}  ({sign}{delta:.4f})")
