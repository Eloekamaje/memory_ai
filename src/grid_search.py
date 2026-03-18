"""
grid_search.py — Recherche par grille des hyperparamètres optimaux (Phase E).

Explore les combinaisons de paramètres du segmenter et du splitter
pour maximiser l'ARI (Adjusted Rand Index) sur un dataset labellisé.

Stratégie :
  1. Pré-charger une seule fois artifacts, embeddings, entities, goals
     (le coût principal — ~80% du temps)
  2. Pour chaque combinaison de params : segment → consolidate → split → évaluer
     (rapide — ~20ms par run)
  3. Trier par ARI décroissant, afficher le top-N

Usage :
    python grid_search.py
"""

import sys
import os
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import load_dataset
from embedding_engine import embed_texts
from episode_algorithm import EpisodeSegmenter
from episode_splitter import EpisodeSplitter, SplitConfig
from goal_heuristics import compute_goal_vectors
from metrics import evaluate_segmentation
from entity_extractor import EntityExtractor
from entity_resolver import EntityResolver, enrich_artifacts_with_entities
from models import Artifact

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ================================================================== #
# Résultat d'une configuration                                        #
# ================================================================== #

@dataclass
class GridResult:
    """Résultat d'un run de grid search."""
    params: Dict[str, float]
    ari: float = 0.0
    nmi: float = 0.0
    coherence: float = 0.0
    n_episodes: int = 0
    n_true: int = 0
    fragmentation: float = 0.0
    split_count: int = 0
    duration_ms: float = 0.0


# ================================================================== #
# Pré-chargement des données                                          #
# ================================================================== #

def preload_data(
    data_file: str,
    ner_backend: str = "spacy",
) -> Dict[str, Any]:
    """
    Charge et prépare les données une seule fois.

    Returns
    -------
    Dict contenant artifacts, embeddings, goal_vectors, resolver, extractor.
    """
    data_path = os.path.join(DATA_DIR, data_file)

    print(f"[preload] Chargement de {data_file}…")
    artifacts = load_dataset(data_path)
    print(f"  → {len(artifacts)} artefacts")

    # Entity Resolution
    print(f"[preload] Entity Extraction ({ner_backend})…")
    extractor = EntityExtractor(backend=ner_backend)
    mentions = extractor.extract_batch(artifacts)

    resolver = EntityResolver(
        string_threshold=0.75,
        use_embeddings=False,
        min_mention_confidence=0.50,
    )
    entities = resolver.resolve(mentions)
    enrich_artifacts_with_entities(artifacts, resolver)
    print(f"  → {len(mentions)} mentions → {len(entities)} entités")

    # Embeddings
    print("[preload] Embeddings…")
    embeddings = embed_texts([a.content for a in artifacts])
    print(f"  → shape {embeddings.shape}")

    # Goals
    print("[preload] Goal vectors…")
    goal_vectors = compute_goal_vectors(artifacts, embed_texts)
    print(f"  → shape {goal_vectors.shape}")

    print("[preload] ✓ Données prêtes\n")

    return {
        "artifacts": artifacts,
        "embeddings": embeddings,
        "goal_vectors": goal_vectors,
        "resolver": resolver,
        "extractor": extractor,
    }


# ================================================================== #
# Run d'une configuration                                             #
# ================================================================== #

def run_config(
    data: Dict[str, Any],
    params: Dict[str, float],
) -> GridResult:
    """
    Exécute segment → consolidate → split → évaluer pour une config donnée.

    Ne refait PAS le chargement, NER, embeddings, goals (déjà dans `data`).
    """
    artifacts = data["artifacts"]
    embeddings = data["embeddings"]

    t0 = time.perf_counter()

    # --- Segmentation ---
    segmenter = EpisodeSegmenter(
        time_threshold_minutes=params.get("time_threshold", 120),
        attach_threshold=params.get("attach_threshold", 0.20),
        alpha=params.get("alpha", 0.55),
        beta=params.get("beta", 0.15),
        gamma=params.get("gamma", 0.10),
        delta=params.get("delta", 0.20),
        rho=params.get("rho", 0.05),
        dormancy_minutes=params.get("dormancy_minutes", 1440),
        allow_reactivation=True,
    )

    episodes = segmenter.segment(artifacts, embeddings)

    # --- Consolidation ---
    episodes = segmenter.consolidate(episodes)

    # --- Splitting ---
    splitter = EpisodeSplitter(SplitConfig(
        min_cohesion=params.get("split_min_cohesion", 0.55),
        min_size_to_split=int(params.get("split_min_size", 8)),
        max_span_hours=params.get("split_max_span", 168.0),
        max_splits=int(params.get("split_max_k", 6)),
        min_sub_size=int(params.get("split_min_sub", 3)),
        silhouette_threshold=params.get("split_sil_threshold", 0.10),
    ))

    episodes = splitter.split(episodes, artifacts, embeddings, verbose=False)

    # --- Évaluation ---
    results = evaluate_segmentation(artifacts, episodes, embeddings)

    dt = (time.perf_counter() - t0) * 1000  # ms

    return GridResult(
        params=params,
        ari=results.get("adjusted_rand_index", 0.0),
        nmi=results.get("normalized_mutual_info", 0.0),
        coherence=results.get("mean_coherence", 0.0),
        n_episodes=results.get("predicted_episode_count", len(episodes)),
        n_true=results.get("true_episode_count", 0),
        fragmentation=results.get("fragmentation_ratio", 0.0),
        split_count=splitter.split_count,
        duration_ms=dt,
    )


# ================================================================== #
# Grille de paramètres                                                #
# ================================================================== #

def build_grid() -> List[Dict[str, float]]:
    """
    Construit la grille d'exploration.

    Stratégie : explorer les axes les plus impactants en priorité :
      1. attach_threshold : contrôle la granularité épisode (très sensible)
      2. alpha/beta/gamma/delta : poids de l'AttachScore
      3. split_min_cohesion : seuil de déclenchement du splitting
      4. rho : pénalité d'âge
    """

    # Axe 1 : attach_threshold
    attach_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]

    # Axe 2 : distributions de poids (α, β, γ, δ) — doivent refléter
    # l'importance relative ; on teste quelques profils
    weight_profiles = [
        # (alpha, beta, gamma, delta) — description
        (0.55, 0.15, 0.10, 0.20),  # baseline
        (0.45, 0.25, 0.10, 0.20),  # +entity
        (0.40, 0.30, 0.10, 0.20),  # ++entity
        (0.50, 0.20, 0.15, 0.15),  # +temporal
        (0.45, 0.20, 0.10, 0.25),  # +goal
        (0.35, 0.35, 0.10, 0.20),  # entity-heavy
        (0.50, 0.25, 0.05, 0.20),  # -temporal
        (0.40, 0.25, 0.15, 0.20),  # balanced
    ]

    # Axe 3 : rho (pénalité aging)
    rhos = [0.03, 0.05, 0.08]

    # Axe 4 : split_min_cohesion
    split_cohesions = [0.50, 0.55, 0.60, 0.65]

    # Combinaisons
    configs = []
    for attach in attach_thresholds:
        for (a, b, g, d) in weight_profiles:
            for rho in rhos:
                for sc in split_cohesions:
                    configs.append({
                        "attach_threshold": attach,
                        "alpha": a,
                        "beta": b,
                        "gamma": g,
                        "delta": d,
                        "rho": rho,
                        "split_min_cohesion": sc,
                        # constantes
                        "time_threshold": 120,
                        "dormancy_minutes": 1440,
                        "split_min_size": 8,
                        "split_max_span": 168.0,
                        "split_max_k": 6,
                        "split_min_sub": 3,
                        "split_sil_threshold": 0.10,
                    })

    return configs


# ================================================================== #
# Grid Search principal                                                #
# ================================================================== #

def grid_search(
    data_file: str,
    ner_backend: str = "spacy",
    top_k: int = 20,
    grid: Optional[List[Dict[str, float]]] = None,
) -> List[GridResult]:
    """
    Lance la recherche par grille.

    Parameters
    ----------
    data_file : fichier CSV dans data/
    ner_backend : "spacy" ou "gliner"
    top_k : nombre de meilleurs résultats à afficher
    grid : grille personnalisée (sinon build_grid())

    Returns
    -------
    Liste de GridResult triés par ARI décroissant.
    """
    # 1. Pré-chargement (une seule fois)
    data = preload_data(data_file, ner_backend=ner_backend)

    # 2. Grille
    if grid is None:
        grid = build_grid()

    n_configs = len(grid)
    print(f"╔═══════════════════════════════════════════╗")
    print(f"║  GRID SEARCH : {n_configs} configurations          ║")
    print(f"║  Dataset: {data_file:30s}   ║")
    print(f"╚═══════════════════════════════════════════╝")

    # 3. Exécution
    results: List[GridResult] = []
    t_start = time.perf_counter()

    for i, params in enumerate(grid):
        result = run_config(data, params)
        results.append(result)

        # progression tous les 10%
        if (i + 1) % max(1, n_configs // 10) == 0 or i == n_configs - 1:
            elapsed = time.perf_counter() - t_start
            pct = 100 * (i + 1) / n_configs
            best_so_far = max(r.ari for r in results)
            print(f"  [{pct:5.1f}%] {i+1}/{n_configs}  "
                  f"elapsed={elapsed:.1f}s  "
                  f"best_ARI={best_so_far:.4f}")

    # 4. Tri par ARI
    results.sort(key=lambda r: r.ari, reverse=True)

    total_time = time.perf_counter() - t_start

    # 5. Affichage
    print(f"\n{'═'*100}")
    print(f"  TOP-{top_k} CONFIGURATIONS (sur {n_configs}) — "
          f"temps total: {total_time:.1f}s")
    print(f"{'═'*100}")

    header = (
        f"{'#':>3}  {'ARI':>7} {'NMI':>7} {'Coh':>7} "
        f"{'#Ep':>4} {'Frag':>5} {'Spl':>3} "
        f"{'attach':>6} {'α':>5} {'β':>5} {'γ':>5} {'δ':>5} "
        f"{'ρ':>5} {'s_coh':>5} {'ms':>6}"
    )
    print(header)
    print("─" * 100)

    for rank, r in enumerate(results[:top_k], 1):
        p = r.params
        print(
            f"{rank:>3}  {r.ari:7.4f} {r.nmi:7.4f} {r.coherence:7.4f} "
            f"{r.n_episodes:4d} {r.fragmentation:5.2f} {r.split_count:3d} "
            f"{p['attach_threshold']:6.2f} "
            f"{p['alpha']:5.2f} {p['beta']:5.2f} "
            f"{p['gamma']:5.2f} {p['delta']:5.2f} "
            f"{p['rho']:5.2f} {p['split_min_cohesion']:5.2f} "
            f"{r.duration_ms:6.1f}"
        )

    # 6. Meilleur résultat détaillé
    best = results[0]
    print(f"\n{'─'*60}")
    print(f"  🏆 MEILLEURE CONFIG")
    print(f"{'─'*60}")
    for k, v in best.params.items():
        print(f"    {k:25s} = {v}")
    print(f"\n    ARI          = {best.ari:.4f}")
    print(f"    NMI          = {best.nmi:.4f}")
    print(f"    Cohérence    = {best.coherence:.4f}")
    print(f"    #Épisodes    = {best.n_episodes} (vrai: {best.n_true})")
    print(f"    Fragmentation= {best.fragmentation:.4f}")
    print(f"    Splits       = {best.split_count}")

    return results


# ================================================================== #
# CLI                                                                  #
# ================================================================== #

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "synthetic_200.csv"
    backend = sys.argv[2] if len(sys.argv) > 2 else "spacy"

    results = grid_search(dataset, ner_backend=backend, top_k=20)
