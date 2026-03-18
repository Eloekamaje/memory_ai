"""
experiment_runner.py — Pipeline complet V2 : chargement → goals → embedding → segmentation → évaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import load_dataset
from embedding_engine import embed_texts
from episode_algorithm import EpisodeSegmenter
from goal_heuristics import compute_goal_vectors
from metrics import evaluate_segmentation
from entity_extractor import EntityExtractor
from entity_resolver import EntityResolver, enrich_artifacts_with_entities
from memory_graph import MemoryGraph
from recall_engine import RecallEngine
from episode_splitter import EpisodeSplitter, SplitConfig
from episode_summarizer import EpisodeSummarizer, attach_summaries
from conversation_preprocessor import (
    preprocess_conversation, PreprocessConfig, BurstConfig
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def run_experiment(data_file: str,
                   time_threshold: float = 120,
                   attach_threshold: float = 0.22,
                   alpha: float = 0.40,
                   beta: float = 0.30,
                   gamma: float = 0.10,
                   delta: float = 0.20,
                   rho: float = 0.10,
                   dormancy_minutes: float = 1440,
                   use_goals: bool = True,
                   allow_reactivation: bool = True,
                   use_entity_resolution: bool = True,
                   ner_backend: str = "spacy",
                   build_graph: bool = True,
                   hard_break_minutes: float = 0,
                   chat_preprocess: bool = False):
    """
    Exécute une expérience complète V2 sur un fichier de données.
    """

    data_path = os.path.join(DATA_DIR, data_file)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT V2 : {data_file}")
    print(f"  params: time={time_threshold}min  attach={attach_threshold}")
    print(f"          α={alpha:.2f}  β={beta:.2f}  γ={gamma:.2f}  δ={delta:.2f}")
    print(f"          ρ={rho:.2f}  dormancy={dormancy_minutes}min")
    print(f"          goals={'ON' if use_goals else 'OFF'}  NER={'ON' if use_entity_resolution else 'OFF'} ({ner_backend})")
    if hard_break_minutes > 0:
        print(f"          hard_break={hard_break_minutes}min  chat_preprocess={'ON' if chat_preprocess else 'OFF'}")
    print(f"{'='*60}")

    # 1. chargement
    artifacts = load_dataset(data_path)
    print(f"\n[1] Artifacts loaded: {len(artifacts)}")

    # 1a. prétraitement conversationnel (optionnel)
    if chat_preprocess:
        pp_config = PreprocessConfig(
            remove_noise=True,
            min_substantive_chars=4,
            remove_participant_names=False,  # fait APRÈS NER
            merge_bursts_enabled=True,
            burst_config=BurstConfig(max_gap_seconds=30, max_burst_size=5),
        )
        artifacts, pp_stats = preprocess_conversation(artifacts, pp_config)
        print(f"[1a] Prétraitement chat: {pp_stats['original_count']} → "
              f"{pp_stats['final_count']} arts "
              f"(noise={pp_stats['noise_count']}, "
              f"burst_merges={pp_stats['burst_merges']})")

    # 1b. Entity Resolution (B1)
    if use_entity_resolution:
        extractor = EntityExtractor(backend=ner_backend)
        mentions = extractor.extract_batch(artifacts)
        resolver = EntityResolver(
            string_threshold=0.75,
            use_embeddings=False,
            min_mention_confidence=0.50,
        )
        entities = resolver.resolve(mentions)
        enrich_artifacts_with_entities(artifacts, resolver)
        n_mentions = len(mentions)
        n_entities = len(entities)
        print(f"[1b] Entity Resolution: {n_mentions} mentions → {n_entities} entités canoniques")

        # retirer les noms de participants des entités (chat uniquement)
        if chat_preprocess:
            from conversation_preprocessor import remove_participant_entities
            removed = remove_participant_entities(artifacts)
            if removed:
                print(f"      Participants retirés des entités: {removed}")
    else:
        print(f"[1b] Entity Resolution disabled")

    # 2. embeddings
    embeddings = embed_texts([a.content for a in artifacts])
    print(f"[2] Embeddings shape: {embeddings.shape}")

    # 3. goal vectors (V2)
    if use_goals:
        goal_vectors = compute_goal_vectors(artifacts, embed_texts)
        print(f"[3] Goal vectors computed: {goal_vectors.shape}")
    else:
        # pas de goal_vector → δ sera ignoré car goal_similarity retourne 0
        delta = 0.0
        print(f"[3] Goals disabled (δ forced to 0)")

    # 4. segmentation
    segmenter = EpisodeSegmenter(
        time_threshold_minutes=time_threshold,
        attach_threshold=attach_threshold,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        rho=rho,
        dormancy_minutes=dormancy_minutes,
        allow_reactivation=allow_reactivation,
        hard_break_minutes=hard_break_minutes,
    )

    episodes = segmenter.segment(artifacts, embeddings)
    print(f"[4] Episodes detected: {len(episodes)}")
    print(f"    Reactivations: {segmenter.reactivation_count}")

    # 4b. consolidation post-segmentation (V2+)
    episodes_before_consol = len(episodes)
    episodes = segmenter.consolidate(episodes)
    consol_merges = getattr(segmenter, 'consolidation_merges', 0)
    if consol_merges > 0:
        print(f"    Consolidation: {episodes_before_consol} → {len(episodes)} ({consol_merges} merges)")

    # compter états
    from models import EpisodeState
    state_counts = {}
    for ep in episodes:
        state_counts[ep.state.value] = state_counts.get(ep.state.value, 0) + 1
    print(f"    States: {state_counts}")

    # 4c. Cohesion-based splitting (Phase D)
    splitter = EpisodeSplitter(SplitConfig(
        min_cohesion=0.65,
        min_size_to_split=8,
        max_span_hours=168.0,
        max_splits=6,
        min_sub_size=3,
        silhouette_threshold=0.10,
    ))
    episodes = splitter.split(episodes, artifacts, embeddings)
    if splitter.split_count > 0:
        print(f"[4c] Splitting: {splitter.episodes_before} → "
              f"{splitter.episodes_after} épisodes "
              f"({splitter.split_count} splits)")
    else:
        print(f"[4c] Splitting: aucun candidat (tous cohérents)")

    # 4d. Memory Graph (B2)
    graph = None
    if build_graph and use_entity_resolution:
        graph = MemoryGraph()
        graph.build(
            episodes=episodes,
            artifacts=artifacts,
            embeddings=embeddings,
            resolver=resolver if use_entity_resolution else None,
        )
        bridges = graph.bridge_entities(min_episodes=2)
        cc = graph.connected_components()
        multi = graph.multi_episode_candidates(episodes, threshold_ratio=0.7)
        print(f"[4d] Memory Graph: {graph.n_episodes} épisodes, "
              f"{graph.n_entities} entités, {graph.n_edges} arêtes")
        print(f"     Composantes connexes: {len(cc)}  "
              f"Entités-pont: {len(bridges)}  "
              f"Artefacts multi-épisode: {len(multi)}")
    else:
        print("[4d] Memory Graph disabled")

    # 5. Recall Engine (Phase C)
    recall = RecallEngine(
        episodes=episodes,
        artifacts=artifacts,
        embeddings=embeddings,
        embed_fn=embed_texts,
        memory_graph=graph,
        extractor=extractor if use_entity_resolution else None,
        resolver=resolver if use_entity_resolution else None,
    )
    print(f"[5] Recall Engine ready ({len(episodes)} épisodes indexés)")

    # 5b. Episode Summarizer (Phase G)
    summarizer = EpisodeSummarizer()
    summaries = summarizer.summarize_all(episodes, artifacts, embeddings)
    summary_map = attach_summaries(episodes, summaries)
    n_labeled = sum(1 for s in summaries if s.label != "Conversation générale")
    print(f"[5b] Summaries: {len(summaries)} épisodes résumés ({n_labeled} labellisés)")

    # Connecter les summaries au RecallEngine
    recall.summary_map = summary_map

    # 6. évaluation
    results = evaluate_segmentation(artifacts, episodes, embeddings)
    results["reactivation_count"] = segmenter.reactivation_count
    if graph is not None:
        results["graph_entities"] = graph.n_entities
        results["graph_edges"] = graph.n_edges
        results["bridge_entities"] = len(graph.bridge_entities(min_episodes=2))
        results["connected_components"] = len(graph.connected_components())
        results["_graph"] = graph
    results["_recall"] = recall       # objet RecallEngine pour usage notebook
    results["_summaries"] = summary_map  # mapping ep_id → EpisodeSummary

    print(f"\n{'─'*40}")
    print("  RESULTS")
    print(f"{'─'*40}")

    for key, value in results.items():
        if key.startswith('_'):
            continue
        if isinstance(value, float):
            print(f"  {key:30s} : {value:.4f}")
        else:
            print(f"  {key:30s} : {value}")

    # 7. aperçu des épisodes avec labels
    print(f"\n{'─'*40}")
    print("  TOP 10 EPISODES")
    print(f"{'─'*40}")

    sorted_episodes = sorted(episodes, key=lambda e: len(e.artifact_indices), reverse=True)

    for ep in sorted_episodes[:10]:
        n = len(ep.artifact_indices)
        t_start, t_end = ep.time_interval
        state_flag = "🟢" if ep.state.value == "ACTIVE" else ("💤" if ep.state.value == "DORMANT" else "📦")
        s = summary_map.get(ep.id)
        label = s.label if s else "—"
        mood = s.mood if s else ""
        mood_icon = {"positive": "😊", "negative": "😟", "mixed": "🔀"}.get(mood, "")
        print(f"  {state_flag} {ep.id} | {n:>3} arts | "
              f"{t_start.strftime('%m-%d %H:%M')}→{t_end.strftime('%m-%d %H:%M')} | "
              f"{mood_icon} {label}")

    return results


if __name__ == "__main__":

    # ============================================================
    #  ABLATION STUDY — H1 (entity+goal), H2 (aging), H3 (reactivation)
    # ============================================================

    configs = {
        "A: Semantic-only (baseline)": dict(
            alpha=1.0, beta=0.0, gamma=0.0, delta=0.0,
            rho=0.0, attach_threshold=0.28, dormancy_minutes=99999,
            use_goals=False),

        "B: V1 Sem+Ent+Temp": dict(
            alpha=0.70, beta=0.20, gamma=0.10, delta=0.0,
            rho=0.0, attach_threshold=0.20, dormancy_minutes=99999,
            use_goals=False),

        "C: +Goal (H1)": dict(
            alpha=0.50, beta=0.25, gamma=0.10, delta=0.15,
            rho=0.0, attach_threshold=0.28, dormancy_minutes=99999,
            use_goals=True),

        "D: +Aging (H2)": dict(
            alpha=0.50, beta=0.25, gamma=0.10, delta=0.15,
            rho=0.08, attach_threshold=0.28, dormancy_minutes=1080,
            use_goals=True, allow_reactivation=False),

        "E: +Reactivation (H3) [full V2]": dict(
            alpha=0.50, beta=0.25, gamma=0.10, delta=0.15,
            rho=0.08, attach_threshold=0.28, dormancy_minutes=1080,
            use_goals=True, allow_reactivation=True),
    }

    all_results = {}
    for label, params in configs.items():
        print(f"\n{'▓' * 60}")
        print(f"  {label}")
        print(f"{'▓' * 60}")
        all_results[label] = run_experiment("synthetic_200.csv", **params)

    # --- Tableau comparatif ---
    print(f"\n\n{'=' * 90}")
    print("  ABLATION STUDY — SYNTHETIC_200")
    print(f"{'=' * 90}")
    print(f"{'Config':40s} | {'ARI':>7} {'NMI':>7} {'Coh':>7} {'Frag':>6} {'#Ep':>4} {'React':>5}")
    print("-" * 90)

    for label, r in all_results.items():
        ari = r.get('adjusted_rand_index', 0)
        nmi = r.get('normalized_mutual_info', 0)
        coh = r.get('mean_coherence', 0)
        frag = r.get('fragmentation_ratio', 0)
        n_ep = r.get('episode_count', 0)
        react = r.get('reactivation_count', 0)
        print(f"  {label:38s} | {ari:7.4f} {nmi:7.4f} {coh:7.4f} {frag:6.2f} {n_ep:4d} {react:5d}")