"""
memory_pipeline.py — Pipeline complet : données brutes → MemoryAgent (Phase B6).

Remplace run_episode_segmentation.py (V1, obsolète).

Assemble dans l'ordre :
  Parse → Embed → Goals → Entities → Segment → Consolidate
  → Graph → Summarize → Index → Store → RecallEngine → MemoryAgent

Usage en Python :
    from memory_pipeline import build_pipeline, load_pipeline

    # Première fois (ou mise à jour) : construire depuis les données brutes
    state = build_pipeline("data/chat.txt", store_dir="memory/")

    # Rechargement rapide depuis le disque
    state = load_pipeline(store_dir="memory/")

    # Utiliser le système
    engine = state["engine"]
    agent  = state["agent"]   # None si llm_fn non fourni

    results = engine.recall("GEDDVIT budget")
    for r in results:
        print(engine.explain(r))
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# ── modules du projet ──────────────────────────────────────────────
from decision_engine import ActionExecutor, DecisionEngine
from embedding_engine import embed_texts
from entity_extractor import EntityExtractor
from entity_resolver import EntityResolver
from episode_algorithm import EpisodeSegmenter
from episode_splitter import EpisodeSplitter, SplitConfig
from episode_summarizer import EpisodeSummarizer, attach_summaries
from goal_heuristics import compute_goal_vectors
from memory_agent import MemoryAgent
from memory_graph import MemoryGraph
from memory_index import MemoryIndex
from memory_store import MemoryStore
from models import Artifact
from recall_engine import RecallEngine


# ================================================================== #
# Helpers parsers                                                     #
# ================================================================== #

def _parse(data_path: str) -> List[Artifact]:
    """Détecte le format et parse en List[Artifact]."""
    path = Path(data_path)
    ext = path.suffix.lower()

    if ext == ".txt":
        from parsers.whatsapp_parser import parse_whatsapp_chat
        return parse_whatsapp_chat(str(path))

    if ext == ".csv":
        from parsers.experiment_csv_parser import parse_experiment_csv
        return parse_experiment_csv(str(path))

    raise ValueError(f"Format non supporté : {ext!r}. Utilise .txt (WhatsApp) ou .csv.")


# ================================================================== #
# Helpers pipeline                                                    #
# ================================================================== #

def _inject_entities(artifacts, canonical_map, art_id_to_idx, n_artifacts):
    """
    Injecte les entités canoniques dans les artefacts.

    Filtre :
      - entités trop fréquentes (> 30% des artefacts) : participants récurrents,
        salutations — non discriminantes pour la segmentation thématique
      - URLs : bruit

    Returns le nombre d'injections effectuées.
    """
    count = 0
    for ce in canonical_map.values():
        freq = len(ce.artifact_ids) / n_artifacts if n_artifacts else 1.0
        if freq > 0.30 or ce.canonical_name.lower().startswith("http"):
            continue
        for art_id in ce.artifact_ids:
            idx = art_id_to_idx.get(str(art_id))
            if idx is not None and ce.canonical_name not in artifacts[idx].entities:
                artifacts[idx].entities.append(ce.canonical_name)
                count += 1
    return count


def _entity_weights(n_injected: int):
    """
    Retourne (alpha, beta) optimaux selon la présence d'entités injectées.

    +28% ARI mesuré sur synthetic_200 avec entités : beta 0.15→0.25.
    Sans entités, le signal sémantique domine.
    """
    if n_injected > 0:
        return 0.45, 0.25   # entity-boosted
    return 0.55, 0.15       # semantic-dominant


class _TypeLookup:
    """Lookup entity_type depuis le dict sauvegardé {canonical_name → type_string}."""

    def __init__(self, entity_types: dict):
        self._map = {k.lower(): v for k, v in entity_types.items()}

    def lookup(self, name: str):
        type_str = self._map.get(name.lower())
        if type_str is None:
            return None
        return type(
            "_CE", (), {"entity_type": type("_ET", (), {"value": type_str})(), "mention_count": 0}
        )()


def _attach_goal_vectors(artifacts, goal_vectors) -> None:
    """Injecte les goal_vectors non-nuls dans les artefacts."""
    if goal_vectors is None:
        return
    for i, art in enumerate(artifacts):
        if i < len(goal_vectors):
            gv = goal_vectors[i]
            art.goal_vector = gv if not np.all(gv == 0) else None


def _run_aging(store, episodes, archive_after_days: int, log) -> None:
    """Archive et compresse les vieux épisodes si demandé."""
    if archive_after_days <= 0:
        return
    n_archived = store.archive_old(episodes, days_threshold=archive_after_days)
    if n_archived:
        store.compress_archived(episodes)
        log(f"      {n_archived} épisodes archivés")


class _CanonicalLookup:
    """
    Adaptateur pour MemoryGraph.build() qui attend resolver.lookup(surface_form).
    Ici on indexe par canonical_name pour renseigner les entity_types.
    """

    def __init__(self, canonical_map):
        self._map = {
            ce.canonical_name.lower(): ce
            for ce in canonical_map.values()
        }

    def lookup(self, name: str):
        return self._map.get(name.lower())


# ================================================================== #
# build_pipeline()                                                    #
# ================================================================== #

def build_pipeline(
    data_path: str,
    store_dir: str = "memory",
    *,
    # Segmentation
    time_threshold_minutes: float = 120,
    attach_threshold: float = 0.30,
    dormancy_minutes: float = 1440,
    hard_break_minutes: float = 720,   # V3 : sessions de 12h par défaut
    consolidate: bool = True,
    # Splitting post-segmentation (None = config par défaut)
    split_config: Optional[SplitConfig] = None,
    # Aging
    archive_after_days: int = 0,       # 0 = désactivé
    # Decision Engine (post-build)
    run_decision_engine: bool = True,
    # LLM (optionnel pour le MemoryAgent)
    llm_fn: Optional[Callable] = None,
    top_k_recall: int = 7,
    # Verbosité
    verbose: bool = True,
) -> Dict:
    """
    Construit le pipeline complet depuis les données brutes.

    Parameters
    ----------
    data_path        : chemin vers le fichier de données (.txt WhatsApp ou .csv)
    store_dir        : répertoire de persistance
    archive_after_days : archiver les épisodes CONSOLIDATED > N jours (0 = off)
    llm_fn           : callable(system, user) -> str pour le MemoryAgent
    verbose          : affiche la progression

    Returns
    -------
    Dict avec clés : episodes, artifacts, embeddings, summaries,
                     engine, agent, graph, index, store
    """
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    t0 = time.time()

    # ── 1. Parse ───────────────────────────────────────────────────
    log(f"[1/10] Parsing {data_path} …")
    artifacts = _parse(data_path)
    log(f"      {len(artifacts)} artefacts chargés")

    # ── 2. Embeddings ──────────────────────────────────────────────
    log("[2/10] Embeddings (all-MiniLM-L6-v2) …")
    texts = [a.content for a in artifacts]
    embeddings = embed_texts(texts)
    embeddings = embeddings.astype(np.float32)
    log(f"      shape={embeddings.shape}")

    # ── 3. Goal vectors ────────────────────────────────────────────
    log("[3/10] Goal heuristics …")
    goal_vectors = compute_goal_vectors(artifacts, embed_fn=embed_texts)
    _attach_goal_vectors(artifacts, goal_vectors)

    # ── 4. Entity extraction ───────────────────────────────────────
    log("[4/10] Entity extraction …")
    extractor = EntityExtractor()
    mentions = extractor.extract_batch(artifacts)
    log(f"      {len(mentions)} mentions extraites")

    # ── 5. Entity resolution ───────────────────────────────────────
    log("[5/10] Entity resolution …")
    resolver = EntityResolver(string_threshold=0.75)
    canonical_map = resolver.resolve(mentions)
    # Injecter les entités canoniques dans les artefacts via artifact_ids.
    # On exclut les entités non-discriminantes :
    #   - trop fréquentes (> 50% des artefacts) : auteurs de conversation, salutations
    #   - URLs : bruit pour la segmentation thématique
    n_artifacts = len(artifacts)
    art_id_to_idx = {art.id: i for i, art in enumerate(artifacts)}
    n_injected = _inject_entities(artifacts, canonical_map, art_id_to_idx, n_artifacts)
    log(f"      {len(canonical_map)} entités canoniques, {n_injected} injections")

    # ── 6. Segmentation ────────────────────────────────────────────
    log("[6/10] Segmentation épisodique …")
    seg_alpha, seg_beta = _entity_weights(n_injected)
    segmenter = EpisodeSegmenter(
        time_threshold_minutes=time_threshold_minutes,
        attach_threshold=attach_threshold,
        dormancy_minutes=dormancy_minutes,
        hard_break_minutes=hard_break_minutes,
        alpha=seg_alpha,
        beta=seg_beta,
    )
    episodes = segmenter.segment(artifacts, embeddings)
    log(f"      {len(episodes)} épisodes bruts")

    if consolidate:
        episodes = segmenter.consolidate(episodes)
        log(f"      → {len(episodes)} après consolidation")

    # ── 7. Splitting post-segmentation ─────────────────────────────
    log("[7/10] Splitting méga-épisodes …")
    cfg = split_config if split_config is not None else SplitConfig()
    splitter = EpisodeSplitter(cfg)
    episodes = splitter.split(episodes, artifacts, embeddings)
    n_after = splitter.episodes_after if splitter.split_count > 0 else len(episodes)
    log(f"      {splitter.episodes_before} → {n_after} épisodes ({splitter.split_count} splits)")

    # ── 8. Graph ───────────────────────────────────────────────────
    log("[8/10] Memory graph …")
    graph = MemoryGraph()
    graph.build(episodes, artifacts, embeddings, _CanonicalLookup(canonical_map))
    log(f"      {graph.n_episodes} nœuds épisodes, {graph.n_entities} entités")

    # ── 9. Summaries ───────────────────────────────────────────────
    log("[9/10] Summarization …")
    summarizer = EpisodeSummarizer(entity_first=(n_injected > 0))
    summaries_list = summarizer.summarize_all(episodes, artifacts, embeddings)
    summary_map = attach_summaries(episodes, summaries_list)
    log(f"      {len(summaries_list)} résumés générés")

    # ── 10. Index + Store ──────────────────────────────────────────
    log("[10/10] Index + persistance …")
    index = MemoryIndex(n_neighbors=min(20, len(episodes)))
    index.build(episodes)

    store = MemoryStore(store_dir)

    # Aging optionnel avant sauvegarde
    _run_aging(store, episodes, archive_after_days, log)

    entity_types = {
        ce.canonical_name: ce.entity_type.value
        for ce in canonical_map.values()
        if hasattr(ce.entity_type, "value")
    }
    store.save(episodes, artifacts, embeddings, summaries_list, entity_types)
    index.save(os.path.join(store_dir, "memory_index"))
    log(f"      Sauvegardé dans {store_dir}/")

    # ── Decision Engine (post-build) ───────────────────────────────
    de_report = None
    if run_decision_engine:
        log("[DE] Decision Engine …")
        de = DecisionEngine()
        actions = de.evaluate(episodes)
        executor = ActionExecutor(summarizer=summarizer)
        de_report = executor.apply(actions, artifacts, embeddings)
        log(f"      {de_report.summary().splitlines()[0]}")

    # ── RecallEngine ───────────────────────────────────────────────
    engine = RecallEngine(
        episodes=episodes,
        artifacts=artifacts,
        embeddings=embeddings,
        embed_fn=embed_texts,
        memory_graph=graph,
        extractor=extractor,
        resolver=resolver,
        index=index,
    )
    engine.summary_map = summary_map

    # ── MemoryAgent (optionnel) ─────────────────────────────────────
    agent = None
    if llm_fn is not None:
        agent = MemoryAgent(recall_engine=engine, llm_fn=llm_fn, top_k_recall=top_k_recall)

    elapsed = time.time() - t0
    log(f"\n  Pipeline terminé en {elapsed:.1f}s")

    return {
        "episodes":      episodes,
        "artifacts":     artifacts,
        "embeddings":    embeddings,
        "summaries":     summary_map,
        "engine":        engine,
        "agent":         agent,
        "graph":         graph,
        "index":         index,
        "store":         store,
        "de_report":     de_report,
        # composants internes exposés pour ingest()
        "segmenter":     segmenter,
        "summarizer":    summarizer,
        "extractor":     extractor,
        "resolver":      resolver,
        "canonical_map": canonical_map,
    }


# ================================================================== #
# load_pipeline()                                                     #
# ================================================================== #

def load_pipeline(
    store_dir: str = "memory",
    *,
    llm_fn: Optional[Callable] = None,
    top_k_recall: int = 7,
    verbose: bool = True,
) -> Dict:
    """
    Recharge le pipeline depuis le disque (rapide, pas de re-embedding).

    Returns
    -------
    Même structure que build_pipeline().
    """
    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    t0 = time.time()
    log(f"Chargement depuis {store_dir}/ …")

    store = MemoryStore(store_dir)
    state = store.load()

    episodes    = state["episodes"]
    artifacts   = state["artifacts"]
    embeddings  = state["embeddings"]
    summary_map = state["summaries"]
    entity_types = state.get("entity_types", {})

    log(f"  {len(episodes)} épisodes, {len(artifacts)} artefacts")

    # Reconstruire l'index (rapide, pas d'embedding)
    index_prefix = os.path.join(store_dir, "memory_index")
    index = MemoryIndex(n_neighbors=min(20, len(episodes)))
    if Path(index_prefix + "_centroids.npy").exists():
        index.load(index_prefix)
        log(f"  Index chargé : {index}")
    else:
        index.build(episodes)
        log(f"  Index reconstruit : {index}")

    # Reconstruire le graph avec les types d'entités sauvegardés
    graph = MemoryGraph()
    graph.build(episodes, artifacts, embeddings, _TypeLookup(entity_types))

    engine = RecallEngine(
        episodes=episodes,
        artifacts=artifacts,
        embeddings=embeddings,
        embed_fn=embed_texts,
        memory_graph=graph,
        index=index,
    )
    engine.summary_map = summary_map

    agent = None
    if llm_fn is not None:
        agent = MemoryAgent(recall_engine=engine, llm_fn=llm_fn, top_k_recall=top_k_recall)

    log(f"  Chargé en {time.time() - t0:.1f}s")

    return {
        "episodes":  episodes,
        "artifacts": artifacts,
        "embeddings": embeddings,
        "summaries": summary_map,
        "engine":    engine,
        "agent":     agent,
        "graph":     graph,
        "index":     index,
        "store":     store,
    }


# ================================================================== #
# ingest() — pipeline incrémental                                     #
# ================================================================== #

def ingest(
    artifact: Artifact,
    state:    Dict,
    *,
    attach_threshold: float = 0.30,
    now:              Optional[datetime] = None,
    verbose:          bool = False,
) -> Dict:
    """
    Ingère un seul artefact dans le pipeline existant — sans rebuild complet.

    Étapes :
        1. Embed le nouvel artefact
        2. Extrait ses entités (si extractor disponible dans state)
        3. Attache à l'épisode le plus proche ou crée un nouvel épisode
        4. Met à jour MemoryGraph et MemoryIndex en place
        5. Lance le Decision Engine (SURFACE proactif sur l'artefact entrant)

    Limitations v0 :
        - Pas de boundary detector (TCN/Transformer) — attach cosine direct
        - MemoryStore non mis à jour automatiquement (appeler state['store'].save())
        - Embeddings matrix non étendue (RecallEngine sur anciens seuls)

    Retourne
    --------
    Dict avec clés : episode, is_new, sim, de_report
    """
    def log(msg):
        if verbose:
            print(f"  [ingest] {msg}", flush=True)

    episodes = state["episodes"]

    # ── 1. Embedding ─────────────────────────────────────────────────
    emb = embed_texts([artifact.content])[0].astype(np.float32)
    log(f"embed OK")

    # ── 2. Entités ───────────────────────────────────────────────────
    _ingest_entities(artifact, state.get("extractor"))
    log(f"entités : {artifact.entities[:5]}")

    # ── 3. Attach ou créer épisode ───────────────────────────────────
    target_ep, is_new, sim = _attach_or_create(
        artifact, emb, episodes, attach_threshold)
    log(f"{'Nouvel épisode' if is_new else f'Attaché (sim={sim:.3f})'} → {target_ep.id}")

    # ── 4. Mise à jour Graph + Index ─────────────────────────────────
    state["graph"].add_episode(target_ep)
    state["index"].add_episode(target_ep)
    log("graph + index mis à jour")

    # ── 5. Decision Engine ───────────────────────────────────────────
    artifact.goal_vector = emb
    de        = DecisionEngine()
    de_actions = de.evaluate(episodes, new_artifact=artifact, now=now)
    executor  = ActionExecutor(summarizer=state.get("summarizer"))
    de_report = executor.apply(de_actions, state["artifacts"], state["embeddings"])
    log(f"Decision Engine : {len(de_actions)} actions")

    return {"episode": target_ep, "is_new": is_new, "sim": sim, "de_report": de_report}


# ── helpers ingest ──────────────────────────────────────────────────

def _ingest_entities(artifact: Artifact, extractor) -> None:
    """Extrait et injecte les entités dans l'artefact (best-effort)."""
    if extractor is None:
        return
    try:
        for m in extractor.extract(artifact):
            if m.canonical_name not in artifact.entities:
                artifact.entities.append(m.canonical_name)
    except Exception:
        pass


def _attach_or_create(
    artifact:         Artifact,
    emb:              np.ndarray,
    episodes:         list,
    attach_threshold: float,
):
    """
    Retourne (episode, is_new, best_sim).
    Si best_sim < attach_threshold → crée un nouvel épisode.
    Sinon → met à jour l'épisode existant le plus proche (EMA centroïde).
    """
    best_ep, best_sim = _best_active_episode(emb, episodes)

    if best_ep is None or best_sim < attach_threshold:
        return _create_episode(artifact, emb, episodes), True, best_sim or 0.0

    _update_episode_ema(best_ep, artifact, emb)
    return best_ep, False, best_sim


def _best_active_episode(emb: np.ndarray, episodes: list):
    """Retourne (episode, sim) de l'épisode actif/dormant le plus proche."""
    best_ep, best_sim = None, 0.0
    norm_emb = np.linalg.norm(emb) + 1e-9
    for ep in episodes:
        if ep.state.value not in ("ACTIVE", "DORMANT") or ep.centroid is None:
            continue
        sim = float(np.dot(emb, ep.centroid) / (norm_emb * np.linalg.norm(ep.centroid) + 1e-9))
        if sim > best_sim:
            best_sim, best_ep = sim, ep
    return best_ep, best_sim


def _create_episode(artifact: Artifact, emb: np.ndarray, episodes: list):
    """Crée et enregistre un nouvel épisode à partir d'un artefact."""
    from models import Episode
    ep = Episode(
        id            = f"ep_{len(episodes) + 1:04d}",
        centroid      = emb.copy(),
        centroid_init = emb.copy(),
        time_interval = (artifact.timestamp, artifact.timestamp),
        memory_score  = 1.0,
        entity_weights= {ent: 1.0 for ent in artifact.entities},
    )
    episodes.append(ep)
    return ep


def _update_episode_ema(ep, artifact: Artifact, emb: np.ndarray,
                         alpha: float = 0.15) -> None:
    """Met à jour centroïde (EMA), intervalle temporel et entités."""
    ep.centroid = (1 - alpha) * ep.centroid + alpha * emb
    ep.centroid /= np.linalg.norm(ep.centroid) + 1e-9
    t0 = ep.time_interval[0] if ep.time_interval else artifact.timestamp
    ep.time_interval = (t0, artifact.timestamp)
    for ent in artifact.entities:
        ep.entity_weights[ent] = ep.entity_weights.get(ent, 0.0) + 0.5
