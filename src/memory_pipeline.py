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
from episode_segmenter_hybrid import HybridEpisodeSegmenter
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
# Configuration                                                       #
# ================================================================== #

from dataclasses import dataclass, field as _field


@dataclass
class SegmentConfig:
    """
    Paramètres de segmentation épisodique pour build_pipeline().

    Segmentation greedy (Stage 2 AttachScore) :
      time_threshold_minutes  : gap temporel au-delà duquel un épisode est candidat à la clôture
      attach_threshold        : seuil minimum de AttachScore pour rattacher un message
      dormancy_minutes        : délai sans activité avant de passer ACTIVE → DORMANT
      hard_break_minutes      : gap absolu forçant un nouvel épisode (ignore AttachScore)
      consolidate             : fusionner les épisodes unitaires adjacents

    Stage 1 (Transformer boundary detector — optionnel) :
      boundary_detector_path  : chemin vers un fichier .pt entraîné.
                                Si None ou introuvable → AttachScore pur (Stage 2 seulement).
      boundary_k              : facteur de calibration probabiliste [0, 1].
                                0 = seuil fixe, 0.3 = défaut recommandé.
    """
    time_threshold_minutes: float = 120.0
    attach_threshold: float = 0.30
    dormancy_minutes: float = 1440.0
    hard_break_minutes: float = 720.0
    consolidate: bool = True
    boundary_detector_path: Optional[str] = None
    boundary_k: float = 0.3


# ================================================================== #
# Helpers parsers                                                     #
# ================================================================== #

def _parse_one(data_path: str) -> List[Artifact]:
    """Parse un seul fichier selon son extension."""
    path = Path(data_path)
    ext = path.suffix.lower()

    if ext == ".txt":
        from parsers.whatsapp_parser import parse_whatsapp_chat
        return parse_whatsapp_chat(str(path))

    if ext == ".csv":
        from parsers.experiment_csv_parser import parse_experiment_csv
        return parse_experiment_csv(str(path))

    raise ValueError(f"Format non supporté : {ext!r}. Utilise .txt (WhatsApp) ou .csv.")


def _parse(data_paths) -> List[Artifact]:
    """
    Parse un ou plusieurs fichiers, merge et trie par timestamp.

    Accepte :
      - str              : un seul fichier (comportement original)
      - List[str]        : plusieurs fichiers multi-sources

    Les artefacts sont triés chronologiquement après merge.
    Le champ artifact.source est conservé pour distinguer les canaux.
    """
    if isinstance(data_paths, str):
        return _parse_one(data_paths)

    all_artifacts: List[Artifact] = []
    for path in data_paths:
        all_artifacts.extend(_parse_one(path))

    # Tri chronologique global — clé pour que le Transformer et gap_log aient du sens
    all_artifacts.sort(key=lambda a: a.timestamp)
    return all_artifacts


# ================================================================== #
# Cache (embeddings + mentions GLiNER)                               #
# ================================================================== #

import gc
import hashlib
import json


def _file_hash(path: str) -> str:
    """SHA-256 des 4 premiers Mo d'un fichier."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:16]


def _files_hash(paths: List[str]) -> str:
    """Hash combiné de plusieurs fichiers — change si l'un d'eux change."""
    h = hashlib.sha256()
    for path in sorted(paths):          # sorted → ordre stable indépendant de l'appel
        h.update(path.encode())
        with open(path, "rb") as f:
            h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:16]


def _cache_dir(store_dir: str) -> Path:
    p = Path(store_dir) / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Arrays numpy (embeddings, goal_vectors) ────────────────────────

def _load_cached(store_dir: str, file_hash: str, name: str, log):
    cp = _cache_dir(store_dir) / f"{name}_{file_hash}.npy"
    if cp.exists():
        arr = np.load(str(cp))
        log(f"      ↩ {name} chargé depuis cache ({cp.name})")
        return arr
    return None


def _save_cache(store_dir: str, file_hash: str, name: str, arr: np.ndarray) -> None:
    cp = _cache_dir(store_dir) / f"{name}_{file_hash}.npy"
    np.save(str(cp), arr)


# ── Mentions GLiNER (JSON) ──────────────────────────────────────────

def _mentions_cache_path(store_dir: str, file_hash: str) -> Path:
    return _cache_dir(store_dir) / f"mentions_{file_hash}.json"


def _load_mentions_cache(store_dir: str, file_hash: str, log):
    """Recharge les mentions depuis le cache JSON. Retourne None si absent."""
    from entity_extractor import EntityType, Mention
    cp = _mentions_cache_path(store_dir, file_hash)
    if not cp.exists():
        return None
    with open(cp, encoding="utf-8") as f:
        raw = json.load(f)
    mentions = [
        Mention(
            surface_form=m["surface_form"],
            entity_type=EntityType[m["entity_type"]],
            source_artifact_id=m["source_artifact_id"],
            confidence=m["confidence"],
            extraction_method=m["extraction_method"],
        )
        for m in raw
    ]
    log(f"      ↩ mentions chargées depuis cache ({cp.name}, {len(mentions)} entrées)")
    return mentions


def _save_mentions_cache(store_dir: str, file_hash: str, mentions) -> None:
    cp = _mentions_cache_path(store_dir, file_hash)
    raw = [
        {
            "surface_form": m.surface_form,
            "entity_type": m.entity_type.name,
            "source_artifact_id": m.source_artifact_id,
            "confidence": m.confidence,
            "extraction_method": m.extraction_method,
        }
        for m in mentions
    ]
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)


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
        n_occ = len(ce.artifact_ids)
        freq = n_occ / n_artifacts if n_artifacts else 1.0
        name = ce.canonical_name.strip()
        if not _entity_is_useful(ce.entity_type, name, freq, n_occ):
            continue
        for art_id in ce.artifact_ids:
            idx = art_id_to_idx.get(str(art_id))
            if idx is not None and name not in artifacts[idx].entities:
                artifacts[idx].entities.append(name)
                count += 1
    return count


def _entity_is_useful(entity_type, name: str, freq: float, count: int = 1) -> bool:
    """Filtre de qualité pour les entités canoniques injectées dans les artefacts."""
    from entity_extractor import EntityType
    _KEEP_TYPES = {
        EntityType.PERSON, EntityType.ORG, EntityType.PLACE,
        EntityType.PROJECT, EntityType.PRODUCT, EntityType.EVENT,
        EntityType.DATE_EVENT, EntityType.DOCUMENT,
    }
    if freq > 0.30:                         # trop fréquent = non discriminant
        return False
    if entity_type not in _KEEP_TYPES:      # URL / HANDLE / UNKNOWN / CONCEPT
        return False
    if len(name) < 3:                       # trop court
        return False
    if len(name.split()) > 3:              # expression > 3 mots = bruit GLiNER
        return False
    if count < 2:                           # singleton = bruit probable
        return False
    return True


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


def _build_segmenter(
    seg: "SegmentConfig",
    alpha: float,
    beta: float,
    embeddings: "np.ndarray",
    log,
):
    """
    Instancie le segmenteur selon seg.boundary_detector_path.

    - Si un .pt valide est fourni → HybridEpisodeSegmenter (Stage 1 + 2 + 3)
    - Sinon                       → EpisodeSegmenter (Stage 2 + 3, AttachScore pur)
    """
    common = {
        "time_threshold_minutes": seg.time_threshold_minutes,
        "attach_threshold": seg.attach_threshold,
        "dormancy_minutes": seg.dormancy_minutes,
        "hard_break_minutes": seg.hard_break_minutes,
        "alpha": alpha,
        "beta": beta,
    }

    bd_path = seg.boundary_detector_path
    if bd_path and Path(bd_path).exists():
        from boundary_detector_transformer import TransformerBoundaryDetector
        emb_dim = embeddings.shape[1]
        detector = TransformerBoundaryDetector(d_input=emb_dim + 1).load(bd_path)
        log(f"      Stage 1 Transformer chargé ({bd_path}) — d_input={emb_dim + 1}")
        return HybridEpisodeSegmenter(
            detector=detector,
            boundary_k=seg.boundary_k,
            **common,
        )

    if bd_path:
        log(f"      ⚠ boundary_detector_path={bd_path!r} introuvable → AttachScore pur")
    return EpisodeSegmenter(**common)


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
    data_path,                          # str OU List[str]
    store_dir: str = "memory",
    *,
    segment_config: Optional[SegmentConfig] = None,
    split_config: Optional[SplitConfig] = None,
    archive_after_days: int = 0,
    run_decision_engine: bool = True,
    llm_fn: Optional[Callable] = None,
    top_k_recall: int = 7,
    ner_backend: str = "gliner",
    summarizer_backend: str = "extractive",
    gguf_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Construit le pipeline complet depuis les données brutes.

    Parameters
    ----------
    data_path        : str ou List[str] — un ou plusieurs fichiers de données.
                       Formats supportés : .txt (WhatsApp), .csv.
                       Si plusieurs fichiers, les artefacts sont mergés et triés par timestamp.
    store_dir        : répertoire de persistance
    segment_config   : paramètres de segmentation (voir SegmentConfig). None = défauts.
    split_config     : paramètres de splitting post-segmentation. None = défauts.
    archive_after_days : archiver les épisodes CONSOLIDATED > N jours (0 = off)
    run_decision_engine : appliquer le Decision Engine après segmentation
    llm_fn           : callable(system, user) -> str pour le MemoryAgent
    top_k_recall     : nombre max de résultats pour RecallEngine
    ner_backend      : "gliner" (défaut) | "spacy" | "llm" (Qwen2.5 GGUF)
    summarizer_backend : "extractive" (défaut) | "llm" (Qwen2.5 GGUF)
    gguf_path        : chemin GGUF pour les backends LLM (NER + summarizer).
                       Si None, auto-détection du modèle Qwen2.5.
    verbose          : affiche la progression

    Returns
    -------
    Dict avec clés : episodes, artifacts, embeddings, summaries,
                     engine, agent, graph, index, store, de_report,
                     segmenter, summarizer, extractor, resolver, canonical_map
    """
    seg = segment_config if segment_config is not None else SegmentConfig()

    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    t0 = time.time()

    # ── 1. Parse ───────────────────────────────────────────────────
    paths = [data_path] if isinstance(data_path, str) else list(data_path)
    sources_str = ", ".join(Path(p).name for p in paths)
    log(f"[1/10] Parsing {sources_str} …")
    artifacts = _parse(data_path)
    sources = sorted({a.source for a in artifacts if a.source})
    log(f"      {len(artifacts)} artefacts chargés ({len(paths)} source(s) : {', '.join(sources) or '—'})")

    # Hash combiné de tous les fichiers sources → invalidation cache si l'un change
    fhash = _files_hash(paths)

    # ── 2. Embeddings ──────────────────────────────────────────────
    log("[2/10] Embeddings (multilingual-e5-base 768d) …")
    embeddings = _load_cached(store_dir, fhash, "embeddings", log)
    if embeddings is None:
        texts = [a.content for a in artifacts]
        embeddings = embed_texts(texts).astype(np.float32)
        _save_cache(store_dir, fhash, "embeddings", embeddings)
    log(f"      shape={embeddings.shape}")

    # ── 3. Goal vectors ────────────────────────────────────────────
    log("[3/10] Goal heuristics …")
    goal_vectors = _load_cached(store_dir, fhash, "goal_vectors", log)
    if goal_vectors is None:
        goal_vectors = compute_goal_vectors(artifacts, embed_fn=embed_texts)
        if goal_vectors is not None:
            _save_cache(store_dir, fhash, "goal_vectors", goal_vectors)
    _attach_goal_vectors(artifacts, goal_vectors)

    # ── 4. Entity extraction ───────────────────────────────────────
    # Backend sélectionnable : "gliner" (défaut, lourd), "spacy", ou "llm" (Qwen2.5 GGUF).
    # GLiNER (multi-v2.1) pour le batch → qualité maximale, résultat mis en cache.
    # Le backend "llm" unifie batch ET query → plus de mismatch NER.
    _ner_label = {"gliner": "GLiNER multi-v2.1", "spacy": "spaCy", "llm": "Qwen2.5 GGUF"}
    log(f"[4/10] Entity extraction ({_ner_label.get(ner_backend, ner_backend)}) …")
    mentions = _load_mentions_cache(store_dir, fhash, log)
    if mentions is None:
        _ner_kwargs = {"model_path": gguf_path, "verbose": verbose} if ner_backend == "llm" and gguf_path else {}
        _extractor_batch = EntityExtractor(backend=ner_backend, **_ner_kwargs)
        mentions = _extractor_batch.extract_batch(artifacts)
        _save_mentions_cache(store_dir, fhash, mentions)
        if ner_backend != "llm":
            del _extractor_batch
            gc.collect()
    log(f"      {len(mentions)} mentions extraites")

    # Extracteur pour les queries recall :
    # Si backend="llm", on réutilise le même → NER unifié (plus de mismatch)
    # Sinon, spaCy léger (~200 MB) par défaut
    if ner_backend == "llm":
        _query_kwargs = {"model_path": gguf_path, "verbose": verbose} if gguf_path else {}
        extractor = EntityExtractor(backend="llm", **_query_kwargs)
    else:
        extractor = EntityExtractor(backend="spacy")

    # ── 5. Entity resolution ───────────────────────────────────────
    log("[5/10] Entity resolution …")
    resolver = EntityResolver(string_threshold=0.75)
    canonical_map = resolver.resolve(mentions)
    n_artifacts = len(artifacts)
    art_id_to_idx = {art.id: i for i, art in enumerate(artifacts)}
    n_injected = _inject_entities(artifacts, canonical_map, art_id_to_idx, n_artifacts)
    log(f"      {len(canonical_map)} entités canoniques, {n_injected} injections")

    # ── 6. Segmentation ────────────────────────────────────────────
    log("[6/10] Segmentation épisodique …")
    seg_alpha, seg_beta = _entity_weights(n_injected)
    segmenter = _build_segmenter(seg, seg_alpha, seg_beta, embeddings, log)
    episodes = segmenter.segment(artifacts, embeddings)
    log(f"      {len(episodes)} épisodes bruts")

    if seg.consolidate:
        episodes = segmenter.consolidate(episodes)
        log(f"      → {len(episodes)} après consolidation")


    # ── 7. Splitting post-segmentation ─────────────────────────────
    log("[7/10] Splitting méga-épisodes …")
    cfg = split_config if split_config is not None else SplitConfig(max_span_hours=48.0)
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
    _sum_label = "LLM abstractif (Qwen2.5)" if summarizer_backend == "llm" else "extractif (TF-IDF)"
    log(f"[9/10] Summarization ({_sum_label}) …")
    if summarizer_backend == "llm":
        from llm_summarizer import create_llm_summarizer
        summarizer = create_llm_summarizer(
            model_path=gguf_path, verbose=verbose
        )
        summarizer.entity_first = (n_injected > 0)
    else:
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
    log("embed OK")

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
        entity_weights= dict.fromkeys(artifact.entities, 1.0),
    )
    episodes.append(ep)
    return ep


def _update_episode_ema(ep, artifact: Artifact, emb: np.ndarray,
                         alpha: float = 0.80) -> None:
    """Met à jour centroïde (EMA), intervalle temporel et entités.

    alpha=0.80 : cohérent avec EpisodeSegmenter.ema_alpha (valeur Optuna-validée).
    Le centroïde reflète le contexte récent — logique pour l'attachement de nouveaux messages.
    """
    ep.centroid = (1 - alpha) * ep.centroid + alpha * emb
    ep.centroid /= np.linalg.norm(ep.centroid) + 1e-9
    t0 = ep.time_interval[0] if ep.time_interval else artifact.timestamp
    ep.time_interval = (t0, artifact.timestamp)
    for ent in artifact.entities:
        ep.entity_weights[ent] = ep.entity_weights.get(ent, 0.0) + 0.5
