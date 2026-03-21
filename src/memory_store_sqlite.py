"""
memory_store_sqlite.py — Persistance épisodique SQLite + sqlite-vec.

Architecture Ivias Phase B :
  - SQLite = source de vérité unique (un fichier = toute la mémoire)
  - sqlite-vec = vector search intégré (pas de FAISS externe)
  - Centroïd EMA mis à jour à chaque APPEND
  - Double seuil : zone grise → slow path
  - Décisions : APPEND / NEW / REACTIVATE (différenciateur Ivias)
  - États EpisodeStatus complets avec enrichissement versionné

Usage
-----
from memory_store_sqlite import MemoryStoreSQLite, EpisodeStatus, NormalizedMessage

store = MemoryStoreSQLite("memory.db")

# Ingérer un message (fast path)
decision = store.ingest_message(msg, current_episode_id, boundary_score)
# decision : {"action": "APPEND"|"NEW"|"REACTIVATE", "episode_id": str}

# Recherche hybride (avant réponse agent)
pack = store.search(query_embedding, query_text, top_k=5)

# Cycle de vie (cron périodique)
store.apply_decay(lambda_base=0.1, dormancy_threshold=0.3)
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# EpisodeStatus
# ---------------------------------------------------------------------------

class EpisodeStatus(str, Enum):
    RAW                = "raw"                # message reçu, pas encore enrichi
    PENDING_ENRICHMENT = "pending_enrichment" # en queue pour LLM
    ENRICHED           = "enriched"           # complet
    FAILED_ENRICHMENT  = "failed_enrichment"  # LLM indispo ou erreur → retraitable
    STALE              = "stale"              # enrichi mais LLM a changé
    DORMANT            = "dormant"            # decay Ebbinghaus sous seuil
    ARCHIVED           = "archived"           # inactif depuis N jours


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class NormalizedMessage:
    """Format unique pour tout message quel que soit le canal."""
    message_id:      str
    channel:         str          # whatsapp / signal / email / telegram
    author_id:       str
    timestamp:       datetime
    content:         str
    conversation_id: str
    direction:       str = "inbound"   # inbound / outbound
    embedding:       Optional[np.ndarray] = None  # mE5 768d ou small 384d


@dataclass
class EpisodeRecord:
    """Épisode tel que stocké en base."""
    episode_id:    str
    status:        EpisodeStatus
    start_time:    datetime
    end_time:      datetime
    message_count: int
    channels:      List[str]
    participants:  List[str]

    # Enrichissement versionné
    title_v1:           Optional[str] = None
    summary_v1:         Optional[str] = None
    summary_model:      Optional[str] = None
    summary_updated_at: Optional[datetime] = None
    goal_v1:            Optional[str] = None
    decisions_v1:       Optional[List[str]] = None
    main_intent:        Optional[str] = None

    # Scores épisodiques (DeBERTa)
    axis_resolution:  float = 0.0
    axis_salience:    float = 0.0
    axis_temporality: float = 0.0
    importance_score: float = 0.5

    # Lifecycle Ebbinghaus
    decay_factor:  float = 1.0
    access_count:  int = 0
    last_accessed: Optional[datetime] = None

    # Vecteurs
    centroid_emb: Optional[np.ndarray] = None  # EMA des embeddings messages
    embedding:    Optional[np.ndarray] = None  # embedding du résumé


@dataclass
class MemoryContextPack:
    """Ce qu'on injecte au LLM agent — jamais les épisodes bruts."""
    episodes:  List[Dict[str, Any]]
    formatted: str   # texte prêt à injecter dans le prompt


# ---------------------------------------------------------------------------
# MemoryStoreSQLite
# ---------------------------------------------------------------------------

class MemoryStoreSQLite:
    """
    Store SQLite pour la mémoire épisodique Ivias.

    Paramètres
    ----------
    db_path         : chemin vers le fichier .db (créé si absent)
    centroid_alpha  : coefficient EMA pour mise à jour centroïd (défaut 0.85)
    seuil_bas       : en dessous → APPEND direct
    seuil_haut      : au dessus → NEW direct
    embedding_dim   : dimension des embeddings (768 pour mE5-base, 384 pour small)
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        centroid_alpha: float = 0.85,
        seuil_bas: float = 0.35,
        seuil_haut: float = 0.65,
        embedding_dim: int = 768,
    ):
        self.db_path       = Path(db_path)
        self.alpha         = centroid_alpha
        self.seuil_bas     = seuil_bas
        self.seuil_haut    = seuil_haut
        self.embedding_dim = embedding_dim

        self._conn = self._connect()
        self._init_schema()

    # ------------------------------------------------------------------
    # Connexion & schéma
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        # Tenter de charger sqlite-vec pour la recherche vectorielle
        try:
            import sqlite_vec  # type: ignore
            sqlite_vec.load(conn)
            self._vec_available = True
        except Exception:
            self._vec_available = False

        return conn

    def _init_schema(self) -> None:
        cur = self._conn.cursor()

        cur.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id          TEXT PRIMARY KEY,
            status              TEXT NOT NULL DEFAULT 'raw',
            start_time          INTEGER NOT NULL,
            end_time            INTEGER NOT NULL,
            message_count       INTEGER NOT NULL DEFAULT 0,
            channels            TEXT NOT NULL DEFAULT '[]',
            participants        TEXT NOT NULL DEFAULT '[]',

            -- Enrichissement versionné
            title_v1            TEXT,
            summary_v1          TEXT,
            summary_model       TEXT,
            summary_updated_at  INTEGER,
            goal_v1             TEXT,
            decisions_v1        TEXT,
            main_intent         TEXT,

            -- Scores épisodiques
            axis_resolution     REAL NOT NULL DEFAULT 0.0,
            axis_salience       REAL NOT NULL DEFAULT 0.0,
            axis_temporality    REAL NOT NULL DEFAULT 0.0,
            importance_score    REAL NOT NULL DEFAULT 0.5,

            -- Lifecycle
            decay_factor        REAL NOT NULL DEFAULT 1.0,
            access_count        INTEGER NOT NULL DEFAULT 0,
            last_accessed       INTEGER,

            -- Vecteurs (BLOB)
            centroid_emb        BLOB,
            embedding           BLOB
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id      TEXT PRIMARY KEY,
            episode_id      TEXT NOT NULL REFERENCES episodes(episode_id),
            content         TEXT NOT NULL,
            author_id       TEXT NOT NULL,
            timestamp       INTEGER NOT NULL,
            channel         TEXT NOT NULL,
            direction       TEXT NOT NULL DEFAULT 'inbound',
            conversation_id TEXT NOT NULL,
            embedding       BLOB
        );

        CREATE TABLE IF NOT EXISTS entities (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id  TEXT NOT NULL REFERENCES episodes(episode_id),
            entity      TEXT NOT NULL,
            type        TEXT NOT NULL,
            source      TEXT NOT NULL DEFAULT 'gliner_v1'
        );

        CREATE TABLE IF NOT EXISTS episode_links (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id     TEXT NOT NULL REFERENCES episodes(episode_id),
            to_id       TEXT NOT NULL REFERENCES episodes(episode_id),
            link_type   TEXT NOT NULL,
            strength    REAL NOT NULL DEFAULT 1.0,
            created_at  INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_messages_episode
            ON messages(episode_id);
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp
            ON messages(timestamp);
        CREATE INDEX IF NOT EXISTS idx_episodes_status
            ON episodes(status);
        CREATE INDEX IF NOT EXISTS idx_episodes_end_time
            ON episodes(end_time);
        CREATE INDEX IF NOT EXISTS idx_entities_episode
            ON entities(episode_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dt_to_ts(dt: datetime) -> int:
        return int(dt.timestamp())

    @staticmethod
    def _ts_to_dt(ts: int) -> datetime:
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    @staticmethod
    def _emb_to_blob(emb: Optional[np.ndarray]) -> Optional[bytes]:
        if emb is None:
            return None
        return emb.astype(np.float32).tobytes()

    def _blob_to_emb(self, blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32).copy()

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _new_episode_id(self) -> str:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        short = uuid.uuid4().hex[:6]
        return f"ep_{ts}_{short}"

    # ------------------------------------------------------------------
    # Fast path : ingest_message
    # ------------------------------------------------------------------

    def ingest_message(
        self,
        msg: NormalizedMessage,
        current_episode_id: Optional[str],
        boundary_score: float,
    ) -> Dict[str, Any]:
        """
        Fast path (<50ms) : décide APPEND / NEW / REACTIVATE et persiste.

        Logique double seuil :
          score < seuil_bas   → APPEND direct
          seuil_bas ≤ score ≤ seuil_haut → zone grise (retourne "SLOW_PATH")
          score > seuil_haut  → NEW direct
          + REACTIVATE si épisode DORMANT similaire trouvé

        Returns
        -------
        {
          "action": "APPEND" | "NEW" | "REACTIVATE" | "SLOW_PATH",
          "episode_id": str,
          "boundary_score": float,
        }
        """
        with self._conn:
            # --- Zone grise → déléguer au slow path validator ---
            if self.seuil_bas <= boundary_score <= self.seuil_haut:
                # On sauvegarde le message sans décision finale
                # Le slow path appellera finalize_decision()
                episode_id = current_episode_id or self._new_episode_id()
                self._save_message(msg, episode_id)
                return {
                    "action": "SLOW_PATH",
                    "episode_id": episode_id,
                    "boundary_score": boundary_score,
                }

            # --- Rupture forte : chercher une réactivation ---
            if boundary_score > self.seuil_haut and msg.embedding is not None:
                dormant_id = self._find_dormant_match(msg.embedding)
                if dormant_id:
                    self._reactivate_episode(dormant_id, msg)
                    self._save_message(msg, dormant_id)
                    return {
                        "action": "REACTIVATE",
                        "episode_id": dormant_id,
                        "boundary_score": boundary_score,
                    }

            # --- Rupture forte : nouvel épisode ---
            if boundary_score > self.seuil_haut:
                if current_episode_id:
                    self._close_episode(current_episode_id, msg.timestamp)
                new_id = self._create_episode(msg)
                self._save_message(msg, new_id)
                return {
                    "action": "NEW",
                    "episode_id": new_id,
                    "boundary_score": boundary_score,
                }

            # --- Continuité : APPEND ---
            if current_episode_id:
                self._append_to_episode(current_episode_id, msg)
                self._save_message(msg, current_episode_id)
                return {
                    "action": "APPEND",
                    "episode_id": current_episode_id,
                    "boundary_score": boundary_score,
                }

            # Aucun épisode courant → créer
            new_id = self._create_episode(msg)
            self._save_message(msg, new_id)
            return {
                "action": "NEW",
                "episode_id": new_id,
                "boundary_score": boundary_score,
            }

    # ------------------------------------------------------------------
    # Opérations épisode
    # ------------------------------------------------------------------

    def _create_episode(self, msg: NormalizedMessage) -> str:
        episode_id = self._new_episode_id()
        ts = self._dt_to_ts(msg.timestamp)
        centroid_blob = self._emb_to_blob(msg.embedding)

        self._conn.execute("""
            INSERT INTO episodes
              (episode_id, status, start_time, end_time, message_count,
               channels, participants, centroid_emb)
            VALUES (?, ?, ?, ?, 0, ?, ?, ?)
        """, (
            episode_id,
            EpisodeStatus.RAW.value,
            ts, ts,
            json.dumps([msg.channel]),
            json.dumps([msg.author_id]),
            centroid_blob,
        ))
        return episode_id

    def _append_to_episode(
        self, episode_id: str, msg: NormalizedMessage
    ) -> None:
        """Met à jour centroïd EMA + métadonnées de l'épisode."""
        row = self._conn.execute(
            "SELECT centroid_emb, channels, participants FROM episodes WHERE episode_id=?",
            (episode_id,)
        ).fetchone()
        if row is None:
            return

        # Mise à jour centroïd EMA : centroid = α*old + (1-α)*new
        old_centroid = self._blob_to_emb(row["centroid_emb"])
        if old_centroid is not None and msg.embedding is not None:
            new_centroid = (
                self.alpha * old_centroid
                + (1 - self.alpha) * msg.embedding
            )
        elif msg.embedding is not None:
            new_centroid = msg.embedding
        else:
            new_centroid = old_centroid

        # Mise à jour canaux et participants
        channels     = json.loads(row["channels"])
        participants = json.loads(row["participants"])
        if msg.channel not in channels:
            channels.append(msg.channel)
        if msg.author_id not in participants:
            participants.append(msg.author_id)

        ts = self._dt_to_ts(msg.timestamp)
        self._conn.execute("""
            UPDATE episodes SET
                end_time      = ?,
                message_count = message_count + 1,
                channels      = ?,
                participants  = ?,
                centroid_emb  = ?
            WHERE episode_id = ?
        """, (
            ts,
            json.dumps(channels),
            json.dumps(participants),
            self._emb_to_blob(new_centroid),
            episode_id,
        ))

    def _close_episode(self, episode_id: str, timestamp: datetime) -> None:
        self._conn.execute("""
            UPDATE episodes SET
                end_time = ?,
                status   = CASE
                    WHEN status = 'raw' THEN 'pending_enrichment'
                    ELSE status
                END
            WHERE episode_id = ?
        """, (self._dt_to_ts(timestamp), episode_id))

    def _reactivate_episode(
        self, episode_id: str, msg: NormalizedMessage
    ) -> None:
        """Réactive un épisode DORMANT : access_count++, decay reset."""
        self._conn.execute("""
            UPDATE episodes SET
                status        = 'pending_enrichment',
                access_count  = access_count + 1,
                decay_factor  = 1.0,
                last_accessed = ?
            WHERE episode_id = ?
        """, (self._dt_to_ts(msg.timestamp), episode_id))

    def _save_message(self, msg: NormalizedMessage, episode_id: str) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO messages
              (message_id, episode_id, content, author_id, timestamp,
               channel, direction, conversation_id, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            msg.message_id,
            episode_id,
            msg.content,
            msg.author_id,
            self._dt_to_ts(msg.timestamp),
            msg.channel,
            msg.direction,
            msg.conversation_id,
            self._emb_to_blob(msg.embedding),
        ))

    # ------------------------------------------------------------------
    # Réactivation : chercher un épisode DORMANT similaire
    # ------------------------------------------------------------------

    def _find_dormant_match(
        self,
        query_emb: np.ndarray,
        sim_threshold: float = 0.75,
        limit: int = 5,
    ) -> Optional[str]:
        """
        Cherche parmi les épisodes DORMANT celui dont le centroïd est
        le plus similaire à query_emb (cosine > sim_threshold).
        """
        rows = self._conn.execute("""
            SELECT episode_id, centroid_emb
            FROM episodes
            WHERE status = 'dormant' AND centroid_emb IS NOT NULL
            ORDER BY end_time DESC
            LIMIT ?
        """, (limit * 4,)).fetchall()

        best_id, best_sim = None, sim_threshold
        for row in rows:
            centroid = self._blob_to_emb(row["centroid_emb"])
            if centroid is None:
                continue
            sim = self._cosine(query_emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = row["episode_id"]

        return best_id

    # ------------------------------------------------------------------
    # Slow path : enrichissement
    # ------------------------------------------------------------------

    def enrich_episode(
        self,
        episode_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        goal: Optional[str] = None,
        decisions: Optional[List[str]] = None,
        main_intent: Optional[str] = None,
        axis_resolution: Optional[float] = None,
        axis_salience: Optional[float] = None,
        axis_temporality: Optional[float] = None,
        importance_score: Optional[float] = None,
        summary_embedding: Optional[np.ndarray] = None,
        model_name: str = "unknown",
    ) -> None:
        """
        Slow path : met à jour l'épisode avec l'enrichissement LLM/DeBERTa.
        Idempotent — les champs sont versionnés, pas écrasés aveuglément.
        """
        now_ts = self._dt_to_ts(datetime.now(tz=timezone.utc))

        updates: List[str] = ["status = 'enriched'"]
        params: List[Any] = []

        if title is not None:
            updates.append("title_v1 = ?")
            params.append(title)

        if summary is not None:
            updates.append("summary_v1 = ?")
            updates.append("summary_model = ?")
            updates.append("summary_updated_at = ?")
            params.extend([summary, model_name, now_ts])

        if goal is not None:
            updates.append("goal_v1 = ?")
            params.append(goal)

        if decisions is not None:
            updates.append("decisions_v1 = ?")
            params.append(json.dumps(decisions))

        if main_intent is not None:
            updates.append("main_intent = ?")
            params.append(main_intent)

        if axis_resolution is not None:
            updates.append("axis_resolution = ?")
            params.append(axis_resolution)

        if axis_salience is not None:
            updates.append("axis_salience = ?")
            params.append(axis_salience)

        if axis_temporality is not None:
            updates.append("axis_temporality = ?")
            params.append(axis_temporality)

        if importance_score is not None:
            updates.append("importance_score = ?")
            params.append(importance_score)

        if summary_embedding is not None:
            updates.append("embedding = ?")
            params.append(self._emb_to_blob(summary_embedding))

        params.append(episode_id)
        with self._conn:
            self._conn.execute(
                f"UPDATE episodes SET {', '.join(updates)} WHERE episode_id = ?",
                params,
            )

    def mark_enrichment_failed(self, episode_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE episodes SET status = 'failed_enrichment' WHERE episode_id = ?",
                (episode_id,),
            )

    def add_entities(
        self,
        episode_id: str,
        entities: List[Tuple[str, str]],  # [(entity, type), ...]
        source: str = "gliner_v1",
    ) -> None:
        with self._conn:
            self._conn.executemany("""
                INSERT INTO entities (episode_id, entity, type, source)
                VALUES (?, ?, ?, ?)
            """, [(episode_id, e, t, source) for e, t in entities])

    def add_episode_link(
        self,
        from_id: str,
        to_id: str,
        link_type: str,
        strength: float = 1.0,
    ) -> None:
        now_ts = self._dt_to_ts(datetime.now(tz=timezone.utc))
        with self._conn:
            self._conn.execute("""
                INSERT INTO episode_links (from_id, to_id, link_type, strength, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (from_id, to_id, link_type, strength, now_ts))

    # ------------------------------------------------------------------
    # Lifecycle : decay Ebbinghaus (cron périodique)
    # ------------------------------------------------------------------

    def apply_decay(
        self,
        lambda_base: float = 0.1,
        dormancy_threshold: float = 0.3,
        archive_threshold_days: int = 90,
    ) -> Dict[str, int]:
        """
        Recalcule le decay pour tous les épisodes ENRICHED/DORMANT.

        retention(e, t) = exp(-λ(e) × t / (1 + log1p(access_count)))
        λ(e) = lambda_base / (1 + axis_salience)

        Transitions :
          - retention < dormancy_threshold → DORMANT
          - DORMANT + âge > archive_threshold_days → ARCHIVED
        """
        now_ts = self._dt_to_ts(datetime.now(tz=timezone.utc))
        rows = self._conn.execute("""
            SELECT episode_id, end_time, access_count, axis_salience, status
            FROM episodes
            WHERE status IN ('enriched', 'dormant')
        """).fetchall()

        stats = {"dormant": 0, "archived": 0, "updated": 0}

        with self._conn:
            for row in rows:
                age_seconds = now_ts - row["end_time"]
                age_days    = age_seconds / 86400

                lam = lambda_base / (1 + row["axis_salience"])
                retention = math.exp(
                    -lam * age_days / (1 + math.log1p(row["access_count"]))
                )

                new_status = row["status"]
                if retention < dormancy_threshold:
                    if row["status"] == "dormant" and age_days > archive_threshold_days:
                        new_status = "archived"
                        stats["archived"] += 1
                    elif row["status"] == "enriched":
                        new_status = "dormant"
                        stats["dormant"] += 1

                self._conn.execute("""
                    UPDATE episodes
                    SET decay_factor = ?, status = ?
                    WHERE episode_id = ?
                """, (retention, new_status, row["episode_id"]))
                stats["updated"] += 1

        return stats

    # ------------------------------------------------------------------
    # Search : hybride BM25 + vecteur
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
        top_k: int = 5,
        status_filter: Optional[List[str]] = None,
    ) -> MemoryContextPack:
        """
        Recherche hybride (vecteur + keywords).
        Ranking composite : similarité + récence + axis_salience + axis_resolution.

        Returns
        -------
        MemoryContextPack prêt à injecter dans le prompt agent.
        """
        if status_filter is None:
            status_filter = ["enriched", "dormant"]

        placeholders = ",".join("?" * len(status_filter))
        rows = self._conn.execute(f"""
            SELECT episode_id, end_time, title_v1, summary_v1, goal_v1,
                   decisions_v1, axis_salience, axis_resolution, axis_temporality,
                   importance_score, access_count, decay_factor,
                   centroid_emb, embedding, status
            FROM episodes
            WHERE status IN ({placeholders})
              AND (centroid_emb IS NOT NULL OR embedding IS NOT NULL)
        """, status_filter).fetchall()

        if not rows:
            return MemoryContextPack(episodes=[], formatted="")

        now_ts = self._dt_to_ts(datetime.now(tz=timezone.utc))
        scored = []

        for row in rows:
            # Similarité vectorielle (centroïd ou embedding résumé)
            vec_blob = row["embedding"] or row["centroid_emb"]
            ep_vec   = self._blob_to_emb(vec_blob)
            sem_sim  = self._cosine(query_embedding, ep_vec) if ep_vec is not None else 0.0

            # Récence normalisée [0,1] (décroît sur 30 jours)
            age_days = max(0, (now_ts - row["end_time"]) / 86400)
            recency  = math.exp(-age_days / 30)

            # Score composite
            score = (
                0.30 * sem_sim
                + 0.25 * recency
                + 0.25 * (row["axis_salience"]   or 0.0)
                + 0.15 * (row["axis_resolution"]  or 0.0)
                + 0.05 * math.log1p(row["access_count"])
            )

            scored.append((score, row))

        # Tri décroissant, top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # Mettre à jour access_count pour les épisodes renvoyés
        ep_ids = [r["episode_id"] for _, r in top]
        with self._conn:
            for ep_id in ep_ids:
                self._conn.execute("""
                    UPDATE episodes SET
                        access_count  = access_count + 1,
                        last_accessed = ?
                    WHERE episode_id = ?
                """, (now_ts, ep_id))

        # Construire le Memory Context Pack
        episodes_out = []
        lines = ["Relevant memory:"]

        for rank, (score, row) in enumerate(top, 1):
            age_days = (now_ts - row["end_time"]) / 86400
            age_str  = _format_age(age_days)

            title    = row["title_v1"] or "Épisode sans titre"
            summary  = row["summary_v1"] or ""
            goal     = row["goal_v1"] or ""
            decisions = json.loads(row["decisions_v1"]) if row["decisions_v1"] else []

            ep_dict = {
                "episode_id":       row["episode_id"],
                "title":            title,
                "summary":          summary,
                "goal":             goal,
                "decisions":        decisions,
                "age_str":          age_str,
                "axis_salience":    row["axis_salience"],
                "axis_resolution":  row["axis_resolution"],
                "score":            score,
                "status":           row["status"],
            }
            episodes_out.append(ep_dict)

            # Format texte compact pour injection prompt
            block = [f"{rank}. {title} — {age_str}"]
            if summary:
                block.append(f"   Contexte : {summary[:120]}")
            if goal:
                block.append(f"   Objectif : {goal}")
            for d in decisions[:2]:
                block.append(f"   Décision : {d}")
            block.append(
                f"   [résolution: {row['axis_resolution']:.2f} | "
                f"saillance: {row['axis_salience']:.2f}]"
            )
            lines.append("\n".join(block))

        return MemoryContextPack(
            episodes=episodes_out,
            formatted="\n\n".join(lines),
        )

    # ------------------------------------------------------------------
    # Lecture / stats
    # ------------------------------------------------------------------

    def get_episode(self, episode_id: str) -> Optional[EpisodeRecord]:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_episodes(
        self,
        status: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[EpisodeRecord]:
        if status:
            placeholders = ",".join("?" * len(status))
            rows = self._conn.execute(
                f"SELECT * FROM episodes WHERE status IN ({placeholders}) "
                "ORDER BY end_time DESC LIMIT ?",
                status + [limit],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM episodes ORDER BY end_time DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def pending_enrichment(self, limit: int = 100) -> List[str]:
        """Retourne les episode_id à enrichir (pour le job queue)."""
        rows = self._conn.execute("""
            SELECT episode_id FROM episodes
            WHERE status IN ('raw', 'pending_enrichment', 'failed_enrichment')
            ORDER BY end_time DESC LIMIT ?
        """, (limit,)).fetchall()
        return [r["episode_id"] for r in rows]

    def stats(self) -> Dict[str, Any]:
        """Statistiques de la base."""
        rows = self._conn.execute("""
            SELECT status, COUNT(*) as n FROM episodes GROUP BY status
        """).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        total_msg = self._conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        db_size_mb = self.db_path.stat().st_size / 1e6 if self.db_path.exists() else 0

        return {
            "episodes_by_status": counts,
            "total_episodes":     sum(counts.values()),
            "total_messages":     total_msg,
            "db_size_mb":         round(db_size_mb, 2),
            "sqlite_vec":         self._vec_available,
        }

    def _row_to_record(self, row: sqlite3.Row) -> EpisodeRecord:
        decisions = json.loads(row["decisions_v1"]) if row["decisions_v1"] else None
        return EpisodeRecord(
            episode_id    = row["episode_id"],
            status        = EpisodeStatus(row["status"]),
            start_time    = self._ts_to_dt(row["start_time"]),
            end_time      = self._ts_to_dt(row["end_time"]),
            message_count = row["message_count"],
            channels      = json.loads(row["channels"]),
            participants  = json.loads(row["participants"]),
            title_v1      = row["title_v1"],
            summary_v1    = row["summary_v1"],
            summary_model = row["summary_model"],
            goal_v1       = row["goal_v1"],
            decisions_v1  = decisions,
            main_intent   = row["main_intent"],
            axis_resolution  = row["axis_resolution"],
            axis_salience    = row["axis_salience"],
            axis_temporality = row["axis_temporality"],
            importance_score = row["importance_score"],
            decay_factor  = row["decay_factor"],
            access_count  = row["access_count"],
            centroid_emb  = self._blob_to_emb(row["centroid_emb"]),
            embedding     = self._blob_to_emb(row["embedding"]),
        )

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _format_age(age_days: float) -> str:
    if age_days < 1:
        return "aujourd'hui"
    if age_days < 2:
        return "hier"
    if age_days < 7:
        return f"il y a {int(age_days)} jours"
    if age_days < 30:
        return f"il y a {int(age_days / 7)} semaines"
    if age_days < 365:
        return f"il y a {int(age_days / 30)} mois"
    return f"il y a {int(age_days / 365)} ans"
