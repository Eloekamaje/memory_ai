"""
memory_index.py — Index sémantique et entités pour le recall (Phase B5).

Problème résolu :
  recall_engine.py:352 itère sur TOUS les épisodes à chaque requête — O(n).
  Sur 5 000 épisodes avec embeddings dim=384, ça fait ~10M dot products par requête.

Solution :
  1. Index sémantique ANN sur les centroïdes (sklearn NearestNeighbors, cosine)
     → réduit O(n) à O(k log n) pour la partie sémantique
  2. Index inversé entités → episode_ids (dict, O(1) lookup)
     → union des candidats entité sans scan

Usage :
    index = MemoryIndex()
    index.build(episodes)

    # dans RecallEngine.recall() :
    candidates = index.candidates(query_emb, query_entity_names, top_k_sem=20)
    # → scorer uniquement ces candidats au lieu de tous les épisodes

Persistance :
    index.save("memory/index")   # → index_centroids.npy + index_meta.json
    index.load("memory/index")   # → reconstruit NearestNeighbors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.neighbors import NearestNeighbors

from models import Episode, EpisodeState


class MemoryIndex:
    """
    Index en mémoire pour accélérer le recall multi-signal.

    Deux composantes :
      - _nn  : NearestNeighbors sur les centroïdes (sémantique)
      - _ent : Dict[entity_name → Set[episode_id]] (entités)

    L'index est rebuilt intégralement à chaque appel à build().
    C'est suffisant tant que le nombre d'épisodes < ~50 000.
    Pour plus, passer à FAISS HNSW (même interface).
    """

    def __init__(self, n_neighbors: int = 20):
        """
        Parameters
        ----------
        n_neighbors : nombre de voisins sémantiques retournés par défaut.
            Doit être ≥ top_k utilisé dans candidates().
        """
        self.n_neighbors = n_neighbors
        self._nn: Optional[NearestNeighbors] = None
        self._ep_ids: List[str] = []                       # index → episode_id
        self._ep_id_set: Set[str] = set()                  # pour lookup rapide
        self._ent: Dict[str, Set[str]] = {}                # entity → episode_ids
        self._built = False

    # -------------------------------------------------------------- #
    # build()                                                          #
    # -------------------------------------------------------------- #

    def build(
        self,
        episodes: List[Episode],
        exclude_archived: bool = False,
    ) -> "MemoryIndex":
        """
        Construit l'index à partir des épisodes.

        Seuls les épisodes avec un centroïde non-None sont indexés
        (les épisodes compressés par MemoryStore.compress_archived()
        gardent leur centroïde — ils restent donc searchables).

        Parameters
        ----------
        episodes : liste d'Episode (après segmentation ou chargement)
        exclude_archived : si True, exclut les ARCHIVED de l'index ANN
        """
        self._ep_ids = []
        self._ep_id_set = set()
        self._ent = {}
        centroids: List[np.ndarray] = []

        for ep in episodes:
            # --- Index entités (tous les épisodes, même sans centroïde) ---
            for ent in ep.entity_weights:
                self._ent.setdefault(ent, set()).add(ep.id)

            # --- Index sémantique (besoin d'un centroïde) ---
            if ep.centroid is None:
                continue
            if exclude_archived and ep.state == EpisodeState.ARCHIVED:
                continue

            self._ep_ids.append(ep.id)
            self._ep_id_set.add(ep.id)
            centroids.append(ep.centroid)

        if not centroids:
            self._built = False
            return self

        mat = np.array(centroids, dtype=np.float32)

        # NearestNeighbors avec cosine distance
        # n_neighbors limité à ce qui est disponible
        k = min(self.n_neighbors, len(self._ep_ids))
        self._nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        self._nn.fit(mat)
        self._centroids = mat   # conservé pour save/load

        self._built = True
        return self

    # -------------------------------------------------------------- #
    # candidates()                                                     #
    # -------------------------------------------------------------- #

    def candidates(
        self,
        query_emb: np.ndarray,
        query_entity_names: Optional[Set[str]] = None,
        top_k_sem: int = 20,
    ) -> Set[str]:
        """
        Retourne un ensemble d'episode_ids candidats pour le scoring complet.

        Candidats = union de :
          - top_k_sem épisodes les plus proches sémantiquement (ANN)
          - tous les épisodes contenant ≥1 entité de la requête

        Le RecallEngine ne score que ces candidats au lieu de n'importe quel épisode.

        Parameters
        ----------
        query_emb : embedding de la requête (1D array)
        query_entity_names : entités extraites de la requête (noms canoniques)
        top_k_sem : nombre de voisins sémantiques à inclure
        """
        result: Set[str] = set()

        # --- Voisins sémantiques ---
        if self._built and self._nn is not None:
            q = query_emb.reshape(1, -1).astype(np.float32)
            k = min(top_k_sem, len(self._ep_ids))
            _, indices = self._nn.kneighbors(q, n_neighbors=k)
            for idx in indices[0]:
                result.add(self._ep_ids[idx])

        # --- Entités ---
        if query_entity_names:
            for ent in query_entity_names:
                key = ent.lower().strip()
                result.update(self._ent.get(key, set()))

        return result

    def add_episode(self, ep: "Episode") -> None:
        """
        Ajoute un épisode à l'index sans rebuild complet.

        - Index entités : O(k) mise à jour immédiate.
        - Index sémantique ANN : rebuild léger (sklearn brute-force, ms pour <5K eps).
          Pour >5K épisodes, migrer vers FAISS IVF-PQ (H5.a).

        Parameters
        ----------
        ep : Episode à ajouter (doit avoir un centroïde non-None pour l'ANN)
        """
        # --- Index entités (immédiat) ---
        for ent in ep.entity_weights:
            key = ent.lower().strip()
            self._ent.setdefault(key, set()).add(ep.id)

        if ep.id in self._ep_id_set or ep.centroid is None:
            return  # déjà indexé ou pas de centroïde → pas d'ANN update

        # --- Index sémantique ANN (rebuild) ---
        self._ep_ids.append(ep.id)
        self._ep_id_set.add(ep.id)

        if hasattr(self, "_centroids") and self._centroids is not None:
            new_centroid = ep.centroid.reshape(1, -1).astype(np.float32)
            mat = np.vstack([self._centroids, new_centroid])
        else:
            mat = ep.centroid.reshape(1, -1).astype(np.float32)

        k = min(self.n_neighbors, len(self._ep_ids))
        self._nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        self._nn.fit(mat)
        self._centroids = mat
        self._built = True

    def entity_episodes(self, entity_name: str) -> Set[str]:
        """Épisodes contenant une entité donnée (O(1))."""
        return self._ent.get(entity_name.lower().strip(), set())

    # -------------------------------------------------------------- #
    # Persistance                                                      #
    # -------------------------------------------------------------- #

    def save(self, prefix: str) -> None:
        """
        Sauvegarde l'index.

        Fichiers créés :
          <prefix>_centroids.npy   — matrice des centroïdes
          <prefix>_meta.json       — ep_ids + entity index
        """
        if not self._built:
            return

        prefix_path = Path(prefix)
        np.save(str(prefix_path) + "_centroids.npy", self._centroids)

        meta = {
            "ep_ids": self._ep_ids,
            "n_neighbors": self.n_neighbors,
            # entity index : Set → List pour JSON
            "ent": {k: list(v) for k, v in self._ent.items()},
        }
        with open(str(prefix_path) + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, prefix: str) -> "MemoryIndex":
        """Recharge l'index depuis le disque et reconstruit NearestNeighbors."""
        prefix_path = Path(prefix)
        centroids_path = str(prefix_path) + "_centroids.npy"
        meta_path = str(prefix_path) + "_meta.json"

        if not Path(centroids_path).exists():
            return self

        self._centroids = np.load(centroids_path)

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self._ep_ids = meta["ep_ids"]
        self._ep_id_set = set(self._ep_ids)
        self.n_neighbors = meta.get("n_neighbors", self.n_neighbors)
        self._ent = {k: set(v) for k, v in meta.get("ent", {}).items()}

        k = min(self.n_neighbors, len(self._ep_ids))
        if k > 0:
            self._nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
            self._nn.fit(self._centroids)
            self._built = True

        return self

    # -------------------------------------------------------------- #
    # Stats                                                            #
    # -------------------------------------------------------------- #

    def __repr__(self) -> str:
        status = "built" if self._built else "empty"
        return (
            f"MemoryIndex({status}, "
            f"{len(self._ep_ids)} episodes, "
            f"{len(self._ent)} entities)"
        )
