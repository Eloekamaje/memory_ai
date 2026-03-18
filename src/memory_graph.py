"""
memory_graph.py — Graphe de mémoire épisodique (Phase B2).

Graphe biparti  Episode ⟷ Entité  qui permet :
  • requêtes cross-épisode  ("quels épisodes mentionnent GEDDVIT ?")
  • similarité inter-épisode via entités partagées
  • cohésion multi-critère par épisode  (§3 du framework)
  • détection d'appartenance floue  (§7 — un artefact ∈ plusieurs épisodes)
  • export Graphviz DOT pour visualisation

Le graphe est construit *après* segmentation + entity resolution.
Il n'interfère pas avec le pipeline existant — c'est une couche d'analyse.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models import Artifact, Episode, EpisodeState

# ---------- tentative d'import entity_resolver (optionnel) ----------
try:
    from entity_resolver import CanonicalEntity, EntityResolver
    from entity_extractor import EntityType
except ImportError:          # pragma: no cover
    CanonicalEntity = None   # type: ignore
    EntityResolver = None    # type: ignore
    EntityType = None        # type: ignore


# ================================================================== #
# Nœuds du graphe                                                     #
# ================================================================== #

@dataclass
class EpisodeNode:
    """Nœud épisode dans le graphe biparti."""
    episode_id: str
    entity_names: Set[str] = field(default_factory=set)
    artifact_count: int = 0
    time_interval: Optional[Tuple[datetime, datetime]] = None
    centroid: Optional[np.ndarray] = field(default=None, repr=False)
    state: EpisodeState = EpisodeState.ACTIVE


@dataclass
class EntityNode:
    """Nœud entité dans le graphe biparti."""
    canonical_name: str
    entity_type: str = "UNKNOWN"       # EntityType.value
    episode_ids: Set[str] = field(default_factory=set)
    total_weight: float = 0.0          # ∑ poids dans tous les épisodes
    mention_count: int = 0             # total des mentions brutes


@dataclass
class CohesionScore:
    """Score de cohésion multi-critère pour un épisode (§3)."""
    semantic: float = 0.0              # SemCoh(e)
    entity: float = 0.0               # EntCoh(e)
    temporal: float = 0.0             # TempCoh(e)
    global_score: float = 0.0         # Coh(e) = α·Sem + β·Ent + γ·Temp


# ================================================================== #
# Memory Graph                                                         #
# ================================================================== #

class MemoryGraph:
    """
    Graphe biparti Episode ⟷ Entité.

    Arêtes pondérées :  (episode_id, entity_name) → poids
    Le poids correspond à ``episode.entity_weights[entity]``.
    """

    def __init__(self):
        # --- Nœuds ---
        self.episode_nodes: Dict[str, EpisodeNode] = {}
        self.entity_nodes: Dict[str, EntityNode] = {}

        # --- Arêtes (biparti) ---
        #   clé = (episode_id, entity_name), valeur = poids
        self._edges: Dict[Tuple[str, str], float] = {}

        # --- Index inversés ---
        #   episode_id  → set d'entity_names
        #   entity_name → set d'episode_ids
        self._ep_to_ents: Dict[str, Set[str]] = defaultdict(set)
        self._ent_to_eps: Dict[str, Set[str]] = defaultdict(set)

        # --- Références source ---
        self._artifacts: Optional[List[Artifact]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._resolver: Optional[object] = None   # EntityResolver

    # ============================================================== #
    # Requêtes cross-épisode                                           #
    # ============================================================== #

    def entity_episodes(self, entity_name: str) -> List[str]:
        """Retourne les episode_ids contenant *entity_name*."""
        key = entity_name.lower().strip()
        return sorted(self._ent_to_eps.get(key, set()))

    def episode_entities(self, episode_id: str) -> Dict[str, float]:
        """Entités d'un épisode avec leurs poids."""
        return {
            ent: self._edges[(episode_id, ent)]
            for ent in self._ep_to_ents.get(episode_id, set())
        }

    def shared_entities(self, ep1: str, ep2: str) -> Set[str]:
        """Entités partagées entre deux épisodes."""
        return self._ep_to_ents.get(ep1, set()) & self._ep_to_ents.get(ep2, set())

    def episode_entity_similarity(self, ep1: str, ep2: str) -> float:
        """
        Similarité Jaccard pondérée entre deux épisodes via leurs entités.

        $$\\text{sim}(e_i, e_j) = \\frac{\\sum_{u \\in U_i \\cap U_j} (w_i(u) + w_j(u))}
                                        {\\sum_{u \\in U_i \\cup U_j} (w_i(u) + w_j(u))}$$
        """
        ents1 = self._ep_to_ents.get(ep1, set())
        ents2 = self._ep_to_ents.get(ep2, set())

        union = ents1 | ents2
        if not union:
            return 0.0

        inter = ents1 & ents2
        if not inter:
            return 0.0

        w_inter = sum(
            self._edges.get((ep1, e), 0.0) + self._edges.get((ep2, e), 0.0)
            for e in inter
        )
        w_union = sum(
            self._edges.get((ep1, e), 0.0) + self._edges.get((ep2, e), 0.0)
            for e in union
        )

        return w_inter / w_union if w_union > 0 else 0.0

    def related_episodes(
        self,
        episode_id: str,
        min_similarity: float = 0.0,
        top_k: int = 0,
    ) -> List[Tuple[str, float, FrozenSet[str]]]:
        """
        Épisodes reliés à *episode_id* via des entités partagées.

        Returns
        -------
        Liste de tuples (other_ep_id, similarity, shared_entity_set)
        triés par similarité décroissante.
        """
        ents = self._ep_to_ents.get(episode_id, set())
        if not ents:
            return []

        # tous les épisodes partageant ≥1 entité
        neighbor_ids: Set[str] = set()
        for ent in ents:
            neighbor_ids |= self._ent_to_eps.get(ent, set())
        neighbor_ids.discard(episode_id)

        results = []
        for nid in neighbor_ids:
            shared = self.shared_entities(episode_id, nid)
            sim = self.episode_entity_similarity(episode_id, nid)
            if sim >= min_similarity:
                results.append((nid, sim, frozenset(shared)))

        results.sort(key=lambda x: x[1], reverse=True)

        if top_k > 0:
            results = results[:top_k]

        return results

    # ============================================================== #
    # Bridge entities (entités-pont entre épisodes)                    #
    # ============================================================== #

    def bridge_entities(self, min_episodes: int = 2) -> List[Tuple[str, int, float]]:
        """
        Entités apparaissant dans ≥ *min_episodes* épisodes.

        Ce sont les **concepts stables** qui relient la mémoire.

        Returns
        -------
        Liste (entity_name, nb_episodes, total_weight) triée par nb desc.
        """
        results = []
        for ename, enode in self.entity_nodes.items():
            n = len(enode.episode_ids)
            if n >= min_episodes:
                results.append((ename, n, enode.total_weight))
        results.sort(key=lambda x: (-x[1], -x[2]))
        return results

    # ============================================================== #
    # §3 — Cohésion multi-critère                                      #
    # ============================================================== #

    def cohesion(
        self,
        episode_id: str,
        alpha: float = 0.50,
        beta: float = 0.30,
        gamma: float = 0.20,
        lambda_decay: float = 0.001,
    ) -> CohesionScore:
        """
        Calcule la cohésion d'un épisode :

        $$\\text{Coh}(e) = \\alpha \\cdot \\text{SemCoh}
                          + \\beta  \\cdot \\text{EntCoh}
                          + \\gamma \\cdot \\text{TempCoh}$$

        Parameters
        ----------
        alpha, beta, gamma : poids (doivent sommer à 1)
        lambda_decay : λ dans  $\\exp(-\\lambda (t_{max} - t_{min}))$
        """
        ep_node = self.episode_nodes.get(episode_id)
        if ep_node is None:
            return CohesionScore()

        artifact_indices = self._get_artifact_indices(episode_id)

        # ----- SemCoh -----
        sem_coh = self._semantic_cohesion(artifact_indices, ep_node.centroid)

        # ----- EntCoh -----
        ent_coh = self._entity_cohesion(artifact_indices, episode_id)

        # ----- TempCoh -----
        temp_coh = self._temporal_cohesion(ep_node.time_interval, lambda_decay)

        global_score = alpha * sem_coh + beta * ent_coh + gamma * temp_coh

        return CohesionScore(
            semantic=round(sem_coh, 4),
            entity=round(ent_coh, 4),
            temporal=round(temp_coh, 4),
            global_score=round(global_score, 4),
        )

    def _get_artifact_indices(self, episode_id: str) -> List[int]:
        """Retourne les indices d'artefacts d'un épisode (via les Episodes sources)."""
        # On cherche dans les épisodes passés à build()
        # Par convention, les artefacts sont indexés dans la liste self._artifacts
        if self._artifacts is None:
            return []
        ep_node = self.episode_nodes.get(episode_id)
        if ep_node is None:
            return []
        # Reconstruction : on a stocké les episode_ids dans les artefacts
        # mais en fait les Episodes d'origine contiennent artifact_indices.
        # On ne stocke pas les Episode objects — on les reconstruit.
        # Méthode pragmatique : parcourir les artefacts, matcher via entités+timestamps.
        # Mais c'est coûteux. Mieux : stocker le mapping à la construction.
        return self._episode_artifact_indices.get(episode_id, [])

    def build(
        self,
        episodes: List[Episode],
        artifacts: List[Artifact],
        embeddings: Optional[np.ndarray] = None,
        resolver: Optional[object] = None,
    ) -> "MemoryGraph":
        """
        Construit le graphe à partir du pipeline.

        Parameters
        ----------
        episodes : sortie de ``EpisodeSegmenter.segment()``
        artifacts : liste d'artefacts (enrichis d'entités canoniques)
        embeddings : matrice (n, dim) — nécessaire pour la cohésion sémantique
        resolver : instance d'``EntityResolver`` — enrichit les EntityNodes
        """
        self._artifacts = artifacts
        self._embeddings = embeddings
        self._resolver = resolver

        # mapping episode → artifact_indices (pour cohésion)
        self._episode_artifact_indices: Dict[str, List[int]] = {}

        # --- 1. Nœuds épisode + arêtes biparti ---
        for ep in episodes:
            self._episode_artifact_indices[ep.id] = list(ep.artifact_indices)

            enode = EpisodeNode(
                episode_id=ep.id,
                artifact_count=len(ep.artifact_indices),
                time_interval=ep.time_interval,
                centroid=ep.centroid,
                state=ep.state,
            )

            for ent_name, weight in ep.entity_weights.items():
                enode.entity_names.add(ent_name)
                self._edges[(ep.id, ent_name)] = weight
                self._ep_to_ents[ep.id].add(ent_name)
                self._ent_to_eps[ent_name].add(ep.id)

            self.episode_nodes[ep.id] = enode

        # --- 2. Nœuds entité ---
        all_entity_names: Set[str] = set()
        for names in self._ep_to_ents.values():
            all_entity_names |= names

        for ent_name in all_entity_names:
            total_w = sum(
                self._edges.get((eid, ent_name), 0.0)
                for eid in self._ent_to_eps[ent_name]
            )
            enode = EntityNode(
                canonical_name=ent_name,
                episode_ids=set(self._ent_to_eps[ent_name]),
                total_weight=total_w,
            )

            # enrichir via le resolver si dispo
            if resolver is not None and hasattr(resolver, "lookup"):
                ce = resolver.lookup(ent_name)
                if ce is not None:
                    enode.entity_type = ce.entity_type.value if hasattr(ce.entity_type, "value") else str(ce.entity_type)
                    enode.mention_count = ce.mention_count

            self.entity_nodes[ent_name] = enode

        return self

    def _semantic_cohesion(
        self,
        artifact_indices: List[int],
        centroid: Optional[np.ndarray],
    ) -> float:
        """
        $\\text{SemCoh}(e) = \\frac{1}{|A_e|} \\sum_{a_i \\in A_e} \\text{sim}(v_i, C_e)$
        """
        if self._embeddings is None or centroid is None or not artifact_indices:
            return 0.0

        sims = []
        c = centroid.reshape(1, -1)
        for idx in artifact_indices:
            if idx < len(self._embeddings):
                v = self._embeddings[idx].reshape(1, -1)
                sim = float(cosine_similarity(v, c)[0][0])
                sims.append(max(sim, 0.0))

        return float(np.mean(sims)) if sims else 0.0

    def _entity_cohesion(
        self,
        artifact_indices: List[int],
        episode_id: str,
    ) -> float:
        """
        $\\text{EntCoh}(e) = \\frac{1}{|A_e|} \\sum_{a_i \\in A_e} \\text{overlap}(E_i, U_e)$

        overlap = |E_i ∩ keys(U_e)| / |keys(U_e)| si U_e non vide.
        """
        if self._artifacts is None or not artifact_indices:
            return 0.0

        ep_entities = self._ep_to_ents.get(episode_id, set())
        if not ep_entities:
            return 0.0

        overlaps = []
        for idx in artifact_indices:
            if idx < len(self._artifacts):
                a = self._artifacts[idx]
                a_ents = {e.lower().strip() for e in a.entities} if a.entities else set()
                inter = a_ents & ep_entities
                overlaps.append(len(inter) / len(ep_entities))

        return float(np.mean(overlaps)) if overlaps else 0.0

    def _temporal_cohesion(
        self,
        time_interval: Optional[Tuple[datetime, datetime]],
        lambda_decay: float,
    ) -> float:
        """
        $\\text{TempCoh}(e) = \\exp(-\\lambda (t_{max} - t_{min}))$

        Épisode concentré temporellement → TempCoh proche de 1.
        """
        if time_interval is None:
            return 1.0
        t_min, t_max = time_interval
        span_minutes = (t_max - t_min).total_seconds() / 60.0
        return float(math.exp(-lambda_decay * span_minutes))

    # ============================================================== #
    # §7 — Appartenance floue (multi-épisode)                         #
    # ============================================================== #

    def _artifact_entity_scores(
        self,
        artifact: Artifact,
        episodes: List[Episode],
    ) -> List[Tuple[str, float]]:
        """Score entité d'un artefact contre chaque épisode."""
        a_ents = {e.lower().strip() for e in artifact.entities} if artifact.entities else set()
        if not a_ents:
            return []

        scores: List[Tuple[str, float]] = []
        for ep in episodes:
            ep_ents = set(ep.entity_weights.keys())
            inter = a_ents & ep_ents
            if not inter:
                continue
            union = a_ents | ep_ents
            w_inter = sum(ep.entity_weights.get(e, 0.0) for e in inter)
            max_w = max(ep.entity_weights.values()) if ep.entity_weights else 1.0
            score = w_inter / (len(union) * max_w) if max_w > 0 else 0.0
            scores.append((ep.id, score))

        return scores

    def multi_episode_candidates(
        self,
        episodes: List[Episode],
        threshold_ratio: float = 0.7,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Détecte les artefacts qui auraient pu appartenir à plusieurs épisodes.

        Un candidat est retenu si son score ≥ *threshold_ratio* × score max.

        Returns
        -------
        Dict[artifact_index → List[(episode_id, score)]] pour les artefacts multi.
        """
        if self._artifacts is None:
            return {}

        # index inverse : artifact_index → episode attribué
        art_to_ep: Dict[int, str] = {}
        for ep in episodes:
            for idx in ep.artifact_indices:
                art_to_ep[idx] = ep.id

        results: Dict[int, List[Tuple[str, float]]] = {}

        for idx, artifact in enumerate(self._artifacts):
            if art_to_ep.get(idx) is None:
                continue

            scores = self._artifact_entity_scores(artifact, episodes)
            if len(scores) < 2:
                continue

            scores.sort(key=lambda x: x[1], reverse=True)
            best_score = scores[0][1]
            if best_score <= 0:
                continue

            threshold = threshold_ratio * best_score
            multi = [(eid, s) for eid, s in scores if s >= threshold]
            if len(multi) >= 2:
                results[idx] = multi

        return results

    # ============================================================== #
    # Composantes connexes                                             #
    # ============================================================== #

    def connected_components(self) -> List[Set[str]]:
        """
        Composantes connexes du graphe (sur les épisodes).

        Deux épisodes sont connectés s'ils partagent ≥1 entité.
        Retourne des ensembles d'episode_ids.
        """
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Initialiser
        for eid in self.episode_nodes:
            parent[eid] = eid

        # Unifier via entités partagées
        for ent_name, ep_ids in self._ent_to_eps.items():
            eps = list(ep_ids)
            for i in range(1, len(eps)):
                union(eps[0], eps[i])

        # Regrouper
        components: Dict[str, Set[str]] = defaultdict(set)
        for eid in self.episode_nodes:
            components[find(eid)].add(eid)

        # Trier par taille décroissante
        return sorted(components.values(), key=len, reverse=True)

    # ============================================================== #
    # Statistiques & affichage                                         #
    # ============================================================== #

    @property
    def n_episodes(self) -> int:
        return len(self.episode_nodes)

    @property
    def n_entities(self) -> int:
        return len(self.entity_nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    def density(self) -> float:
        """Densité du graphe biparti : |edges| / (|episodes| × |entities|)."""
        max_edges = self.n_episodes * self.n_entities
        return self.n_edges / max_edges if max_edges > 0 else 0.0

    def summary(self) -> str:
        """Résumé textuel du graphe."""
        lines = [
            "╔══════════════════════════════════════╗",
            "║         MEMORY GRAPH SUMMARY         ║",
            "╚══════════════════════════════════════╝",
            f"  Épisodes   : {self.n_episodes}",
            f"  Entités    : {self.n_entities}",
            f"  Arêtes     : {self.n_edges}",
            f"  Densité    : {self.density():.4f}",
        ]

        # Composantes connexes
        cc = self.connected_components()
        lines.append(f"  Composantes: {len(cc)}")
        if cc:
            sizes = [len(c) for c in cc]
            lines.append(f"    tailles  : {sizes[:10]}{'…' if len(sizes) > 10 else ''}")

        # Épisodes isolés (aucune entité → aucun lien)
        isolated = sum(1 for eid in self.episode_nodes if not self._ep_to_ents.get(eid))
        if isolated:
            lines.append(f"  Isolés     : {isolated} épisodes sans entité")

        # Bridge entities
        bridges = self.bridge_entities(min_episodes=2)
        if bridges:
            lines.append(f"\n  ── Entités-pont (≥2 épisodes) : {len(bridges)} ──")
            for name, n_eps, tw in bridges[:15]:
                etype = self.entity_nodes[name].entity_type if name in self.entity_nodes else "?"
                lines.append(f"    {name:30s}  {etype:10s}  {n_eps} épisodes  poids={tw:.0f}")

        # Top entités par poids
        top_ents = sorted(
            self.entity_nodes.values(),
            key=lambda e: e.total_weight, reverse=True,
        )[:10]
        if top_ents:
            lines.append("\n  ── Top-10 entités (poids total) ──")
            for e in top_ents:
                lines.append(
                    f"    {e.canonical_name:30s}  {e.entity_type:10s}  "
                    f"poids={e.total_weight:.0f}  épisodes={len(e.episode_ids)}"
                )

        return "\n".join(lines)

    # ============================================================== #
    # Matrice de similarité inter-épisodes                             #
    # ============================================================== #

    def episode_similarity_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Matrice N×N de similarité via entités partagées.

        Returns
        -------
        (episode_ids, sim_matrix) où sim_matrix[i,j] = entity_similarity(ep_i, ep_j)
        """
        ep_ids = sorted(self.episode_nodes.keys())
        n = len(ep_ids)
        mat = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                s = self.episode_entity_similarity(ep_ids[i], ep_ids[j])
                mat[i, j] = s
                mat[j, i] = s

        return ep_ids, mat

    # ============================================================== #
    # Export Graphviz DOT                                               #
    # ============================================================== #

    def to_dot(
        self,
        max_episodes: int = 50,
        max_entities: int = 80,
        min_entity_weight: float = 2.0,
    ) -> str:
        """
        Export au format DOT (Graphviz) pour visualisation.

        Filtre les entités de poids < min_entity_weight pour la lisibilité.
        """
        lines = [
            "graph MemoryGraph {",
            "  graph [rankdir=LR, overlap=false, splines=true];",
            '  node [fontname="Helvetica", fontsize=10];',
            "",
        ]

        # Sélection des entités à afficher
        shown_entities: Set[str] = set()
        for ename, enode in self.entity_nodes.items():
            if enode.total_weight >= min_entity_weight:
                shown_entities.add(ename)

        # Limiter
        if len(shown_entities) > max_entities:
            top = sorted(shown_entities, key=lambda e: self.entity_nodes[e].total_weight, reverse=True)
            shown_entities = set(top[:max_entities])

        # Nœuds épisodes
        ep_list = sorted(self.episode_nodes.keys())
        if len(ep_list) > max_episodes:
            # garder les plus gros
            ep_list = sorted(
                ep_list,
                key=lambda eid: self.episode_nodes[eid].artifact_count,
                reverse=True,
            )[:max_episodes]

        state_colors = {
            EpisodeState.ACTIVE: "#4CAF50",
            EpisodeState.DORMANT: "#FFC107",
            EpisodeState.CONSOLIDATED: "#2196F3",
            EpisodeState.ARCHIVED: "#9E9E9E",
        }

        for eid in ep_list:
            en = self.episode_nodes[eid]
            color = state_colors.get(en.state, "#EEEEEE")
            label = f"{eid}\\n{en.artifact_count} arts"
            lines.append(
                f'  "{eid}" [shape=box, style=filled, '
                f'fillcolor="{color}", label="{label}"];'
            )

        # Nœuds entités
        type_shapes = {
            "PERSON": "ellipse",
            "ORG": "diamond",
            "PLACE": "hexagon",
            "PROJECT": "octagon",
            "EVENT": "trapezium",
            "PRODUCT": "parallelogram",
            "DATE_EVENT": "invtrapezium",
            "DOCUMENT": "note",
        }

        for ename in shown_entities:
            en = self.entity_nodes[ename]
            shape = type_shapes.get(en.entity_type, "ellipse")
            safe_label = ename.replace('"', '\\"')
            lines.append(
                f'  "e_{ename}" [shape={shape}, label="{safe_label}\\n'
                f'({en.entity_type})", fontsize=8];'
            )

        # Arêtes
        for (eid, ename), weight in self._edges.items():
            if eid in ep_list and ename in shown_entities:
                pen = max(0.5, min(weight / 2.0, 5.0))
                lines.append(
                    f'  "{eid}" -- "e_{ename}" '
                    f'[penwidth={pen:.1f}, label="{weight:.0f}"];'
                )

        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, path: str, **kwargs) -> None:
        """Enregistre le graphe au format DOT."""
        dot = self.to_dot(**kwargs)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dot)
