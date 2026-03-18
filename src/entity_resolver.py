"""
entity_resolver.py — Résolution d'entités (Entity Resolution).

Fusionne les mentions brutes en entités canoniques :
  "Jean" + "Jean Dupont" + "jdupont@mail.com" → Entity(canonical="Jean Dupont")

Trois signaux de fusion :
  1. String similarity (Jaro-Winkler / token overlap)
  2. Embedding similarity (cosinus sur surface forms)
  3. Co-occurrence dans les mêmes artefacts

Architecture :
  EntityResolver
    ├── normalize() : pré-traitement des surface forms
    ├── resolve()   : clustering incrémental des mentions → entités
    └── EntityStore : index inversé mention→entité + entité→épisodes
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from entity_extractor import EntityType, Mention


# ------------------------------------------------------------------ #
# Modèle d'entité canonique                                           #
# ------------------------------------------------------------------ #

@dataclass
class CanonicalEntity:
    """Une entité résolue avec tous ses alias."""
    id: str                              # ex: "ENT_001"
    canonical_name: str                  # forme canonique choisie
    entity_type: EntityType              # type principal
    aliases: Set[str] = field(default_factory=set)    # toutes les formes vues
    mention_count: int = 0               # nb total de mentions
    artifact_ids: Set[str] = field(default_factory=set)  # artefacts où elle apparaît
    confidence: float = 1.0              # confiance moyenne
    embedding: Optional[np.ndarray] = None  # vecteur du nom canonique

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, CanonicalEntity) and self.id == other.id


# ------------------------------------------------------------------ #
# Fonctions de similarité                                             #
# ------------------------------------------------------------------ #

def _normalize(text: str) -> str:
    """Normalisation agressive d'une surface form."""
    t = text.lower().strip()
    # Retirer ponctuation sauf @, ., - (utiles pour emails/handles)
    t = re.sub(r'[^\w@.\-\s]', '', t)
    # Compacter les espaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _token_set(text: str) -> Set[str]:
    """Ensemble de tokens normalisés."""
    return set(_normalize(text).split())


def _string_similarity(a: str, b: str) -> float:
    """Similarité entre deux chaînes (combinaison ratio + token overlap)."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return 1.0
    # Ratio de SequenceMatcher (approximation Ratcliff-Obershelp)
    seq_ratio = SequenceMatcher(None, na, nb).ratio()
    # Token overlap (Jaccard)
    ta, tb = _token_set(a), _token_set(b)
    if ta and tb:
        jaccard = len(ta & tb) / len(ta | tb)
    else:
        jaccard = 0.0
    # Containment : une forme est contenue dans l'autre ?
    # Sécurité : la sous-chaîne doit être >= 4 chars et >= 40% de la chaîne longue
    containment = 0.0
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if shorter in longer and len(shorter) >= 4 and len(shorter) / max(len(longer), 1) >= 0.4:
        containment = 0.8
    return max(seq_ratio, jaccard, containment)


def _type_compatible(t1: EntityType, t2: EntityType) -> bool:
    """Vérifie si deux types d'entités sont compatibles pour fusion."""
    if t1 == t2:
        return True
    # UNKNOWN est compatible avec tout
    if t1 == EntityType.UNKNOWN or t2 == EntityType.UNKNOWN:
        return True
    # CONCEPT/PROJECT/EVENT/PRODUCT/DATE_EVENT/DOCUMENT : compatibles entre eux
    abstract_like = {
        EntityType.CONCEPT, EntityType.PROJECT, EntityType.EVENT,
        EntityType.PRODUCT, EntityType.DATE_EVENT, EntityType.DOCUMENT,
    }
    if t1 in abstract_like and t2 in abstract_like:
        return True
    # PERSON et EMAIL/HANDLE peuvent être la même personne
    person_like = {EntityType.PERSON, EntityType.EMAIL, EntityType.HANDLE}
    if t1 in person_like and t2 in person_like:
        return True
    return False


def _pick_canonical_name(aliases: Set[str], entity_type: EntityType) -> str:
    """Choisir le meilleur nom canonique parmi les alias."""
    if not aliases:
        return "UNKNOWN"
    # Préférer la forme la plus longue (souvent la plus informative)
    # sauf pour les emails/handles (préférer le nom si disponible)
    candidates = list(aliases)
    if entity_type in {
        EntityType.PERSON, EntityType.ORG, EntityType.PLACE,
        EntityType.PROJECT, EntityType.EVENT,
        EntityType.PRODUCT, EntityType.DATE_EVENT, EntityType.DOCUMENT,
    }:
        # Filtrer les emails et handles
        non_email = [c for c in candidates
                     if '@' not in c and not c.startswith('@')]
        if non_email:
            candidates = non_email
    # Préférer la forme la plus longue avec capitalisation
    def score(s):
        length = len(s)
        has_upper = any(c.isupper() for c in s)
        n_words = len(s.split())
        return (n_words * 10) + length + (5 if has_upper else 0)
    return max(candidates, key=score)


# ------------------------------------------------------------------ #
# EntityResolver — Clustering incrémental                             #
# ------------------------------------------------------------------ #

class EntityResolver:
    """
    Résout les mentions en entités canoniques par clustering incrémental.

    Algorithme :
    1. Pour chaque mention, normaliser la surface form
    2. Chercher une entité existante compatible (string sim > seuil)
    3. Si trouvée → fusionner (ajouter alias)
    4. Sinon → créer une nouvelle entité canonique
    5. Optionnel : deuxième passe avec embeddings pour fusions tardives
    """

    def __init__(self,
                 string_threshold: float = 0.75,
                 embedding_threshold: float = 0.85,
                 use_embeddings: bool = False,
                 min_mention_confidence: float = 0.0):
        """
        Parameters
        ----------
        string_threshold : seuil de similarité string pour fusion
        embedding_threshold : seuil de similarité embedding pour fusion
        use_embeddings : utiliser les embeddings pour fusion tardive
        min_mention_confidence : confiance minimum pour accepter une mention
                                 (0.0 = tout accepter, 0.5 = filtrer les ambigus)
        """
        self.string_threshold = string_threshold
        self.embedding_threshold = embedding_threshold
        self.use_embeddings = use_embeddings
        self.min_mention_confidence = min_mention_confidence

        # État interne
        self.entities: Dict[str, CanonicalEntity] = {}   # id → entity
        self._next_id = 0
        self._norm_index: Dict[str, str] = {}  # normalized_form → entity_id
        self._mention_to_entity: Dict[str, str] = {}  # surface_lower → entity_id

    def _gen_id(self) -> str:
        eid = f"ENT_{self._next_id:04d}"
        self._next_id += 1
        return eid

    def resolve(self, mentions: List[Mention],
                embed_fn=None) -> Dict[str, CanonicalEntity]:
        """
        Résoudre une liste de mentions en entités canoniques.

        Parameters
        ----------
        mentions : mentions brutes (sortie de EntityExtractor)
        embed_fn : fonction d'embedding (texts → ndarray), optionnel

        Returns
        -------
        Dict[entity_id, CanonicalEntity]
        """
        # Pré-filtrage par confiance linguistique
        if self.min_mention_confidence > 0:
            filtered = [m for m in mentions
                        if m.confidence >= self.min_mention_confidence]
        else:
            filtered = mentions

        # Passe 1 : clustering par string similarity
        for mention in filtered:
            self._resolve_single(mention)

        # Passe 2 : fusion tardive par embeddings
        if self.use_embeddings and embed_fn and len(self.entities) > 1:
            self._embedding_merge_pass(embed_fn)

        # Mettre à jour les noms canoniques
        for ent in self.entities.values():
            ent.canonical_name = _pick_canonical_name(ent.aliases, ent.entity_type)

        return dict(self.entities)

    def _resolve_single(self, mention: Mention):
        """Résoudre une seule mention."""
        norm = _normalize(mention.surface_form)

        # Lookup exact d'abord
        if norm in self._mention_to_entity:
            eid = self._mention_to_entity[norm]
            self._add_to_entity(eid, mention)
            return

        # Chercher la meilleure correspondance parmi les entités existantes
        best_eid = None
        best_sim = 0.0

        for eid, entity in self.entities.items():
            if not _type_compatible(mention.entity_type, entity.entity_type):
                continue
            # Comparer avec tous les alias de l'entité
            for alias in entity.aliases:
                sim = _string_similarity(mention.surface_form, alias)
                if sim > best_sim:
                    best_sim = sim
                    best_eid = eid

        if best_sim >= self.string_threshold and best_eid is not None:
            self._add_to_entity(best_eid, mention)
        else:
            self._create_entity(mention)

    def _create_entity(self, mention: Mention):
        """Créer une nouvelle entité canonique."""
        eid = self._gen_id()
        entity = CanonicalEntity(
            id=eid,
            canonical_name=mention.surface_form,
            entity_type=mention.entity_type,
            aliases={mention.surface_form},
            mention_count=1,
            artifact_ids={mention.source_artifact_id},
            confidence=mention.confidence,
        )
        self.entities[eid] = entity
        norm = _normalize(mention.surface_form)
        self._norm_index[norm] = eid
        self._mention_to_entity[norm] = eid

    def _add_to_entity(self, eid: str, mention: Mention):
        """Ajouter une mention à une entité existante."""
        entity = self.entities[eid]
        entity.aliases.add(mention.surface_form)
        entity.mention_count += 1
        entity.artifact_ids.add(mention.source_artifact_id)
        # Moyenne glissante de la confiance
        n = entity.mention_count
        entity.confidence = ((n - 1) * entity.confidence + mention.confidence) / n
        # Promouvoir le type si la mention a un type plus précis
        if entity.entity_type in {EntityType.UNKNOWN, EntityType.CONCEPT}:
            if mention.entity_type not in {EntityType.UNKNOWN, EntityType.CONCEPT}:
                entity.entity_type = mention.entity_type
        # Indexer la nouvelle forme
        norm = _normalize(mention.surface_form)
        self._mention_to_entity[norm] = eid

    def _embedding_merge_pass(self, embed_fn):
        """Deuxième passe : fusionner par similarité d'embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity

        entity_list = list(self.entities.values())
        if len(entity_list) < 2:
            return

        # Embedder les noms canoniques
        names = [e.canonical_name for e in entity_list]
        embeddings = embed_fn(names)
        if embeddings is None or len(embeddings) == 0:
            return

        for e, emb in zip(entity_list, embeddings):
            e.embedding = emb

        # Matrice de similarité
        sim_matrix = cosine_similarity(embeddings)

        # Union-Find pour les fusions
        parent = list(range(len(entity_list)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merge_count = 0
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                ei, ej = entity_list[i], entity_list[j]
                if not _type_compatible(ei.entity_type, ej.entity_type):
                    continue
                if sim_matrix[i][j] >= self.embedding_threshold:
                    # Vérifier aussi string sim minimale pour éviter faux positifs
                    str_sim = max(
                        _string_similarity(a1, a2)
                        for a1 in ei.aliases for a2 in ej.aliases
                    )
                    if str_sim >= 0.45:  # seuil de sécurité string
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
                            merge_count += 1

        if merge_count == 0:
            return

        # Regrouper et fusionner
        groups = defaultdict(list)
        for idx in range(len(entity_list)):
            groups[find(idx)].append(idx)

        new_entities = {}
        for root, members in groups.items():
            if len(members) == 1:
                ent = entity_list[members[0]]
                new_entities[ent.id] = ent
                continue

            # Fusionner les entités du groupe
            member_ents = [entity_list[m] for m in members]
            base = max(member_ents, key=lambda e: e.mention_count)
            for other in member_ents:
                if other.id == base.id:
                    continue
                base.aliases |= other.aliases
                base.artifact_ids |= other.artifact_ids
                base.mention_count += other.mention_count
                # Mettre à jour les index
                for alias in other.aliases:
                    self._mention_to_entity[_normalize(alias)] = base.id
                self._norm_index[_normalize(other.canonical_name)] = base.id

            base.canonical_name = _pick_canonical_name(base.aliases, base.entity_type)
            new_entities[base.id] = base

        self.entities = new_entities
        self._entity_merge_count = merge_count

    # ------------------------------------------------------------------ #
    # API de lookup                                                       #
    # ------------------------------------------------------------------ #

    def lookup(self, surface_form: str) -> Optional[CanonicalEntity]:
        """Retrouver l'entité canonique d'une mention."""
        norm = _normalize(surface_form)
        eid = self._mention_to_entity.get(norm)
        if eid:
            return self.entities.get(eid)
        return None

    def get_entities_for_artifact(self, artifact_id: str) -> List[CanonicalEntity]:
        """Retrouver toutes les entités d'un artefact."""
        return [e for e in self.entities.values() if artifact_id in e.artifact_ids]

    def get_entity_names(self) -> List[str]:
        """Liste des noms canoniques."""
        return [e.canonical_name for e in self.entities.values()]

    def build_artifact_entity_map(self) -> Dict[str, List[str]]:
        """Construire la map artifact_id → [canonical_names]."""
        result = defaultdict(list)
        for entity in self.entities.values():
            for aid in entity.artifact_ids:
                result[aid].append(entity.canonical_name)
        return dict(result)

    def summary(self) -> str:
        """Résumé textuel de la résolution."""
        by_type = defaultdict(int)
        for e in self.entities.values():
            by_type[e.entity_type.value] += 1

        lines = [
            f"EntityResolver: {len(self.entities)} entités canoniques",
            f"  Mentions indexées: {len(self._mention_to_entity)}",
        ]
        for t, n in sorted(by_type.items()):
            lines.append(f"  {t}: {n}")

        # Top-10 par fréquence
        top = sorted(self.entities.values(),
                     key=lambda e: e.mention_count, reverse=True)[:10]
        lines.append("\n  Top-10 entités:")
        for e in top:
            aliases_str = ', '.join(sorted(e.aliases)[:5])
            lines.append(
                f"    {e.canonical_name} ({e.entity_type.value}, "
                f"{e.mention_count}x, {len(e.artifact_ids)} artefacts) "
                f"[{aliases_str}]"
            )
        return '\n'.join(lines)


# ------------------------------------------------------------------ #
# Fonction utilitaire : enrichir les artefacts                        #
# ------------------------------------------------------------------ #

def enrich_artifacts_with_entities(
    artifacts: List,
    resolver: EntityResolver
) -> None:
    """
    Met à jour artifact.entities avec les noms canoniques résolus.
    Modifie les artefacts in-place.
    """
    art_map = resolver.build_artifact_entity_map()
    for artifact in artifacts:
        canonical = art_map.get(artifact.id, [])
        if canonical:
            artifact.entities = canonical
