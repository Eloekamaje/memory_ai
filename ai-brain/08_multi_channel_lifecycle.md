# 08 — AI Memory Personnelle : Multi-Canal & Cycle de Vie

> Vision architecturale pour une mémoire personnelle qui s'alimente en continu depuis plusieurs sources,
> avec ses propres états de vie — comme une mémoire biologique.

---

## Le shift fondamental

Jusqu'ici : `build_pipeline(file.txt)` → analyse batch → résultat statique.

La cible : **une mémoire vivante** qui s'alimente en permanence, qui grandit, qui oublie, qui se réveille.

```
Avant : outil d'analyse
Après  : infrastructure cognitive personnelle
```

---

## Architecture multi-canal

### Principe

Chaque canal est un flux d'artefacts. Tous les flux convergent vers le même Memory Engine.

```
WhatsApp ──────────────┐
Email (Gmail/MBOX) ────┤
Notes Obsidian ────────┤──→ [Channel Router] ──→ [Memory Engine]
Calendrier (ICS) ──────┤                              ↕
Documents (PDF/MD) ────┤                        [Decision Engine]
Terminal history ───────┘                              ↕
                                                  [Agent]
```

### Canal = Parser + Adapter + Poids

Chaque canal a trois propriétés :

| Propriété | Rôle |
|-----------|------|
| **Parser** | Convertit le format brut en `List[Artifact]` |
| **Adapter** | Normalise les métadonnées (timestamp, auteur, type) |
| **Trust weight** | Fiabilité du signal (calendrier > WhatsApp > terminal) |

```python
@dataclass
class Channel:
    name: str            # "whatsapp", "gmail", "obsidian"
    parser: Callable
    poll_interval: int   # secondes (0 = event-driven)
    trust_weight: float  # 0.5 à 1.0
    source_path: str
```

### Canaux et leur nature

| Canal | Type | Signal | Trust |
|-------|------|--------|-------|
| WhatsApp | Conversationnel | Topics, entités, relations | 0.7 |
| Gmail | Transactionnel | Projets, décisions, deadlines | 0.8 |
| Obsidian/Notes | Réflexif | Idées, synthèses, intentions | 0.9 |
| Calendrier (ICS) | Structuré | Événements, personnes, lieux | 1.0 |
| Documents (PDF) | Référentiel | Contexte stable, faits | 0.85 |
| Terminal history | Comportemental | Contexte technique, projets actifs | 0.6 |
| Voice transcription | Conversationnel | Conversations orales | 0.7 |

### Linking inter-canal

La même entité apparaît sur plusieurs canaux → même épisode.

```
WhatsApp  : "réunion GEDDVIT budget demain"    → entité [GEDDVIT, budget]
Calendrier: "Réunion GEDDVIT 14h" (le lendemain) → entité [GEDDVIT]
Notes     : "décision budget Q3 GEDDVIT"          → entité [GEDDVIT, budget]

→ Trois canaux, un seul épisode "Budget GEDDVIT Q3"
```

Le cross-channel linking enrichit l'épisode sans le dupliquer.

---

## Ingestion continue

### Modes d'ingestion

```
Event-driven   inotify (fichiers), webhook (API)    → temps réel, < 1s
Polling        intervalle configurable par canal     → 1 min à 1h
Batch          manuel ou scheduled (nuit)            → pour les gros imports
```

### Pipeline incrémental (cible)

Aujourd'hui `build_pipeline()` reconstruit tout. La cible :

```
Nouvel artefact entrant
  ↓
ChannelRouter.route(artifact)          # quel canal ? quel poids ?
  ↓
EpisodeSegmenter.update(artifact)      # attacher ou créer
  ↓
EntityExtractor.update(artifact)       # entités du nouvel artefact
  ↓
MemoryGraph.update(episode, entities)  # diff du graphe
  ↓
MemoryIndex.update(episode_centroid)   # ajout partiel dans l'index
  ↓
MemoryStore.append(artifact)           # append, pas rebuild
  ↓
DecisionEngine.evaluate(memory_state)  # action à déclencher ?
```

Aucun rebuild. Chaque artefact déclenche une mise à jour partielle.

---

## Cycle de vie de la mémoire

### Les états d'un épisode

```
                    ┌─────────────┐
                    │    BORN     │  Premier artefact
                    └──────┬──────┘
                           │ nouveaux artefacts
                           ▼
                    ┌─────────────┐
                    │   ACTIVE    │  En cours, importance croissante
                    └──────┬──────┘
                           │ inactivité > dormancy_threshold
                           ▼
                    ┌─────────────┐
                    │   DORMANT   │  En veille, accessible
                    └──────┬──────┘
                           │ nouveau lien détecté
                           ▼
              ┌────────────┴────────────┐
              │                         │
    (inactivité prolongée)     (réactivation)
              │                         │
              ▼                         ▼
       ┌─────────────┐          ┌─────────────┐
       │ COMPRESSED  │          │   ACTIVE    │  Résurrection
       └──────┬──────┘          └─────────────┘
              │
              │ très vieux + peu accédé
              ▼
       ┌─────────────┐
       │  ARCHIVED   │  Résumé uniquement, index retiré
       └──────┬──────┘
              │
        ┌─────┴──────┐
        │             │
   (nouveau lien)  (jamais accédé)
        │             │
        ▼             ▼
   ┌─────────┐  ┌──────────┐
   │ REVIVED │  │  DELETED │
   └─────────┘  └──────────┘
```

### Description des états

| État | Description | Accessible | Dans l'index |
|------|-------------|-----------|--------------|
| **BORN** | Épisode créé, 1 artefact | Oui | Oui |
| **ACTIVE** | Artefacts qui s'accumulent, centroïde évolue | Oui | Oui |
| **DORMANT** | Inactif, centroïde figé, pénalité d'âge active | Oui | Oui |
| **COMPRESSED** | Résumé remplace le contenu détaillé | Résumé uniquement | Oui (résumé) |
| **ARCHIVED** | Cold storage, retiré de l'index actif | Sur requête explicite | Non |
| **REVIVED** | Réactivé par un nouveau lien cross-canal ou requête | Oui | Oui |
| **DELETED** | Purgé (explicite ou politique de rétention) | Non | Non |

### Transitions — qui les déclenche ?

| Transition | Déclencheur | Responsable |
|------------|-------------|-------------|
| BORN → ACTIVE | Nouveaux artefacts | EpisodeSegmenter |
| ACTIVE → DORMANT | Inactivité > seuil | EpisodeSegmenter (aging) |
| DORMANT → ACTIVE | Nouvel artefact lié | EpisodeSegmenter (reactivation) |
| DORMANT → COMPRESSED | Âge + importance faible | **Decision Engine** |
| COMPRESSED → ARCHIVED | Âge + accès nul | **Decision Engine** |
| ARCHIVED → REVIVED | Nouveau lien ou recall explicite | **Decision Engine** |
| ANY → DELETED | Politique de rétention | **Decision Engine** (+ user confirm) |

Le Decision Engine gère les transitions non-triviales. C'est son rôle principal.

---

## Vieillissement de la mémoire

La mémoire ne disparaît pas d'un coup. Elle se dégrade par couches, de façon contrôlée, en fonction de l'usage et du temps. C'est le principe de la mémoire biologique — et c'est ce qu'on modélise ici.

---

### La courbe d'oubli — Ebbinghaus adapté

La formule de base d'Ebbinghaus :

```
R(t) = e^(-t / S)

  R = rétention (0 → 1)
  t = temps écoulé depuis le dernier accès
  S = stabilité (fonction de l'importance et des accès passés)
```

Notre adaptation pour les épisodes :

```
retention(e, t) = exp( -λ(e) * t / (1 + log1p(access_count(e))) )

  λ(e)          = taux de décroissance propre à l'épisode
  access_count  = nombre de fois l'épisode a été rappelé
  t             = jours depuis le dernier accès
```

Effet : chaque accès (recall) **aplatit la courbe**. Un épisode consulté 5 fois décroît 3x plus lentement qu'un épisode jamais touché.

```
Rétention
  1.0 │▓▓▓▓▓
      │     ▓▓▓▓                   ← épisode accédé fréquemment (λ faible)
  0.7 │         ▓▓▓▓
      │              ▓▓▓
  0.4 │▓▓▓▓               ▓▓▓
      │    ▓▓▓▓                ▓▓  ← épisode jamais accédé (λ élevé)
  0.1 │        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
      └──────────────────────────→ temps (jours)
         0    7   14   30   60  90
```

---

### Facteurs qui modulent λ (taux de décroissance)

```python
def decay_rate(episode) -> float:
    λ_base = 0.05  # décroissance de base (50% en ~14 jours sans accès)

    # Importance réduit λ — les épisodes importants tiennent plus longtemps
    λ = λ_base / (1 + episode.importance)

    # Canal : certains canaux ont une durée de vie naturellement plus longue
    λ *= CHANNEL_DECAY_FACTOR[episode.primary_channel]

    # Richesse en entités — un épisode dense en entités décroît moins vite
    λ /= (1 + log1p(len(episode.entities)))

    # Épisode non résolu — reste saillant jusqu'à résolution
    if episode.is_open:
        λ *= 0.3  # décroissance très lente pour les épisodes ouverts

    return λ
```

**Facteurs de décroissance par canal :**

| Canal | `CHANNEL_DECAY_FACTOR` | Raison |
|-------|----------------------|--------|
| Calendrier | 0.3 | Événements = faits datés, longue durée |
| Notes/Obsidian | 0.4 | Réflexions intentionnelles, denses |
| Gmail | 0.6 | Transactions, durée variable |
| WhatsApp | 0.8 | Conversationnel, volatile |
| Terminal | 1.0 | Très volatile, technique |

---

### Vieillissement par couches (du détail au résumé)

La mémoire ne supprime pas — elle **comprime**. Chaque couche de compression préserve l'essentiel en effaçant le bruit.

```
COUCHE 1 — Contenu complet
  Tous les artefacts bruts, tous les messages.
  Durée : jusqu'à DORMANT prolongé (ex. 30 jours sans accès)

COUCHE 2 — Contenu partiel + résumé
  Les artefacts les plus importants (top 20%) + résumé extractif.
  Durée : COMPRESSED (jusqu'à 180 jours)

COUCHE 3 — Résumé uniquement
  Résumé LLM (3-5 phrases). Entités, label, importance. Plus d'artefacts.
  Durée : ARCHIVED (jusqu'à 365 jours ou plus)

COUCHE 4 — Trace mémorielle
  Juste le label, les entités principales, la date, le score d'importance.
  Durée : permanente (sauf DELETED)
  → Permet de "savoir qu'on a eu cette conversation" sans en retrouver le contenu.
```

```
Mémoire complète   → Mémoire partielle → Résumé → Trace → Oubli total
[████████████████]    [████████░░░░░░░]   [███░░]   [█░]    [ ]
     ACTIVE/DORMANT        COMPRESSED      ARCHIVED  TRACE   DELETED
```

---

### Consolidation nocturne

Inspiré de la consolidation mémorielle pendant le sommeil. Un batch job tourne quand le système est inactif :

```
23h00 — Consolidation batch
  1. Scanner tous les épisodes DORMANT > 30 jours
  2. Calculer retention(e, t) pour chacun
  3. Si retention < 0.3 → COMPRESS
  4. Scanner COMPRESSED > 90 jours, retention < 0.2 → ARCHIVE
  5. Détecter les liens latents entre épisodes (entités partagées non encore liées)
  6. Mettre à jour les scores d'importance
  7. Recalculer les centroïdes des épisodes ACTIVE (drift correction)
  8. Log de la consolidation : N compressés, N archivés, N liens créés
```

```python
class MemoryConsolidator:
    """Processus de consolidation — à exécuter en background (nuit ou idle)."""

    def run(self, episodes, store, graph) -> ConsolidationReport:
        compressed = self._compress_aged(episodes, store)
        archived   = self._archive_compressed(episodes, store)
        linked     = self._link_latent(episodes, graph)
        return ConsolidationReport(compressed, archived, linked)
```

---

### Résurrection et renforcement

Quand un épisode ARCHIVED est rappelé ou reçoit un nouveau lien :

```
1. Épisode extrait du cold storage
2. Résumé réintégré dans l'index actif
3. Importance recalculée à la hausse (+boost de résurrection)
4. access_count incrémenté → λ réduit
5. État : ARCHIVED → REVIVED → ACTIVE
```

Le boost de résurrection :
```python
importance_new = min(1.0, importance_old * 1.4 + REVIVAL_BONUS)
```

Un épisode ressuscité est traité comme plus important que lors de sa première vie — le fait qu'il soit revenu dans le contexte signale sa pertinence.

---

### Épisodes ouverts vs résolus

Un épisode "ouvert" (question sans réponse, projet en cours, décision en suspens) ne devrait pas vieillir normalement.

```python
def is_open(episode) -> bool:
    """Heuristique : l'épisode contient des signaux de question ouverte."""
    signals = [
        any(kw in episode.label.lower() for kw in ["?", "à faire", "en cours", "prévu"]),
        episode.last_entity_type in ["PROJET", "DECISION", "DEADLINE"],
        episode.resolution_score < 0.3,   # pas de signal de clôture détecté
    ]
    return any(signals)
```

Les épisodes ouverts :
- Décroissance 3x plus lente (λ × 0.3)
- Flagués comme `FLAG` par le Decision Engine → rappel proactif
- Threshold de compression porté à 180 jours (vs 30)

---

## Importance et scoring

### Score d'importance d'un épisode

```python
importance(e) = w1 * entity_richness(e)    # nombre d'entités uniques
              + w2 * access_frequency(e)   # fréquence d'accès en recall
              + w3 * recency(e)            # fraîcheur temporelle
              + w4 * cross_channel_score(e) # apparaît sur N canaux
              + w5 * resolution_score(e)   # épisode résolu ou ouvert ?
```

### Courbe d'oubli (Ebbinghaus adapté)

```
importance_effective(t) = importance(e) * exp(-λ * t / (1 + access_count))
```

Plus un épisode est accédé, plus sa courbe d'oubli est plate. Un épisode jamais consulté décroît vite.

### Nouveauté

```python
novelty(artifact, memory) = 1 - max_similarity(artifact, all_episodes)
```

Un artefact très similaire à ce qui existe déjà a une nouveauté faible → moins d'impact sur l'importance.

---

## Le Decision Engine en détail

C'est le chef d'orchestre du cycle de vie.

```python
class DecisionEngine:

    def evaluate(self, episodes, new_artifact=None) -> List[Action]:
        actions = []

        for ep in episodes:
            # Transition DORMANT → COMPRESSED
            if ep.state == DORMANT and self._should_compress(ep):
                actions.append(Action(COMPRESS, ep))

            # Transition COMPRESSED → ARCHIVED
            elif ep.state == COMPRESSED and self._should_archive(ep):
                actions.append(Action(ARCHIVE, ep))

            # Trigger ACTIVATE (proactif vers l'agent)
            elif ep.state in [ACTIVE, DORMANT] and self._should_surface(ep, new_artifact):
                actions.append(Action(ACTIVATE, ep))

            # Linking cross-épisodes
            if new_artifact and self._should_link(ep, new_artifact):
                actions.append(Action(LINK, ep, new_artifact))

        return actions

    def _should_compress(self, ep) -> bool:
        age_days = (now - ep.t_end).days
        return age_days > 30 and ep.access_count < 3

    def _should_surface(self, ep, new_artifact) -> bool:
        # Proactivité : cet épisode est pertinent pour ce nouvel artefact
        if new_artifact is None:
            return False
        similarity = cosine(new_artifact.embedding, ep.centroid)
        return similarity > 0.75 and ep.importance > 0.6
```

---

## Mémoire questionnable

Une mémoire passive qui stocke sans répondre n'est pas une mémoire — c'est une archive. La mémoire doit être **interrogeable en langage naturel**, à plusieurs niveaux, y compris sur elle-même.

---

### Les 5 types de questions

#### 1. Questions factuelles — *Qu'est-ce que je sais ?*

```
"Qu'est-ce qui s'est passé avec GEDDVIT en mars ?"
"Qu'est-ce qu'on a décidé sur le budget AWS ?"
"C'était quoi la dernière fois qu'on a parlé de Jean ?"
```

→ Recall sémantique + filtrage temporel + entités. Le moteur actuel gère déjà ça.

#### 2. Questions temporelles — *Quand ? Depuis quand ? Combien de temps ?*

```
"Depuis quand je parle de lancer une app ?"
"Il s'est passé quoi la semaine du 10 mars ?"
"Combien de temps j'ai passé sur le projet GEDDVIT ?"
```

→ Nécessite un index temporel + timeline reconstruction.

#### 3. Questions relationnelles — *Qu'est-ce qui relie X et Y ?*

```
"Qu'est-ce qui relie AWS et le projet GEDDVIT ?"
"Quelles personnes reviennent dans mes épisodes importants ?"
"Est-ce que Jean et le budget apparaissent souvent ensemble ?"
```

→ Traversée du MemoryGraph. Chercher les co-occurrences d'entités.

#### 4. Questions méta — *Qu'est-ce que TU sais sur ce que JE sais ?*

```
"Quels sont mes sujets récurrents ?"
"Quels épisodes sont encore ouverts ?"
"Qu'est-ce que tu as dans ta mémoire sur moi ?"
"Qu'est-ce que tu as oublié récemment ?"
"Quels sont mes patterns de la semaine ?"
```

→ La mémoire s'interroge elle-même. Introspection du MemoryGraph + stats lifecycle.

#### 5. Questions prospectives — *Qu'est-ce que je devrais faire ?*

```
"Y a-t-il des choses en suspens que j'ai oubliées ?"
"Qu'est-ce que j'aurais dû faire suite à la réunion du 14 mars ?"
"Est-ce que j'ai des décisions non finalisées ?"
```

→ Épisodes `is_open=True` + Decision Engine + LLM reasoning.

---

### Architecture du Query Engine

```
Question NL
    ↓
[Query Parser]          → type de question (factuelle / temporelle / relationnelle / méta)
    ↓
[Query Planner]         → décompose en sous-requêtes si nécessaire
    ↓
┌───────────────────────────────────────────────────────┐
│  RecallEngine    MemoryGraph     MemoryStore    Stats  │
│  (sémantique)    (relations)     (lifecycle)   (méta)  │
└───────────────────────────────────────────────────────┘
    ↓
[Answer Synthesizer]    → combine les résultats, explique les sources
    ↓
Réponse NL + trace de raisonnement
```

```python
class QueryEngine:
    """Interface unifiée : question NL → réponse structurée."""

    def ask(self, question: str) -> QueryResponse:
        q_type  = self.parser.classify(question)     # factuelle / temporelle / méta / ...
        plan    = self.planner.plan(question, q_type) # sous-requêtes
        results = self._execute(plan)                 # dispatch vers les bons moteurs
        answer  = self.synthesizer.synthesize(question, results)
        return QueryResponse(answer=answer, sources=results, confidence=answer.score)
```

---

### Introspection — la mémoire se connaît elle-même

Questions méta : la mémoire doit pouvoir répondre sur son propre état.

```python
class MemoryIntrospector:

    def state_summary(self) -> str:
        """Qu'est-ce que tu sais sur moi en ce moment ?"""
        return f"""
        Je connais {n_episodes} épisodes couvrant {date_range}.
        Sujets récurrents : {top_entities[:5]}
        Épisodes ouverts  : {open_episodes} (non résolus)
        Épisodes récents  : {active_last_7d} (7 derniers jours)
        Épisodes archivés : {archived} (résumés disponibles)
        Dernière consolidation : {last_consolidation}
        """

    def forgotten_summary(self) -> str:
        """Qu'est-ce que tu as compressé ou archivé récemment ?"""
        # Lister les épisodes COMPRESSED/ARCHIVED des 30 derniers jours
        # avec leurs labels et dates — "je sais que ça a existé"

    def open_threads(self) -> List[Episode]:
        """Quels fils sont encore ouverts ?"""
        return [e for e in episodes if e.is_open and e.state != ARCHIVED]

    def recurring_patterns(self) -> Dict:
        """Quels sujets reviennent ? À quelle fréquence ?"""
        # Analyse de fréquence sur les entités + labels sur 90 jours
```

---

### Format de réponse — toujours explicable

La réponse n'est jamais une boîte noire. Elle expose ses sources.

```
Question : "Qu'est-ce qui s'est passé avec AWS en mars ?"

Réponse  : "En mars, deux épisodes concernent AWS :

  1. Budget AWS infrastructure (14 mars, importance 0.87)
     Canal : WhatsApp + Notes
     Résumé : Discussion sur les coûts EC2, décision de passer en spot instances.
     Statut : RÉSOLU

  2. Galère déploiement staging (21 mars, importance 0.72)
     Canal : WhatsApp
     Résumé : Timeout sur le load balancer, non résolu à la date de l'épisode.
     Statut : OUVERT — aucun suivi détecté depuis.

Sources : épisodes #12, #18. Confiance : 0.91."
```

**Règle :** toute réponse inclut les épisodes sources, leur état de vie, et la confiance du recall.

---

### Modes d'interrogation

| Mode | Interface | Cas d'usage |
|------|-----------|-------------|
| **Chat NL** | `python cli.py chat` | Questions libres, conversation |
| **Recall direct** | `engine.recall("query")` | Requête programmatique |
| **Timeline** | `memory.timeline(from, to)` | Vue chronologique |
| **Graph explore** | `graph.neighbors("GEDDVIT")` | Navigation relationnelle |
| **Introspect** | `memory.status()` | État de la mémoire |
| **Open threads** | `memory.open_threads()` | Ce qui est en suspens |

---

### Exemples bout en bout

```
# Factuelle
> "C'était quoi la réunion avec Jean ?"
→ RecallEngine("réunion Jean") → épisodes #7, #14 → synthèse LLM

# Temporelle
> "Qu'est-ce qui s'est passé la semaine du 10 mars ?"
→ filter(t_start >= 10/03, t_end <= 16/03) → timeline reconstruction

# Relationnelle
> "Qu'est-ce qui relie GEDDVIT et AWS ?"
→ graph.path("GEDDVIT", "AWS") → épisodes communs → entités co-occurrentes

# Méta
> "Qu'est-ce que tu sais sur moi ?"
→ MemoryIntrospector.state_summary() → top entités, sujets ouverts, stats

# Prospective
> "Y a-t-il des choses en suspens ?"
→ open_threads() → DecisionEngine FLAG → liste priorisée par importance
```

---

## Gestion de la vie privée

Tout tourne en local. Aucune donnée ne sort de la machine sans consentement explicite.

```
Données brutes      → stockées localement (MemoryStore)
LLM pour résumés    → Ollama + Mistral 7B (local) par défaut
                    → API Claude optionnelle (qualité supérieure)
Index               → local (sklearn / FAISS)
Graphe              → local (NetworkX + SQLite)
```

**Politique de rétention configurable par canal :**
```yaml
channels:
  whatsapp:
    compress_after_days: 30
    archive_after_days:  180
    delete_after_days:   null  # jamais (défaut)
  calendar:
    compress_after_days: null  # jamais compresser les événements
    archive_after_days:  365
```

---

## Roadmap d'implémentation

### Sprint 1 — Fondation multi-canal (CPU, 1 semaine)

```
src/
├── channel.py              # Channel dataclass + ChannelRouter
├── parsers/
│   ├── whatsapp_parser.py  # ✅ existe
│   ├── gmail_parser.py     # MBOX → List[Artifact]
│   ├── obsidian_parser.py  # Markdown → List[Artifact]
│   └── ics_parser.py       # ICS → List[Artifact] (calendar)
└── decision_engine.py      # ✅ à créer (H0)
```

### Sprint 2 — Pipeline incrémental (CPU, 1 semaine)

```
memory_pipeline.py
  + ingest(artifact)         # update partiel, pas rebuild
  + watch(channel)           # inotify sur un répertoire
```

### Sprint 3 — Cycle de vie complet (CPU, 1 semaine)

```
models.py
  EpisodeState + COMPRESSED, ARCHIVED, REVIVED, DELETED

decision_engine.py
  + _should_compress()
  + _should_archive()
  + _should_revive()
  + _should_surface()

memory_store.py
  + compress(episode)        # LLM local → résumé
  + archive(episode)         # cold storage
  + revive(episode)          # index update
```

### Sprint 4 — Proactivité (CPU + LLM local, 1 semaine)

```
Agent proactif :
  À chaque ingest → DecisionEngine.evaluate()
  Si ACTIVATE → MemoryAgent génère suggestion
  Sortie : notification / réponse CLI / webhook
```

---

## Ce que ça donne concrètement

```
08h12  Tu envoies un WhatsApp : "je galère avec les coûts AWS"
       ↓
       ChannelRouter : canal whatsapp, trust=0.7
       ↓
       EpisodeSegmenter : lien avec épisode "Lancer une app" (14 mars)
       ↓
       MemoryGraph : entités [AWS, coûts] → épisode mis à jour
       ↓
       DecisionEngine : importance élevée, activité récente → ACTIVATE
       ↓
       MemoryAgent : "Il y a 4 jours tu voulais lancer une app.
                      Le coût AWS revient. Tu veux qu'on structure
                      une architecture low-cost ?"

08h12  La mémoire t'a parlé. Sans que tu lui demandes.
```

---

> Une AI memory personnelle n'est pas un outil qu'on consulte.
> C'est un système qui vit avec toi, se nourrit de tes flux,
> et te parle quand c'est pertinent.
