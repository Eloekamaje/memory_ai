# 🧠 Memory AI Lab

> **Segmentation épisodique multi-critères d'un flux d'artefacts personnels**

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![Research](https://img.shields.io/badge/Type-Research-purple)
![Status](https://img.shields.io/badge/Segmentation-V3-green)

---

## Le problème

Dans la vie réelle, les informations arrivent comme un flux : messages, emails, documents, réunions, notes.
La mémoire humaine ne les stocke pas séparément — elle les regroupe en **épisodes**.

> 4 objets pour l'ordinateur. 1 événement pour l'humain.

**Question algorithmique centrale** : comment détecter automatiquement qu'un ensemble d'artefacts appartient au même épisode ?

C'est un problème de **clustering dynamique multi-critères dans un flux temporel** — une intersection entre structures de données, NLP, bases de données et cognition.

---

## Modèle formel

Un artefact $a_i$ possède un temps $t_i$, un type $y_i$, des entités $E_i$, un vecteur sémantique $v_i$ et un vecteur d'objectif $g_i$.

L'**AttachScore** détermine si un artefact rejoint un épisode existant :

$$\text{AttachScore}(a, e) = \alpha \cdot \text{Sem}(a,e) + \beta \cdot \text{Ent}(a,e) + \gamma \cdot \text{Temp}(a,e) + \delta \cdot \text{Goal}(a,e) - \rho \cdot \text{AgePenalty}(e)$$

| Composante | Rôle |
|-----------|------|
| **Sem** | Similarité cosinus artefact ↔ centroïde épisode |
| **Ent** | Recouvrement Jaccard des entités |
| **Temp** | Proximité temporelle (décroissance exponentielle) |
| **Goal** | Similarité d'objectif/intention latente |
| **AgePenalty** | Pénalité pour épisodes dormants (log1p) |

---

## Architecture

```
memory_ai_lab/
├── src/
│   ├── models.py                # Artifact, Episode, EpisodeState
│   ├── episode_algorithm.py     # EpisodeSegmenter V2+ (553 lignes)
│   ├── goal_heuristics.py       # Inférence d'objectifs latents
│   ├── embedding_engine.py      # SentenceTransformer all-MiniLM-L6-v2
│   ├── metrics.py               # ARI, NMI, cohérence, fragmentation
│   ├── experiment_runner.py     # Pipeline + ablation study
│   ├── dataset_generator.py     # Génération synthétique (5 thèmes)
│   ├── dataset_loader.py        # Façade CSV / WhatsApp
│   └── parsers/
│       ├── experiment_csv_parser.py
│       └── whatsapp_parser.py
├── data/
│   ├── synthetic_200.csv        # 127 artefacts, 19 épisodes
│   ├── synthetic_500.csv        # 142 artefacts, 20 épisodes
│   └── real-converstions.txt    # WhatsApp réel (2879 messages)
├── notebooks/
│   └── analysis.ipynb           # 7 sections, visualisations complètes
├── requirements.txt
└── README.md
```

### Pipeline

```
Artefacts bruts
     │
     ▼
  Parser (CSV / WhatsApp)
     │
     ▼
  Embeddings (all-MiniLM-L6-v2, 384d)
     │
  Goal Heuristics (type→intent + regex→action + entities)
     │
     ▼
  EpisodeSegmenter V2+
     │  ├─ AttachScore 4 termes
     │  ├─ Aging (ACTIVE → DORMANT)
     │  ├─ Réactivation (DORMANT → ACTIVE)
     │  └─ Candidats : fenêtre récente + entités partagées
     │
     ▼
  Consolidation (fusion Union-Find + contrainte temporelle)
     │
     ▼
  Épisodes ──→ Métriques (ARI, NMI, cohérence, fragmentation)
```

---

## Résultats acquis

### Performance V1 → V2 → V3 (synthetic_200)

| Version | ARI | Notes |
|---------|:---:|-------|
| **V1** baseline | 0.31 | Segmentation simple, sans entités |
| **V2** entity-boosted | 0.506 | α=0.45, β=0.25, NER spaCy |
| **V3** EMA α=0.80 | **0.5179** | EMA centroid + entity overlap normalization + AgePenalty |
| **V3** sur synthetic_500 | 0.3243 | −37% vs synthetic_200 → surapprentissage confirmé |

> **Note méthodologique** : ARI=0.5179 a été calibré sur le même dataset que l'évaluation.
> Le score sur synthetic_500 (distribution différente) montre que les paramètres ne généralisent pas encore.
> → Plan d'action V4 ci-dessus adresse ce point via SuperDialseg + hold-out.

### Paramètres V3

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| α (Sem) | 0.45 | Poids similarité sémantique |
| β (Ent) | 0.25 | Poids entités (entity-boosted) |
| γ (Temp) | 0.10 | Poids temporel |
| δ (Goal) | 0.20 | Poids objectif |
| ρ (Age) | 0.05 | Pénalité dormance |
| ema_alpha | 0.80 | Poids EMA centroid (calibré sur synthetic_200) |
| active_penalty_hours | 24h | Seuil AgePenalty sur ACTIVE |
| hard_break_minutes | 720 | Sessions 12h (données WhatsApp uniquement) |
| seuil | 0.30 | Seuil d'attachement |
| dormancy | 1440 min | Inactivité → DORMANT |

### Hypothèses validées

| Hypothèse | Statut | Preuve |
|-----------|--------|--------|
| **H1** : Goal améliore la segmentation | ✅ Partiel | Neutre seul, mais contribue en synergie avec aging |
| **H2** : L'aging est un levier décisif | ✅ **Fort** | +93% ARI (game-changer) |
| **H3** : La réactivation récupère les épisodes dormants | ✅ Validé | +10% ARI, 6 réactivations |
| **H4** : La consolidation réduit la fragmentation | ✅ Validé | 3 fusions, ARI +5%, fragmentation −8% |

### Données réelles

- **2879 messages WhatsApp** (fév-mars 2024) → **38 épisodes** cohérents
- Épisodes = sessions conversationnelles journalières (1-3 jours)
- Médiane 26 messages/épisode, top épisode = 475 messages
- Cohérence intra-épisode = 0.63

---

## Couverture du framework théorique

11/19 points du framework couverts, 5 partiels, 3 non commencés :

| Aspect | Statut |
|--------|--------|
| Flux → épisodes (incrémental) | ✅ |
| Multi-critères (4 signaux) | ✅ |
| Structure épisode $(I_e, C_e, U_e, G_e, A_e)$ | ✅ |
| Aging + réactivation | ✅ |
| Fusion (consolidation) | ✅ |
| Micro → macro épisodes | 🟡 Partiel |
| Fermeture probabiliste | 🟡 Seuil fixe |
| Scission d'épisodes | 🟡 Absent |
| **Soft assignment** $P(a \in e_k)$ | 🔴 Hard only |
| **Optimisation globale** $\mathcal{L}$ | 🔴 Glouton |
| **Index (ANN + inversé)** | 🔴 O(n) |

---

## Plan d'action — Modernisation (V4)

Trois changements, classés par ROI. Pas de GPU supplémentaire requis.

---

### Etape 1 — LaBSE remplace MiniLM (30 min)

**Pourquoi :** `all-MiniLM-L6-v2` est entraîné sur de l'anglais. LaBSE (Language-Agnostic BERT Sentence Embeddings) couvre 109 langues dont le français informel et le code-switching FR/EN. Même API `sentence-transformers`, même interface. Impact immédiat sur toutes les étapes en aval : segmentation, recall, graph.

```python
# embedding_engine.py — un seul changement
MODEL_NAME = "sentence-transformers/LaBSE"   # 768d au lieu de 384d
```

**Validation :** re-run `run_v3_experiment.py` → comparer ARI avant/après sur synthetic_200.

---

### Etape 2 — GLiNER remplace spaCy pour le NER (2h)

**Pourquoi :** spaCy `fr_core_news_sm` couvre mal le français informel et le chat. GLiNER (Generalist and Lightweight Named Entity Recognizer) est zero-shot — tu définis les types que tu veux, pas de fine-tuning nécessaire. Multilingue, plus robuste sur les entités informelles.

```python
# entity_extractor.py — nouveau backend
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
labels = ["personne", "projet", "lieu", "organisation", "produit"]
entities = model.predict_entities(text, labels, threshold=0.4)
```

Intégration via `EntityExtractor(backend="gliner")` — le reste du pipeline ne change pas.

**Validation :** comparer le nombre d'entités extraites et leur qualité sur quelques épisodes réels.

---

### Etape 3 — SuperDialseg comme test set externe (3h)

**Pourquoi :** toute l'évaluation actuelle est circulaire — les hyperparamètres ont été calibrés sur `synthetic_200`, généré par le même projet. Un dataset public avec ground truth externe est la seule façon de distinguer "ça fonctionne" de "ça a été trop ajusté".

```bash
git clone https://github.com/salesforce/SuperDialseg data/superdialseg
```

Adapter `parsers/experiment_csv_parser.py` pour lire leur format JSON. Les frontières d'épisodes sont incluses → ARI calculable directement.

**Validation :** ARI sur SuperDialseg sans aucun retuning. Si ARI tombe < 0.30, les paramètres ne généralisent pas.

---

### Protocole d'évaluation rigoureux (parallèle)

| Fix | Action | Temps |
|-----|--------|-------|
| Hold-out figé | Générer `synthetic_1000.csv` une fois, ne plus y toucher | 5 min |
| Variance | 5 seeds différentes sur synthetic_200 → reporter `ARI = μ ± σ` | 10 min |
| Split train/tune/test | synthetic_200 = calibration, synthetic_500 = ablations, SuperDialseg = test final | structurel |

**Règle :** on ne calibre plus un hyperparamètre en regardant le score sur le dataset qu'on va reporter.

---

### Séquence recommandée

```
Jour 1 matin   LaBSE swap + rebuild pipeline + ARI baseline
Jour 1 après   GLiNER backend dans EntityExtractor
Jour 2 matin   Parser SuperDialseg + première évaluation cross-dataset
Jour 2 après   5-fold CV sur synthetic_200 + rapport μ ± σ
```

---

## 🗺️ Roadmap

Deux directions complémentaires et séquencées : **B** (système fonctionnel) d'abord, puis **A** (publication). Un système qui tourne génère les insights empiriques qui nourrissent la formalisation.

### Direction B — Système de mémoire complet (priorité 1)

L'objectif : transformer l'outil d'analyse en un **système de mémoire interrogeable**.

#### Phase B1 : Entity Resolution
> *« Jean », « Jean Dupont », « jdupont@email.com » → même nœud*

- [ ] Modèle de mention : extraction structurée (noms, emails, @handles, organisations)
- [ ] Clustering probabiliste des mentions → entités canoniques
- [ ] Index inversé mention → entité canonique
- [ ] Gestion de l'ambiguïté (Jean Dupont client ≠ Jean Dupont consultant)
- [ ] Évaluation : précision/rappel sur données annotées

#### Phase B2 : Graphe de mémoire
> *Épisodes ↔ Entités ↔ Relations — la représentation unifiée*

- [ ] Structure `MemoryGraph` : nœuds (épisodes, entités, artefacts), arêtes (participe_à, mentionne, cause, suit)
- [ ] Stockage persistant (SQLite ou pickle sérialisé)
- [ ] Mise à jour incrémentale : un nouvel artefact met à jour le graphe
- [ ] Détection de relations inter-épisodes (entités partagées, causalité temporelle)
- [ ] Visualisation du graphe (networkx + export)

#### Phase B3 : Memory Recall
> *`memory.recall("la réunion avec Jean sur le budget")` → épisodes rankés*

- [ ] API `MemoryStore` : `store(artifacts)`, `recall(query, top_k)`, `summarize(episode)`
- [ ] Ranking multi-signal : similarité sémantique + proximité temporelle + importance + fraîcheur
- [ ] Filtrage par entité, période, type d'artefact
- [ ] Gestion de l'ambiguïté : top-k résultats + score de confiance
- [ ] Évaluation : recall@k sur requêtes synthétiques

#### Phase B4 : Aging & compression
> *Courbe d'oubli, résumés, hiérarchie temporelle*

- [ ] Courbe d'Ebbinghaus : memory_score décroît avec le temps, augmente avec les accès
- [ ] Compression : épisode → résumé textuel automatique (LLM ou extractif)
- [ ] Hiérarchie : artefact → épisode → période → projet
- [ ] Forgetting policy : archivage des détails, conservation des résumés
- [ ] Navigation multi-granularité dans les réponses

#### Phase B5 : Index & performance
> *De O(n) à O(k log n)*

- [ ] Index ANN (FAISS ou HNSW) sur les centroïdes d'épisodes
- [ ] Index inversé entités → épisodes pour lookup rapide
- [ ] Cache des embeddings et centroïdes
- [ ] Benchmark : temps de recall sur 10k, 100k, 1M artefacts
- [ ] Profiling et optimisation des hot paths

#### Phase B6 : API & intégration
> *FastAPI + CLI + LLM consumer*

- [ ] API REST : `/store`, `/recall`, `/episodes`, `/graph`
- [ ] CLI : `memory-ai segment --input file.txt`, `memory-ai recall "query"`
- [ ] Intégration LLM : la mémoire épisodique comme contexte RAG
- [ ] Agent conversationnel avec mémoire persistante
- [ ] Documentation OpenAPI + exemples

---

### Direction A — Publication scientifique (priorité 2)

L'objectif : formaliser les résultats pour un paper workshop/conférence.

#### Phase A1 : Soft assignment probabiliste
> *Un artefact peut appartenir à plusieurs épisodes avec des probabilités*

- [ ] $P(a \in e_k)$ pour chaque artefact × épisode candidat
- [ ] Résolution différée : l'ambiguïté se résout avec le contexte futur
- [ ] Soft AttachScore → distribution sur les épisodes
- [ ] Évaluation : impact sur ARI/NMI vs hard assignment
- [ ] Analyse des cas où le soft assignment change la donne

#### Phase A2 : Formulation comme optimisation
> *$\mathcal{L} = \sum Cohesion(e) - \lambda \#Ep - \mu \, Ambiguity$*

- [ ] Définition formelle de $Cohesion(e)$ multi-critères
- [ ] Terme de régularisation : nombre d'épisodes + ambiguïté
- [ ] Approximation EM ou variationnelle pour l'optimisation
- [ ] Comparaison glouton incrémental vs optimisation globale
- [ ] Preuve de convergence (ou bornes empiriques)

#### Phase A3 : Scission d'épisodes
> *Détecter qu'un épisode contient deux fils distincts*

- [ ] Critère de scission : chute de cohérence interne, bi-modalité du centroïde
- [ ] Algorithme : spectral clustering ou détection de communautés sur le sous-graphe
- [ ] Déclencheur : périodique ou événementiel (nouvel artefact ambigu)
- [ ] Évaluation sur cas synthétiques de fils entremêlés

#### Phase A4 : Benchmark formel
> *Cas limites + comparaison baselines*

- [ ] Suite de tests formels (§15 du framework) :
  - Même personne, sujets différents
  - Même sujet, personnes différentes
  - Réactivation après longue pause
  - Artefact ambigu multi-contexte
  - Fusion de contextes
- [ ] Baselines : TextTiling, TopicTiling, BERTopic, fenêtre temporelle fixe
- [ ] Métriques standardisées + significativité statistique
- [ ] Datasets publics (si disponibles) ou protocole de génération reproductible

#### Phase A5 : Rédaction
> *Paper format workshop/conférence*

- [ ] Formalisation mathématique complète
- [ ] Related work : topic segmentation, event detection, episodic memory AI
- [ ] Expériences : ablation, cross-dataset, real data, cas limites
- [ ] Discussion : limites, soft vs hard, scalabilité
- [ ] Cible : workshop AAAI/ACL/EMNLP sur personal AI ou memory-augmented systems

---

### Convergence des directions

```
Direction B (système)              Direction A (paper)
─────────────────────              ──────────────────
B1: Entity Resolution
B2: Graphe de mémoire
B3: Memory Recall        ──→      A1: Soft assignment
                          ╲       A2: Optimisation
B4: Aging & compression    ──→    A3: Scission
B5: Index & performance           A4: Benchmark
B6: API & intégration             A5: Rédaction
```

A peut démarrer en parallèle dès B3 terminé. Les phases B1-B3 fournissent le substrat empirique pour les formalisations de A.

---

## 📚 Datasets de benchmark

### Progression de validation

La stratégie de test suit une montée en complexité :

```
Phase 1 — Dataset synthétique (actuel)
   ↓     contrôle total, vérité terrain parfaite
Phase 2 — WhatsApp / chat personnel (actuel)
   ↓     données réelles, pas de vérité terrain
Phase 3 — Enron Email Dataset
   ↓     données publiques, échelle, multi-sujets
Phase 4 — IRC Conversation Disentanglement
   ↓     problème quasi-identique, 77k messages annotés
Phase 5 — Datasets d'événements (MAVEN, MailEx)
         tester l'extraction d'événements upstream
```

### Datasets pertinents

#### 1. Enron Email Dataset ⭐⭐⭐⭐

Le dataset le plus utilisé en recherche sur les communications professionnelles.

| Propriété | Valeur |
|-----------|--------|
| Taille | ~500 000 emails |
| Participants | ~150 employés (Enron Corp.) |
| Période | 1998–2002 |
| Format | texte / CSV / Maildir |
| Accès | [CMU](https://www.cs.cmu.edu/~enron/) |

**Pertinence** : messages horodatés, conversations multi-jours, sujets entremêlés (budget → réunion → révision → approbation). Idéal pour tester la segmentation temporelle, la reconstruction de threads, et la détection d'épisodes d'activité.

**Limite** : emails formels ≠ chat informel. Messages longs, single-threaded. Ne teste pas le même type de bruit que WhatsApp.

#### 2. IRC Conversation Disentanglement ⭐⭐⭐⭐⭐

> Kummerfeld et al., *"A Large-Scale Corpus for Conversation Disentanglement"*, ACL 2019

Le dataset le **plus directement pertinent** pour notre problème.

| Propriété | Valeur |
|-----------|--------|
| Taille | ~77 000 messages |
| Source | Canal IRC #Ubuntu |
| Annotations | Graphe de réponse, conversations identifiées |
| Accès | [GitHub](https://github.com/jkkummerfeld/irc-disentanglement) |

**Pertinence** : c'est exactement le même problème — un flux de messages où plusieurs conversations coexistent, et il faut identifier les conversations indépendantes :

```
flux messages → identifier conversations indépendantes
flux artefacts → identifier épisodes mémoire
```

**Limite** : multi-party (N utilisateurs dans un canal), pas dyadic comme WhatsApp. Le signal "qui parle à qui" n'existe pas dans notre cas.

#### 3. MAVEN — Event Detection ⭐⭐⭐

> Wang et al., *"MAVEN: A Massive General Domain Event Detection Dataset"*, EMNLP 2020

| Propriété | Valeur |
|-----------|--------|
| Événements | ~118 000 annotés |
| Types | 168 types d'événements |
| Source | Articles Wikipedia |
| Accès | [GitHub](https://github.com/THU-KEG/MAVEN-dataset) |

**Pertinence** : notre pipeline fait implicitement `artefacts → événements → épisodes`. MAVEN permet de tester l'étape de détection d'événements en amont.

**Limite** : texte journalistique, pas conversationnel. C'est de la détection de triggers, pas de la segmentation épisodique.

#### 4. MailEx — Email Event Extraction ⭐⭐

| Propriété | Valeur |
|-----------|--------|
| Taille | ~4 000 emails, ~8 000 événements annotés |
| Annotations | Conversations, événements, arguments |

**Pertinence** : permet de tester `emails → event detection → episode construction`.

**Limite** : petit dataset, niche.

#### 5. AMI/ICSI Meeting Corpus ⭐⭐⭐

| Propriété | Valeur |
|-----------|--------|
| Taille | 100h (AMI) + 75h (ICSI) de réunions |
| Annotations | Segmentation topique, actes de dialogue, résumés |
| Accès | [Edinburgh](https://groups.inf.ed.ac.uk/ami/corpus/) |

**Pertinence** : réunions annotées avec segmentation par topic, actes de dialogue et résumés. Plus proche de la mémoire épisodique que du texte journalistique.

#### 6. SAMSum — Dialogue Summarization ⭐⭐⭐

> Gliwa et al., Samsung R&D Institute, 2019

| Propriété | Valeur |
|-----------|--------|
| Taille | ~16 000 dialogues |
| Annotations | Résumés par conversation |
| Style | Chat informel (créé par linguistes) |

**Pertinence** : dialogues informels avec résumés, fournit implicitement des frontières d'épisodes. Le style chat est plus proche de nos données WhatsApp que les emails.

#### 7. GDELT — Global Events ⭐

| Propriété | Valeur |
|-----------|--------|
| Taille | Millions d'événements mondiaux |
| Données | Acteurs, lieux, timestamps |

**Pertinence limitée** : données structurées géopolitiques, utile uniquement pour tester des algorithmes de clustering d'événements temporels. Pas conversationnel.

### 💡 Positionnement recherche

> **Le problème de segmentation d'épisodes dans un flux d'événements personnels est encore très peu étudié.**

Les datasets existants couvrent :
- ✅ Conversations (IRC, SAMSum)
- ✅ Événements (MAVEN, GDELT)
- ✅ Emails (Enron, MailEx)
- ❌ **Mémoire personnelle multi-modale**

Il n'existe **aucun dataset** qui combine :
- Flux multi-source (messages + emails + notes + documents)
- Annotations d'épisodes de mémoire (pas juste de conversations)
- Dimension temporelle longue (semaines/mois, pas minutes)
- Perspective personnelle (1ère personne, pas observation externe)

**→ La création d'un tel dataset + benchmark serait en soi une contribution publiable.**

---

## 🔬 Analyse critique & pistes d'amélioration

### Diagnostic du système actuel

| Module | Approche | Verdict |
|---|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` (384d, EN) | ⚠️ Modèle anglophone sur données FR/EN mixtes |
| **Segmentation** | Greedy online, AttachScore | ⚠️ Pas d'optimisation globale, centroid drift |
| **NER** | spaCy / GLiNER | ⚠️ URLs capturées comme entités |
| **Goal vectors** | Heuristiques regex → embedding | ❌ Signal quasi-nul sur du chat informel |
| **Splitting** | AgglomerativeClustering + silhouette | ✅ Solide |
| **Consolidation** | Union-Find + contraintes | ✅ Correct |
| **Recall** | Hybrid search multi-signal | ✅ Bien architecturé |
| **Summarizer** | TF-IDF + mood lexicon | ✅ Fonctionnel sans LLM |

### Pistes d'amélioration priorisées

| # | Action | Effort | Impact |
|---|---|---|---|
| 1 | **Embedding multilingue** (BGE-M3 ou paraphrase-multilingual) | 🟢 1 ligne | ⭐⭐⭐⭐⭐ |
| 2 | **Filtre URLs** dans entity_extractor | 🟢 Trivial | ⭐⭐⭐ |
| 3 | **Fenêtre contextuelle** (3-5 msgs) pour embedding | 🟡 Moyen | ⭐⭐⭐⭐ |
| 4 | **Segmentation 2-passes** (change-point + clustering global) | 🔴 Fort | ⭐⭐⭐⭐⭐ |
| 5 | **BERTopic** comme signal de topic | 🟡 Moyen | ⭐⭐⭐⭐ |
| 6 | **LLM local** pour intent/topic extraction | 🔴 Fort | ⭐⭐⭐⭐ |
| 7 | **Vérité terrain manuelle** sur données réelles | 🟡 Moyen | ⭐⭐⭐⭐⭐ |

---

## Installation

```bash
# Cloner et configurer l'environnement
git clone <repo-url> memory_ai_lab
cd memory_ai_lab
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# Lancer l'ablation study V2
cd src
python experiment_runner.py

# Notebook d'analyse
jupyter notebook notebooks/analysis.ipynb
```

## Stack technique

| Composant | Version |
|-----------|---------|
| Python | 3.10.12 |
| sentence-transformers | 5.3.0 (all-MiniLM-L6-v2) |
| scikit-learn | 1.7.2 |
| torch | 2.10.0 (CUDA 12) |
| matplotlib + seaborn | 3.10.8 / 0.13.2 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |

---

## Licence

Projet de recherche personnel. Licence à définir.
