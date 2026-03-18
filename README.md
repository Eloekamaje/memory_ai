# Memory AI Lab

> **Construire la couche mémoire d'une intelligence personnelle continue**

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Pipeline-V3-green)
![Eval](https://img.shields.io/badge/ARI_gold-0.1619-orange)

---

## Vision — Ivias / Mivias

Ce projet est le noyau technique d'**Ivias** (ou *Mivias* — My Ivias), un OS cognitif personnel.

L'architecture centrale repose sur trois pièces :

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Memory Engine  ←──────  Decision Engine  ──────→  Agent │
│   (State)                  (Bridge)                (Policy) │
│                                                     │
│   "sans contexte → agent stupide"                  │
│   "sans action   → mémoire inutile"                │
└─────────────────────────────────────────────────────┘
```

La boucle agentique à 6 étapes :

```
OBSERVE → UPDATE MEMORY → EVALUATE MEMORY → DECIDE → ACT → LOOP
```

Le **Memory Engine** est le State : il capture, structure, vieillit et répond à des requêtes.
L'**Agent** est la Policy : il agit dans le monde.
Le **Decision Engine** est le Bridge manquant : il lit la mémoire et décide quoi faire.

Ce repo implémente le Memory Engine.

---

## Le problème

Dans la vie réelle, les informations arrivent comme un flux continu : messages, emails, notes, réunions.
La mémoire humaine ne les stocke pas séparément — elle les regroupe en **épisodes**.

> 4 objets pour l'ordinateur. 1 événement pour l'humain.

**Question centrale** : comment détecter automatiquement qu'un ensemble d'artefacts appartient au même épisode, dans un flux temporel multi-source ?

---

## Architecture multi-canal

La mémoire se nourrit de plusieurs flux en parallèle, chacun avec un niveau de confiance :

```python
@dataclass
class Channel:
    name: str           # "whatsapp" | "gmail" | "calendar" | "notes" | "terminal"
    parser: Callable    # parse brut → Artifact
    adapter: Callable   # normalise timestamps, auteur, contenu
    trust_weight: float # confiance dans les métadonnées
```

| Canal | Trust | Notes |
|-------|:-----:|-------|
| Calendrier | 1.0 | Événements structurés, timestamps fiables |
| Terminal | 1.0 | Commandes horodatées, traces exactes |
| Notes personnelles | 0.9 | Intention explicite de l'utilisateur |
| Gmail | 0.8 | Threads structurés, mais bruit pub/notif |
| WhatsApp groupe | 0.7 | Informel, multilingue, hors-sujet fréquent |

---

## Cycle de vie des épisodes

Chaque épisode traverse un cycle de vie complet :

```
         Nouveau artefact
              │
              ▼
           BORN ──────────────────────────────────────────┐
              │ premiers artefacts                        │
              ▼                                           │
           ACTIVE ◄──── réactivation ◄──── DORMANT        │
              │  artefact lié                  │          │
              │                     inactivité > 24h      │
              ▼                                │          │
         COMPRESSED ◄── consolidation nocturne ┘          │
              │                                           │
              ▼                                           │
          ARCHIVED ──── accès rare ──────────────────────►│
              │                                    REVIVED│
              ▼                                           │
           DELETED                                        │
              │                                           │
              └──────────────────────────────────────────-┘
```

### Vieillissement (courbe d'Ebbinghaus)

```
retention(e, t) = exp(−λ(e) × t / (1 + log1p(access_count(e))))
```

Le taux de décroissance `λ` dépend de l'importance, du canal et de la richesse en entités :

```python
def decay_rate(episode) -> float:
    λ = 0.05 / (1 + episode.importance)
    λ *= CHANNEL_DECAY_FACTOR[episode.primary_channel]
    λ /= (1 + log1p(len(episode.entities)))
    if episode.is_open:
        λ *= 0.3   # épisode en cours → décroît très lentement
    return λ
```

Compression en 4 couches :
```
[████████████] Contenu complet     (ACTIVE)
[████████░░░░] Contenu partiel     (DORMANT)
[████░░░░░░░░] Résumé              (COMPRESSED)
[█░░░░░░░░░░░] Trace mémorielle   (ARCHIVED)
[░░░░░░░░░░░░] Oubli              (DELETED)
```

---

## Modèle formel

Un artefact $a_i$ possède un temps $t_i$, un type $y_i$, des entités $E_i$, un vecteur sémantique $v_i$ et un vecteur d'objectif $g_i$.

L'**AttachScore** détermine si un artefact rejoint un épisode existant :

$$\text{AttachScore}(a, e) = \alpha \cdot \text{Sem}(a,e) + \beta \cdot \text{Ent}(a,e) + \gamma \cdot \text{Temp}(a,e) + \delta \cdot \text{Goal}(a,e) - \rho \cdot \text{AgePenalty}(e)$$

| Composante | Poids V3 | Rôle |
|-----------|:--------:|------|
| **Sem** | α = 0.45 | Similarité cosinus artefact ↔ centroïde EMA |
| **Ent** | β = 0.25 | Recouvrement Jaccard des entités |
| **Temp** | γ = 0.10 | Proximité temporelle (décroissance exponentielle) |
| **Goal** | δ = 0.20 | Similarité d'objectif latent |
| **AgePenalty** | ρ = 0.05 | Pénalité dormance (log1p) |

---

## Mémoire interrogeable

5 types de requêtes supportés :

| Type | Exemple |
|------|---------|
| **Factuelle** | *"Où en est le projet X ?"* |
| **Temporelle** | *"Qu'est-ce qui s'est passé la semaine dernière ?"* |
| **Relationnelle** | *"Quels épisodes impliquent Jean ?"* |
| **Méta** | *"Qu'est-ce que j'ai oublié ?"* |
| **Prospective** | *"Qu'est-ce que je n'ai pas terminé ?"* |

Architecture du QueryEngine :
```
Query Parser → Query Planner → RecallEngine / MemoryGraph / Stats
                                         ↓
                               Answer Synthesizer
                    (source episodes + lifecycle state + confidence)
```

---

## Pipeline V3

```
Flux d'artefacts (multi-canal)
         │
         ▼
  Parser + Adapter (channel-aware)
         │
         ▼
  Embeddings (all-MiniLM-L6-v2, 384d)  ←── V4: LaBSE 768d
         │
  Goal Heuristics (type→intent + regex + entities)
         │
         ▼
  EpisodeSegmenter V3
         │  ├─ AttachScore 4 termes + AgePenalty
         │  ├─ Centroid EMA (α=0.80)
         │  ├─ ACTIVE → DORMANT (1440 min)
         │  ├─ Réactivation DORMANT → ACTIVE
         │  └─ Candidats : fenêtre récente + entités partagées
         │
         ▼
  Consolidation (Union-Find + contrainte temporelle)
         │
         ▼
  EpisodeSplitter (silhouette + cohérence)
         │
         ▼
  Épisodes → MemoryGraph → RecallEngine → QueryEngine
```

---

## Résultats

### Synthetic (calibration)

| Version | ARI | Dataset |
|---------|:---:|---------|
| V1 baseline | 0.31 | synthetic_200 |
| V2 entity-boosted | 0.506 | synthetic_200 |
| V3 EMA | **0.5179** | synthetic_200 |
| V3 cross-dataset | 0.3243 | synthetic_500 → surapprentissage |

### Gold dataset réel — "Les taras" (18/03/2026)

Dataset : 11 740 messages WhatsApp, groupe de 15 participants, 943 jours (août 2023 → mars 2026).
Gold annoté via pipeline LLM silver → review humaine.

| Métrique | Valeur |
|----------|:------:|
| **ARI** | **+0.1619** |
| **NMI** | **0.8289** |
| Messages évalués | 11 740 |
| Épisodes gold | 641 |
| Épisodes prédits | 5 360 |

**Diagnostic :**

- NMI = 0.83 → le pipeline capture bien la structure globale
- ARI = 0.16 → **sur-fragmentation massive** (×8 trop d'épisodes)
- Cause : `hard_break_minutes=720` coupe trop, `attach_threshold=0.30` trop bas

**Plan de correction (à tester en Colab) :**
```python
# Pistes grid search
attach_threshold    : 0.30 → 0.40–0.50
hard_break_minutes  : 720  → 2880 ou 0 (désactivé)
time_threshold_minutes : 120 → 240–480
```

### Gold dataset — statistiques

| Propriété | Valeur |
|-----------|--------|
| Période | 16/08/2023 → 17/03/2026 (943 jours) |
| Messages | 11 740 |
| Épisodes gold | 641 (moy. 18.3 msgs, médiane 9) |
| Solo (1 msg) | 34 épisodes |
| Longs (>50 msgs) | 48 épisodes |
| Frontières | 640 (608 auto-approuvées ≥2h, 32 review humaine) |

---

## Plan V4 — Modernisation

### Étape 1 — LaBSE remplace MiniLM (30 min)

```python
# embedding_engine.py
MODEL_NAME = "sentence-transformers/LaBSE"  # 109 langues, 768d
```

Couvre le français informel et le code-switching FR/EN. Même API.

### Étape 2 — GLiNER remplace spaCy (2h)

```python
# entity_extractor.py
extractor = EntityExtractor(backend="gliner")
# zero-shot, multilingue, labels personnalisables
```

### Étape 3 — Grid search sur gold (Colab GPU)

Utiliser `colab/01_eval_ari.ipynb` — embeddings en cache Drive, grid search en <5 min.

### Étape 4 — SuperDialseg comme test externe (3h)

```bash
git clone https://github.com/salesforce/SuperDialseg data/superdialseg
```

ARI sur données publiques sans retuning → validation de généralisation.

---

## Workflow

```
VSCode + Claude (dev)     GitHub/memory_ai      Google Drive (data)
        │                       │                       │
        ├── code + algo          │                       ├── group_anon.txt
        └── git push ───────────►                       ├── group_gold.json
                                 │                      └── group_embeddings.npy
                          Colab GPU ◄──────────────────────┘
                     !git clone + drive.mount
                     → run notebooks (GPU T4)
```

```bash
# Dev local
git add -A && git commit -m "..." && git push

# Colab — ouvrir colab/01_eval_ari.ipynb
# Cellule 1 : git pull auto
# GPU T4 : embeddings ~10 sec, grid search <5 min
```

---

## Roadmap

```
Court terme (< 1 mois)
  ├── LaBSE + GLiNER                     plan V4 — 2 jours
  ├── Grid search params sur gold         Colab GPU — 1 jour
  ├── SuperDialseg evaluation             plan V4 — 1 jour
  ├── 5-fold CV + variance report         protocole — 1 jour
  └── Decision Engine v0 (règles)         H0 — 1 semaine

Moyen terme (1-3 mois)
  ├── ✅ Dataset gold réel (641 épisodes) H6.a — DÉJÀ FAIT
  ├── Bi-encoder fine-tuné sur gold       H2.a — débloqué, 1 session Colab
  ├── Embedding contextuel 5-window       H2.b — CPU
  ├── MLP AttachScore appris              H3.a — CPU
  ├── Recall proactif background          H4.a
  └── Pipeline incrémental               H5.b

Long terme (3-12 mois)
  ├── Soft assignment probabiliste        H3.b
  ├── Knowledge Graph sémantique          H4.b
  ├── FAISS à 1M artefacts               H5.a
  └── Multi-source (email, notes, cal)   H8.a

Vision (12+ mois)
  └── OS de la connaissance personnelle  H8.c
```

> **Note :** `data/group_gold.json` — 641 épisodes annotés sur 11 740 messages réels (943 jours, 15 participants) — remplace et dépasse H6.a (200 épisodes synthétiques). Le bi-encoder fine-tuné (H2.a) passe de "long terme" à "moyen terme" grâce à ce dataset.

### Hypothèses de recherche (détail dans `ai-brain/07_visionary_hypotheses.md`)

| ID | Hypothèse | Priorité |
|----|-----------|:--------:|
| **H0** | Decision Engine comme pièce manquante Ivias | 🔴 Court terme |
| **H1** | Goal vectors améliorent la segmentation | 🟡 Moyen |
| **H2a** | Bi-encoder fine-tuné sur épisodes annotés | 🟡 Long |
| **H2b** | Embedding contextuel 5-window | 🟢 Moyen |
| **H3a** | MLP AttachScore appris > heuristique | 🟡 Moyen |
| **H3b** | Soft assignment probabiliste | 🔴 Long |
| **H4a** | Recall proactif déclenché par la mémoire | 🟢 Moyen |
| **H5a** | FAISS → O(log n) à 1M artefacts | 🔴 Long |
| **H6a** | 200 épisodes annotés → fine-tuning | 🟡 Moyen |
| **H8a** | Multi-source complet (email+cal+notes) | 🔴 Long |
| **H8c** | OS cognitif personnel | 🌟 Vision |

---

## Installation locale

```bash
git clone https://github.com/Eloekamaje/memory_ai.git
cd memory_ai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

## Stack

| Composant | Version |
|-----------|---------|
| Python | 3.10 |
| sentence-transformers | 5.3.0 (all-MiniLM-L6-v2 → LaBSE V4) |
| spaCy / GLiNER | 3.8 / 0.2 |
| scikit-learn | 1.7 |
| torch | 2.10 (CUDA 12) |
| networkx | 3.x |

---

*Projet de recherche personnel — Majella / Ivias*
