# 09 — Architecture hybride : Boundary Detector + Episode Identity

> Document de conception. Issu du brainstorming du 18/03/2026.

---

## Le problème de l'architecture actuelle

`EpisodeSegmenter.segment()` appelle `AttachScore` sur **chaque message**
contre **tous les épisodes candidats** — même quand le message continue
manifestement la conversation courante.

```
11 740 messages × k candidats actifs = des millions d'appels Python
```

C'est du gaspillage : la grande majorité des messages ne sont pas des
frontières d'épisode. On cherche une rupture là où il n'y en a pas.

---

## La séparation fondamentale

L'algorithme actuel confond deux problèmes distincts :

```
Problème A — "est-ce que le topic change ici ?"
Problème B — "si oui, est-ce un nouvel épisode ou une réactivation ?"
```

DialSeg/BERTSeg résolvent A mais pas B.
Notre AttachScore résout B mais l'appelle inutilement partout.

---

## Architecture hybride en 3 stages

```
mE5-base (multilingue, code-switching FR/EN)
         ↓  embeddings (GPU, batch, déjà fait)

┌─────────────────────────────────────────┐
│  STAGE 1 — Boundary Detector           │
│                                         │
│  Input  : [emb_{t-1}, emb_t]           │
│           + diff sémantique             │
│           + gap temporel (log)          │
│  Modèle : MLP léger (entraînable CPU)  │
│  Output : P(frontière ici)             │
│                                         │
│  Complexité : O(n) — parallèle GPU     │
│  Entraînable : 20 min Colab sur gold   │
└──────────────┬──────────────────────────┘
               │ seulement si P > seuil
               ↓ (~640 appels sur 11 740)

┌─────────────────────────────────────────┐
│  STAGE 2 — Episode Identity            │
│                                         │
│  Notre AttachScore (inchangé)          │
│  Compare le nouvel artefact aux        │
│  centroïdes EMA de tous les épisodes   │
│  → Nouvel épisode OU réactivation      │
│                                         │
│  Complexité : O(k) — 18x moins appelé │
└──────────────┬──────────────────────────┘
               ↓

┌─────────────────────────────────────────┐
│  STAGE 3 — Lifecycle Management        │
│                                         │
│  EMA centroid update                   │
│  ACTIVE → DORMANT (1440 min)           │
│  Réactivation DORMANT → ACTIVE         │
│  entity_weights, goal_centroid         │
│                                         │
│  Inchangé — notre différenciateur clé  │
└─────────────────────────────────────────┘
```

---

## Gain de performance théorique

| Opération | Avant | Après |
|-----------|:-----:|:-----:|
| AttachScore calls | 11 740 × k | 640 × k |
| Facteur | 1x | **18x moins** |
| Boundary detector | — | O(n) batch GPU |

---

## Boundary Detector — détail

### Features par message t

```python
diff    = emb_t - emb_{t-1}              # (768,) — direction du changement
sim     = cosine_sim(emb_{t-1}, emb_t)   # (1,)   — continuité sémantique
gap_log = log1p(gap_minutes) / log1p(1440) # (1,) — gap normalisé par 24h

features = concat([diff, sim, gap_log])  # (770,)
```

### Architecture MLP

```python
BoundaryMLP : 770 → 128 → ReLU → Dropout(0.2) → 32 → ReLU → 1 → Sigmoid
```

### Données d'entraînement (depuis gold tune)

```
Positifs : ~448 frontières (start_idx de chaque épisode)
Négatifs : ~8 506 non-frontières
Ratio    : ~19:1 → pos_weight=19 dans BCEWithLogitsLoss
```

### Entraînement

- 30 epochs, Adam lr=1e-3
- Validation sur tune (F1 score des frontières)
- Threshold optimisé sur tune (pas sur test)
- CPU : ~2 min | GPU : ~20 sec

---

## Embeddings — switch vers mE5-base

```python
# Avant
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # EN only, 384d

# Après
MODEL_NAME = "intfloat/multilingual-e5-base"  # 100+ langues, 768d
PREFIX     = "passage: "  # requis par mE5 pour les documents
```

Le switch couvre automatiquement :
- Français informel et argot
- Code-switching FR/EN dans le même message
- Futur multi-source (emails EN, notes FR, etc.)

**Note :** les embeddings Drive (`group_embeddings.npy`) doivent être
recalculés une fois après le switch.

---

## Implémentation

| Fichier | Rôle |
|---------|------|
| `src/boundary_detector.py` | BoundaryMLP, training, inference |
| `src/episode_segmenter_hybrid.py` | HybridEpisodeSegmenter (Stage 1+2+3) |
| `colab/02_train_boundary_detector.ipynb` | Entraînement + évaluation |
| `colab/01_eval_ari.ipynb` | Mise à jour avec hybrid segmenter |

---

## Métriques d'évaluation du Boundary Detector

L'ARI mesure la qualité globale de la segmentation.
Pour le boundary detector seul, on utilise :

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| F1 frontières | F1(y_boundary) | précision + rappel des coupures |
| WindowDiff | erreur de fenêtre glissante | standard topic segmentation |
| Pk | probabilité d'erreur | standard topic segmentation |

---

## Ce que cette architecture préserve

> Notre valeur unique est le **Stage 2 — Episode Identity**.
> Ni DialSeg, ni un LLM en fenêtre glissante ne font la distinction
> "nouvel épisode vs réactivation d'un épisode dormant".
>
> Le Boundary Detector est un accélérateur, pas un remplacement.
> L'identité épisodique reste notre différenciateur scientifique.
