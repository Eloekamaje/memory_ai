# Memory AI Lab — Vision Architecture

> Version 2025-03
> Statut : document de référence — mis à jour à chaque pivot architectural

---

## 1. Problème et originalité

### Ce qu'on fait

Segmenter automatiquement une conversation de groupe WhatsApp (FR+EN, 18 mois,
~11 700 messages) en **épisodes** — des unités cohérentes de mémoire collective,
non nécessairement contiguës dans le temps.

### Ce qui rend ce problème unique

La littérature SOTA (SuperDialseg, TextSeg, BERT-CRF) traite la **segmentation
topique** : des segments contigus, dans des conversations courtes, sans retour
en arrière. Notre tâche est différente sur trois points fondamentaux :

| Propriété | SOTA dialogue segmentation | Notre tâche |
|---|---|---|
| Contiguïté | Segments adjacents | Épisodes non-contigus |
| Unité | Topic | Épisode (topic + goal + acteurs) |
| Réactivation | Impossible par construction | DORMANT → REACTIVATED |
| Durée | < 1 heure | Jusqu'à 18 mois |
| Langue | Mono-lingue | FR + EN mélangés |

Aucun papier publié à ce jour ne traite exactement cette combinaison.
Les travaux les plus proches sont :
- **Conversation Disentanglement** (Kummerfeld et al. 2019) — nécessite des reply-to explicites
- **MemGPT** (Packer et al. 2023) — mémoire hiérarchique LLM, pas de segmentation structurée
- **GraphRAG** (Microsoft 2024) — organisation en graphe, pas de lifecycle épisodique

---

## 2. Ancrage théorique — Mémoire épisodique humaine

### Qu'est-ce qui caractérise un épisode mémorable ?

D'après Tulving (1972), Zacks *Event Segmentation Theory* (2007) et Conway (2000),
un épisode s'ancre dans la mémoire longue terme quand il réunit :

**1. Conséquence / Outcome**
Un épisode est mémorable quand il *change quelque chose* : une décision est prise,
un plan arrêté, un problème résolu. Les échanges sans outcome s'effacent.

**2. Rupture de situation** (Event Segmentation Theory)
Le cerveau coupe les épisodes quand le *modèle de situation* doit être mis à jour :
nouvel acteur, nouvel objectif, changement de registre émotionnel, information
externe. C'est exactement la boundary detection.

**3. Charge émotionnelle**
Surprise, frustration, joie intense → encodage renforcé (effet von Restorff).
Un échange neutre et routinier s'efface rapidement.

**4. Réactivation**
Les épisodes forts reviennent dans la conversation ("suite à notre discussion sur X").
C'est un signal de saillance rétrospective — et un défi technique majeur.

**5. Cohérence interne (goal unity)**
Un épisode fort a une unité d'objectif : on sait *de quoi* il s'agit et *pourquoi*.
Topic + intention partagée = épisode. Topic seul = fragment.

### Les actes de langage structurent l'épisode (Austin/Searle)

Chaque message accomplit un **acte** qui a un effet sur l'état épisodique :

| Acte | Exemple | Effet |
|---|---|---|
| **Directif** | "Tu peux t'en occuper ?" | *Ouvre* une obligation |
| **Commissif** | "Je le fais ce soir" | *Ferme* avec tâche assignée |
| **Assertif** | "Le budget est à 5K" | *Développe* l'épisode |
| **Expressif** | "Merci c'est parfait 👍" | *Clôt* un sous-épisode |
| **Déclaratif** | "C'est décidé, on part sur X" | *Ferme définitivement* |

Un épisode s'ouvre sur un **directif ou une question** et se ferme sur un
**commissif ou déclaratif**. Tant que la seconde partie de la paire adjacente
n'est pas venue (réponse, accord, engagement), l'épisode est *en attente*.

---

## 3. Architecture cible

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│  ENCODAGE MULTI-SIGNAL  (par message m_t, FR ou EN ou mélangé) │
│                                                                 │
│  mE5-base (100 langues)     → 768d  sémantique                 │
│  mDeBERTa-v3 XNLI           →   8d  speech acts               │
│  GLiNER gliner_multi-v2.1   →   Nd  entités nommées            │
│  Regex bilingues FR+EN       →   3d  marqueurs discursifs       │
│  len(content) normalisé      →   1d  longueur du message        │
│  author_change(t)            →   1d  changement de participant  │
│  gap temporel log-normalisé  →   1d  écart depuis msg précédent │
│                              ─────────                         │
│                               782d  vecteur d'entrée TCN        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│  STAGE 1 — TCN BOUNDARY DETECTOR                                │
│                                                                 │
│  Fenêtre causale [t-w ... t], w=100                             │
│  TCN 3 blocs, kernel=3, channels=128                            │
│  → P(boundary_t) ∈ [0,1]                                       │
│                                                                 │
│  Entraîné sur tune_early (65% du gold tune)                     │
│  Seuil optimisé sur recall ≥ 0.85                               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │  p_t > threshold ?      │
                     │  NON → Stage 2          │
                     │  OUI → Retrieval        │
                     └────────────┬────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│  RETRIEVAL — Nouvel épisode ou réactivation ?                   │
│                                                                 │
│  Pour chaque épisode DORMANT E_k :                              │
│    score(m_t, E_k) = MLP_retrieval([                            │
│        cos_sim(emb_t, E_k.centroid),                            │
│        entity_jaccard(entities_t, E_k.entity_weights),          │
│        speech_act_alignment(acts_t, E_k.act_history),           │
│        time_decay(t, E_k.last_active),                          │
│    ])                                                           │
│                                                                 │
│  max_score > θ_reactivate → REACTIVATE E_k                      │
│  sinon                    → CREATE new episode                  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│  STAGE 2 — ATTACH SCORE (MLP appris, pas Optuna)                │
│                                                                 │
│  Pour chaque épisode ACTIF E_k :                                │
│    score(m_t, E_k) = MLP_attach([                               │
│        cos_sim(emb_t, E_k.centroid),          # α·Sem           │
│        entity_overlap(m_t, E_k),              # β·Ent           │
│        time_decay(t, E_k.last_active),        # γ·Temp          │
│        cos_sim(goal_t, E_k.goal_centroid),    # δ·Goal          │
│        age_penalty(E_k),                      # ρ·Age           │
│        speech_act_score(m_t),                 # nouveau signal  │
│        discourse_marker(m_t),                 # nouveau signal  │
│    ])                                                           │
│                                                                 │
│  assign m_t → argmax(score)                                     │
│  UPDATE E_k.centroid, goal_centroid, entity_weights via EMA     │
└─────────────────────────────────────────────────────────────────┘
```

### Pourquoi MLP appris plutôt qu'Optuna

Optuna cherche dans un espace de paramètres scalaires (α, β, γ...) sans gradient.
Un MLP à 2 couches appris sur les paires (message, épisode) gold :
- Capture des **interactions** entre signaux (pas juste une somme pondérée)
- Généralise via régularisation (dropout, weight decay)
- Peut être ré-entraîné quand de nouvelles données gold arrivent

---

## 4. Multi-signal — détail des features

### 4.1 Sémantique — mE5-base (768d)

```
Encodage isolé : "passage: " + content
1 message = 1 vecteur, sans fenêtrage contextuel
```

H2.b (fenêtrage contextuel) a été testé → régression ARI (frontières lissées).
H2.a (fine-tuning contrastif) a été testé → régression ARI (over-segmentation).
Baseline mE5 isolé reste optimal sur ce dataset.

### 4.2 Entités — GLiNER gliner_multi-v2.1 (overlap calculé)

```
Backend  : urchade/gliner_multi-v2.1
Langue   : multilingue, zero-shot
Types    : person, org, location, project, event, product, document
Seuil    : confidence ≥ 0.55
Cache    : group_entities_gliner.json
```

Utilisation dans le pipeline :
- `artifact.entities` → liste de surface forms
- `entity_jaccard(m_t, E_k)` = |entities_t ∩ E_k.top_entities| / |entities_t ∪ E_k.top_entities|
- Active le terme **β·Ent** dans AttachScore (actuellement = 0 car entities=[])

### 4.3 Speech Acts — mDeBERTa-v3 XNLI (8d)

```
Modèle   : MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
Langue   : 15 langues dont FR et EN — NLI cross-lingue
           premise FR + hypothesis EN → fonctionne
Taille   : ~280MB, ~2000 msgs/min GPU T4
```

**8 labels bilingues** (FR + EN dans la même hypothèse — documenté MoritzLaurer 2023) :

```python
# Axe Résolution (état de l'épisode)
"posing a question or raising an open problem / poser une question ou soulever un problème",
"committing to an action or making a promise / s'engager à faire quelque chose",
"declaring a final decision or closing a topic / déclarer une décision finale",

# Axe Saillance (mémorabilité)
"expressing urgency surprise or strong emotion / exprimer de l'urgence ou une émotion forte",
"casual acknowledgment or small talk / accusé de réception ou bavardage",

# Axe Temporalité (position dans le cycle épisodique)
"introducing a new topic or situation / introduire un nouveau sujet",
"continuing the current thread / continuer la discussion en cours",
"referencing or reactivating a past episode / référencer ou réactiver un épisode passé",
```

Sortie : `axis_scores (n, 3)` — [résolution, saillance, temporalité] ∈ [0,1]
et `goal_vector (n, 768)` — représentation dans l'espace mE5.

### 4.4 Marqueurs discursifs — Regex bilingues (3d)

```
Pas de modèle. Regex FR + EN. Résultat : vecteur binaire (3,)
  [0] transition   — "en passant / by the way / sinon / anyway"
  [1] closure      — "c'est décidé / it's settled / validé / done deal"
  [2] reactivation — "suite à / following up / comme on avait dit / as discussed"
```

Gratuit, déterministe, 0ms de latence. Signal de frontière fort sur cas explicites.

### 4.5 Signaux structurels (2d)

```
len_norm      = min(len(content.split()) / 50, 1.0)
               0.0 = "ok" · 1.0 = message élaboré (> 50 mots)
               Court = acknowledgment/clôture · Long = élaboration/cœur

author_change = 1 si author_t ≠ author_{t-1} else 0
               Signal de transition sociale — souvent corrélé aux frontières
```

---

## 5. Stratégie multilingue

Chaque couche utilise l'outil optimal pour sa langue :

| Couche | Outil | Stratégie langue |
|---|---|---|
| Sémantique | mE5-base | Nativement multilingue (100L) |
| Entités | GLiNER-multi | Nativement multilingue |
| Speech acts | mDeBERTa-v3 XNLI | Cross-lingual NLI : premise FR, hypothesis EN/FR |
| Marqueurs | Regex | Listes parallèles FR + EN |
| Longueur | Comptage mots | Agnostique |

**Principe** : ne pas détecter la langue et router — couvrir les deux langues
nativement à chaque couche. La conversation est souvent mélangée dans un même
message (code-switching FR/EN) ; la détection de langue par message serait
inexacte et fragile.

---

## 6. Résultats expérimentaux

### Baseline et tentatives

| Expérience | ARI test | Note |
|---|---|---|
| EpisodeSegmenterFast (Stage 2+3 seul) | ~0.85 | Pas de Stage 1 |
| + TCN Boundary Detector 769d (emb+gap) | **0.9012** | ✅ Meilleur résultat |
| + cos_sim feature (770d) | 0.8473 | Shortcut bias → over-segmentation |
| + H2.b embeddings contextuels | régression | Frontières lissées |
| + H2.a fine-tuning contrastif mE5 | 0.8053 | Over-segmentation |
| + SuperDialseg pre-training | régression | Gap de distribution trop grand |
| + NSP zero-shot coherence | prec=0.056 | BERT NSP ≠ WhatsApp |

### Oracle Stage 1 (borne supérieure théorique)

```
ARI oracle test = 0.9624  (precision=recall=1.0 sur les frontières)
ARI actuel test = 0.9012
Gap potentiel   = +0.062
```

Le gap est réel mais Stage 1 n'est pas le seul goulot : β·Ent = 0 et δ·Goal = 0
signifient que Stage 2 fonctionne à 3/5 de sa capacité théorique.

---

## 7. Roadmap

### Phase A — Activer les signaux existants (priorité immédiate)

```
A1. GLiNER entities → artifact.entities → β·Ent actif
    Cache : group_entities_gliner.json
    Notebook : 05_enrich_artifacts.ipynb cellule 5

A2. DeBERTa speech acts → goal_vector + axis_scores → δ·Goal actif
    Cache : group_goal_vectors.npy + group_episode_scores.npy
    Notebook : 05_enrich_artifacts.ipynb cellule 7

A3. Intégrer dans notebooks 01 + 02 (snippet cellule 9 du notebook 05)
    Re-entraîner TCN, re-lancer Optuna → mesurer delta ARI
```

### Phase B — Enrichir le TCN (features additionnelles)

```
B1. Nouveau module discourse_markers.py
    Input  : artifact.content
    Output : (3,) binaire [transition, closure, reactivation]
    Modèle : regex FR+EN, pas de ML

B2. Switch mDeBERTa-v3-base-mnli-xnli (remplace DeBERTa-v3-small)
    Labels bilingues FR+EN dans la même hypothèse
    Impact : speech acts corrects sur messages français

B3. TCN 769d → 782d
    +8d speech acts (axis_scores)
    +3d marqueurs discursifs
    +1d longueur normalisée
    +1d author_change
    Re-entraîner, évaluer

B4. Mesurer la contribution de chaque groupe de features
    via ablation (retirer un groupe, mesurer la chute de prec/F1 TCN)
```

### Phase C — Appris plutôt qu'Optuna (moyen terme)

```
C1. MLP AttachScore appris
    Input  : vecteur de 7 features (Sem, Ent, Temp, Goal, Age, Act, Marker)
    Target : paires (message, épisode) positives/négatives depuis gold
    Loss   : BCE ou margin ranking
    Remplace : α,β,γ,δ,ρ fixes + Optuna

C2. Retrieval épisodes dormants
    Problème : identifier si m_t réactive un épisode dormant parmi {E_1..E_k}
    Approche : bi-encoder fine-tuné sur paires (message_réactivation, épisode_passé)
    Données  : annoter les cas de réactivation dans gold tune
```

### Phase D — Mémoire longue terme (vision finale)

```
D1. Episode store avec retrieval vectoriel (FAISS)
    Chaque épisode clôturé → embedding de résumé → index FAISS
    Réactivation = ANN search + seuil

D2. Résumé automatique d'épisode (LLM petit : Phi-3 mini ou Mistral 7B)
    Titre + participants + outcome + entités clés
    Entrée dans le Memory Engine (Ivias)
```

---

## 8. Structure du code

```
src/
  models.py                 — Artifact, Episode, EpisodeState
  parsers/
    whatsapp_parser.py      — Parser WhatsApp → List[Artifact]
  episode_algorithm_fast.py — EpisodeSegmenterFast (Stage 2+3)
  episode_segmenter_hybrid.py — HybridEpisodeSegmenter (Stage 1+2+3)
  boundary_detector_tcn.py  — TCNBoundaryDetector (Stage 1)
  episode_splitter_fast.py  — EpisodeSplitterFast (Stage 3a)
  episode_merger.py         — EpisodeMerger (Stage 3b)
  episode_resegmenter_fast.py — EpisodeResegmenterFast (Stage 3c)
  entity_extractor.py       — GLiNER + spaCy backends
  goal_heuristics.py        — DeBERTa speech acts → goal_vector + axis_scores
  context_embedder.py       — Fenêtrage contextuel (testé, régression, non utilisé)
  nsp_coherence.py          — NSP BERT (testé, abandonné)

colab/
  01_eval_ari.ipynb         — Évaluation ARI complète (tune + test)
  02_train_boundary_detector.ipynb — Entraînement TCN Stage 1
  04_finetune_biencoder.ipynb — Fine-tuning mE5 (testé, abandonné)
  05_enrich_artifacts.ipynb — GLiNER entities + DeBERTa goal vectors
```

---

## 9. Décisions architecturales clés

| Décision | Justification |
|---|---|
| Embeddings isolés (pas contextuels) | H2.b contextuel → régression. mE5 seul = optimal sur ce dataset |
| mE5-base (pas fine-tuné) | H2.a contrastif → -0.096 ARI. Espace générique = meilleur pour Stage 2 |
| TCN sur fenêtre (pas Transformer) | Dataset trop petit pour Transformer. TCN = bon compromis séquentiel |
| Protocole B+C anti-leak | tune_early (65%) → TCN · tune_late (35%) → Optuna · test = score final |
| Optuna CMA-ES (pas grid search) | 160 trials > grid sur espace continu multi-paramètre |
| mDeBERTa-v3 XNLI (pas DeBERTa-small) | FR+EN mixed → cross-lingual NLI indispensable |
| Regex bilingues (pas détection langue) | Code-switching dans le message → détection par message fragile |
