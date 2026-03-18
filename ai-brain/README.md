# AI Brain — Système de travail Memory AI Lab

Système de prompts et de docs pour travailler comme une équipe AI sur le Memory Engine.

## Structure

```
ai-brain/
├── README.md                    ← ce fichier
├── 01_memory_research.md        ← état de l'art, positionnement
├── 02_segmentation_challenges.md ← failles, robustification
├── 03_memory_ideas.md           ← idées différenciantes
├── 04_memory_architecture.md    ← décisions d'architecture, B4-B6
├── 05_memory_experiments.md     ← log des expériences et résultats
├── 06_agentic_product.md        ← vision produit, MVP
├── 07_visionary_hypotheses.md   ← hypothèses ambitieuses, vision long terme
├── 08_multi_channel_lifecycle.md ← AI memory multi-canal, cycle de vie épisodes, Decision Engine
└── prompts/
    ├── memory_research.txt      ← chercheur état de l'art
    ├── segmentation_challenger.txt ← reviewer critique brutal
    ├── memory_ideas.txt         ← inventeur d'idées
    ├── memory_architect.txt     ← architecte B4-B6
    ├── memory_ml_engineer.txt   ← ingénieur ML implémentation
    ├── memory_experiment.txt    ← expert expérimentation
    └── agentic_product.txt      ← expert produit SaaS
```

## Workflow par situation

### Tu travailles sur la segmentation
```
segmentation_challenger.txt → memory_ml_engineer.txt → memory_experiment.txt
```

### Tu veux avancer vers l'agentic (B4-B6)
```
memory_ideas.txt → memory_architect.txt → agentic_product.txt
```

### Tu bloques
```
segmentation_challenger.txt
```

### Tu veux te repositionner dans l'état de l'art
```
memory_research.txt
```

## Mini-cycle par feature

```
Idea → Challenger → Architecture → Implementation → Experiment → Product value
```

## Règles d'utilisation

1. **Copie le prompt** du fichier `.txt` correspondant
2. **Complète les placeholders** `[REMPLACE PAR ...]` si présents
3. **Colle dans Claude** (ou autre LLM)
4. **Écris les résultats** dans le `.md` correspondant
5. **Mets à jour** les décisions dans `04_memory_architecture.md`

## État du projet (mis à jour manuellement)

| Phase | Statut | ARI / Métrique |
|-------|--------|---------------|
| B1 Entity Resolution | ✅ fait | entity_resolver.py |
| B2 Memory Graph | ✅ fait | memory_graph.py |
| B3 Memory Recall | ✅ fait | recall_engine.py — ARI 0.69 |
| B4 Aging & Compression | 🔲 à faire | — |
| B5 Index & Performance | 🔲 à faire | — |
| B6 API/CLI + LLM | 🔲 à faire | — |
