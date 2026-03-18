# Memory Architecture — Décisions & Design

> Utilise le prompt `prompts/memory_architect.txt` pour peupler cette section.

## Architecture actuelle (layers)

```
Layer 1 — Parsing      : conversation_preprocessor.py, parsers/
Layer 2 — Embeddings   : embedding_engine.py
Layer 3 — Goals        : goal_heuristics.py
Layer 4 — Entities     : entity_extractor.py → entity_resolver.py
Layer 5 — Segmentation : episode_algorithm.py (AttachScore), episode_splitter.py
Layer 6 — Summaries    : episode_summarizer.py
Layer 7 — Graph        : memory_graph.py (épisodes ↔ entités)
Layer 8 — Recall       : recall_engine.py (retrieval multi-signal)
Layer 9 — Eval         : metrics.py, experiment_runner.py, grid_search.py
```

## Phases B4-B6 à architecturer

### B4 — Aging & Compression
- Objectif : courbe d'oubli, hiérarchie épisode→période→projet
- Décisions : À compléter
- Risques : À compléter

### B5 — Index & Performance
- Objectif : FAISS/HNSW pour centroïdes, O(k log n)
- Décisions : À compléter
- Risques : À compléter

### B6 — API/CLI + LLM consumer
- Objectif : FastAPI, CLI, LLM qui interroge recall_engine.py
- Format contexte LLM : À compléter
- Stratégie compression : À compléter

## Décisions architecturales prises

| Décision | Raison | Date |
|----------|--------|------|
| Séquencement B→A | Un système qui tourne génère les insights pour la formalisation | — |
