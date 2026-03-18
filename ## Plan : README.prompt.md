## Plan : README.md — Roadmap Memory AI Lab

**TL;DR** — Créer un README complet à la racine. Mon choix d'expert : présenter les deux directions comme **complémentaires et séquencées** — d'abord B (système fonctionnel) puis A (publication). Raison : un système qui tourne donne des insights empiriques qui nourrissent la formalisation théorique. Pas de CHANGELOG/CONTRIBUTING à ce stade — c'est un projet de recherche solo.

### Steps

1. **En-tête + vision** dans README.md : titre, one-liner scientifique (*"Segmentation épisodique multi-critères d'un flux d'artefacts personnels"*), badges (Python 3.10, research), le problème en 5 lignes, le modèle formel $\text{AttachScore} = \alpha \cdot Sem + \beta \cdot Ent + \gamma \cdot Temp + \delta \cdot Goal - \rho \cdot Age$.

2. **Architecture** : arbre du projet annoté (`src/`, `data/`, `notebooks/`), diagramme textuel du pipeline (Artefacts → Parser → Embeddings + Goals → `EpisodeSegmenter` → Episodes → Consolidation → Métriques).

3. **Résultats acquis** : tableau V1→V2→V2+ (ARI 0.31→0.66→0.69), hypothèses H1-H4 avec statuts, couverture du framework théorique (11/19 points couverts), résultats données réelles (38 épisodes, 2879 messages).

4. **Direction B — Système complet** (priorité 1, ~6 phases) :
   - **Phase B1** : Entity Resolution — clustering probabiliste des mentions, index inversé mention→entité canonique
   - **Phase B2** : Graphe de mémoire — structure `MemoryGraph` (épisodes ↔ entités ↔ relations), stockage persistant
   - **Phase B3** : Memory Recall — `memory.recall(query)` avec ranking multi-signal (sémantique + temporel + importance), API retrieval
   - **Phase B4** : Aging & compression — courbe d'oubli, résumés automatiques, hiérarchie épisode→période→projet
   - **Phase B5** : Index & performance — FAISS/HNSW pour centroïdes, index inversé entités→épisodes, cible O(k log n)
   - **Phase B6** : API/CLI — FastAPI + CLI, intégration LLM comme consumer de la mémoire

5. **Direction A — Publication** (priorité 2, ~5 phases) :
   - **Phase A1** : Soft assignment — $P(a \in e_k)$ probabiliste, résolution différée de l'ambiguïté
   - **Phase A2** : Formulation variationnelle — $\mathcal{L} = \sum Cohesion(e) - \lambda \#Ep - \mu Ambiguity$, approximation EM
   - **Phase A3** : Scission d'épisodes — détection de sous-clusters, split automatique
   - **Phase A4** : Benchmark formel — suite de cas limites (§15 du framework), comparaison TextTiling/BERTopic/TopicTiling
   - **Phase A5** : Rédaction — papier format workshop/conférence, formalisation complète

6. **Section setup** : `python -m venv venv`, `pip install -r requirements.txt` (à créer depuis le venv existant), commandes pour lancer expériences et notebook.

### Considérations (décisions prises)

1. **Séquencement B→A** : Direction B d'abord car un système qui tourne génère les données et insights nécessaires pour la formalisation de A. Les phases B1-B3 sont le socle, A peut démarrer en parallèle dès B3 terminé.
2. **Pas de CONTRIBUTING/CHANGELOG** : projet de recherche solo, on reste lean. On ajoutera si le projet est ouvert.
