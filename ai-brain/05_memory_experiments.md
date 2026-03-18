# Memory Experiments — Log des expériences

> Utilise le prompt `prompts/memory_experiment.txt` pour planifier une nouvelle expérience.

## Résultats de référence

| Version | ARI | Notes |
|---------|-----|-------|
| V1 | 0.31 | baseline |
| V2 | 0.66 | entity resolution + goal heuristics |
| V2+ | 0.69 | episode splitter + grid search |
| V3 | — | prochaine cible |

## Expériences planifiées

| # | Feature testée | Dataset | Protocole | Métriques | Statut |
|---|---------------|---------|-----------|-----------|--------|
| 1 | | | | | planifié |

## Expériences terminées

| # | Feature | Résultat clé | Décision prise |
|---|---------|-------------|----------------|
| E1 | segmentation V1 | ARI 0.31 | baseline établie |
| E2 | entity + goals | ARI 0.66 | intégration validée |
| E3 | splitter + grid | ARI 0.69 | paramètres figés |

## Biais identifiés

- [ ] À compléter au fil des expériences
