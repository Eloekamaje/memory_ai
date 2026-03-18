# Segmentation — Failles & Challenges

> Utilise le prompt `prompts/segmentation_challenger.txt` pour peupler cette section.

## Algorithme actuel

```
AttachScore = α·Sem + β·Ent + γ·Temp + δ·Goal - ρ·Age
```

Modules : `episode_algorithm.py`, `episode_splitter.py`, `entity_resolver.py`, `goal_heuristics.py`

Métriques actuelles : ARI 0.69 (V2+)

## Failles identifiées

| # | Faille | Sévérité | Module impacté | Fix envisagé |
|---|--------|----------|----------------|--------------|
| 1 | Reactivation failure | | | |
| 2 | Topic drift progressif | | | |
| 3 | Ambiguïté de frontière | | | |
| ... | À compléter | | | |

## Cas de rupture en production

- [ ] À compléter via le prompt segmentation_challenger.txt

## Plan de robustification

- [ ] À compléter via le prompt segmentation_challenger.txt
