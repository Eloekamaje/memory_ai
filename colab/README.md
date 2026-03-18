# Memory AI Lab — Colab Setup

## Workflow

```
VSCode local (toi + Claude)        Google Colab (GPU)
        │                                  │
        ├── code + algo                    ├── embeddings (~10 sec GPU)
        ├── annotation tools              ├── eval ARI
        └── gold dataset                  └── grid search params
                    │                          │
                    └──── Google Drive ────────┘
```

## 1. Uploader le projet sur Google Drive

Dans Drive, créer cette structure :
```
Mon Drive/
  memory_ai_lab/
    src/          ← copier tout le dossier src/
    data/
      group_anon.txt
      group_gold.json
    colab/        ← ce dossier
```

## 2. Ouvrir un notebook dans Colab

1. Aller sur [colab.research.google.com](https://colab.research.google.com)
2. `Fichier > Ouvrir > Google Drive` → sélectionner `colab/01_eval_ari.ipynb`
3. Activer le GPU : `Exécution > Modifier le type d'exécution > GPU T4`
4. Exécuter les cellules dans l'ordre

## 3. Après le premier run

Le fichier `data/group_embeddings.npy` est créé sur Drive.
Les runs suivants (grid search, etc.) rechargent le cache → **instantané**.

## Notebooks disponibles

| Notebook | Objectif |
|---|---|
| `01_eval_ari.ipynb` | Évaluation ARI V3 sur le gold dataset |
| *(à venir)* `02_grid_search.ipynb` | Optimisation des paramètres |
| *(à venir)* `03_labse_swap.ipynb` | Test LaBSE vs all-MiniLM |

## Ce qui reste local (VSCode + Claude)

- Écriture du code (`src/`, `tools/`)
- Annotation LLM (`tools/llm_annotator.py`)
- Documentation (`ai-brain/`)
- Tout ce qui ne nécessite pas de GPU
