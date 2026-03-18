# Memory AI Lab — Workflow Colab

## Architecture du projet

```
GitHub (code)                    Google Drive (données privées)
  Eloekamaje/memory_ai              memory_ai_data/
  ├── src/                            ├── group_anon.txt
  ├── tools/                          ├── group_gold.json
  ├── ai-brain/                       └── group_embeddings.npy  ← généré auto
  ├── colab/
  └── requirements_colab.txt
              │                                    │
              └──────────── Colab ─────────────────┘
                       !git clone + drive.mount
```

## Workflow quotidien

### Dev local (VSCode + Claude)
```bash
# Modifier le code
# ...
git add -A
git commit -m "feat: ..."
git push
```

### Exécution Colab (GPU)
1. Ouvrir le notebook depuis Drive ou GitHub
2. Activer GPU : `Exécution > Modifier le type d'exécution > GPU T4`
3. Run "Tout exécuter" — la cellule 1 fait `git pull` automatiquement

## Upload des données sur Drive (une seule fois)

Dans Google Drive, créer le dossier `memory_ai_data/` et y uploader :
- `data/group_anon.txt`
- `data/group_gold.json`

Le fichier `group_embeddings.npy` sera généré automatiquement au premier run.

## Notebooks

| Notebook | Objectif |
|---|---|
| `01_eval_ari.ipynb` | Évaluation ARI V3 sur le gold dataset |
| *(à venir)* `02_labse_swap.ipynb` | Swap all-MiniLM → LaBSE (V4 plan) |
| *(à venir)* `03_gliner_ner.ipynb` | NER GLiNER vs spaCy |

## Règles

- **Code** → GitHub (public, pas de données sensibles)
- **Données** → Drive uniquement (`.gitignore` les exclut)
- **Embeddings** → Drive (cache, généré par Colab)
- **Résultats** → noter dans `README.md` et pousser sur GitHub
