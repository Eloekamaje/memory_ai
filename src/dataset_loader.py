"""
dataset_loader.py — Façade unifiée de chargement de données.

Détecte le type de fichier et appelle le bon parser.
Retourne toujours une List[Artifact] triée par timestamp.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from typing import List
from models import Artifact


def load_dataset(path: str) -> List[Artifact]:
    """
    Charge un dataset depuis un fichier et retourne une liste d'Artifact.

    Formats supportés :
      - .csv → experiment_csv_parser
      - .txt → whatsapp_parser
    """

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        from parsers.experiment_csv_parser import parse_experiment_csv
        artifacts = parse_experiment_csv(path)

    elif ext == ".txt":
        from parsers.whatsapp_parser import parse_whatsapp_chat
        artifacts = parse_whatsapp_chat(path)

    else:
        raise ValueError(f"Format non supporté : {ext}")

    # tri par timestamp
    artifacts.sort(key=lambda a: a.timestamp)

    return artifacts