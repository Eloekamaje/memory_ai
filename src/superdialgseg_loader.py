"""
superdialgseg_loader.py — Chargement et conversion du dataset SuperDialseg.

Convertit les dialogues SuperDialseg au format compatible avec TCNBoundaryDetector :
    utterances  : List[str]          — texte brut pour l'embedding
    y_true      : List[topic_id]     — IDs de topics (segments gold)
    artifacts   : List[FakeArtifact] — objets avec timestamp synthétique

Spécificités SuperDialseg :
    - Pas de timestamps réels → synthétiques (1 msg/min, reset entre dialogues)
    - Frontière marquée par changement de topic_id (aligné avec extract_boundary_labels)
    - Taux de frontières ~25% (vs ~5% dans nos données) → poids focal loss différent

Usage
-----
from superdialgseg_loader import load_superseg, FakeArtifact

utterances, y_true, artifacts = load_superseg(
    '/path/to/SuperDialseg/superseg/segmentation_file_train.json',
    max_dialogues=None,   # None = tous
)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple


# ================================================================== #
# Artifact synthétique (compatible avec boundary_detector_tcn)        #
# ================================================================== #

@dataclass
class FakeArtifact:
    """
    Artifact minimal compatible avec _sequence_features() du TCN.

    Fournit uniquement timestamp (pour log1p(gap_min)).
    Les autres champs (entities, goal_vector) ne sont pas utilisés
    par le TCN — uniquement par AttachScore de Stage 2.
    """
    timestamp: datetime
    entities: list = field(default_factory=list)
    goal_vector: Optional[object] = None
    text: str = ""


# ================================================================== #
# Chargement JSON                                                      #
# ================================================================== #

def load_superseg(
    json_path: str,
    max_dialogues: Optional[int] = None,
    gap_within_minutes: float = 1.0,
    gap_between_minutes: float = 1440.0,
) -> Tuple[List[str], List[str], List[FakeArtifact]]:
    """
    Charge un fichier SuperDialseg JSON et retourne :
        utterances : List[str]   — texte de chaque message (pour embedding)
        y_true     : List[str]   — topic_id sous forme de string unique
                                   compatible avec extract_boundary_labels()
        artifacts  : List[FakeArtifact]  — timestamps synthétiques

    Parameters
    ----------
    json_path           : chemin vers segmentation_file_train/test.json
    max_dialogues       : limite le nombre de dialogues chargés (None = tous)
    gap_within_minutes  : gap entre messages consécutifs dans un dialogue (1 min)
    gap_between_minutes : gap entre deux dialogues (1440 min = 24h)
                          → le TCN apprend à distinguer les frontières de dialogue
    """
    path = Path(json_path)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    # SuperDialseg JSON structure :
    # { "dial_data": { "<dataset_name>": [ { "dial_id": ..., "turns": [...] } ] } }
    all_dialogues = []
    for dataset_name, dialogues in data['dial_data'].items():
        all_dialogues.extend(dialogues)

    if max_dialogues is not None:
        all_dialogues = all_dialogues[:max_dialogues]

    utterances: List[str]         = []
    y_true:     List[str]         = []
    artifacts:  List[FakeArtifact] = []

    # Temps courant — avance de gap_within entre messages, gap_between entre dialogues
    current_time = datetime(2020, 1, 1, 0, 0, 0)
    dial_offset  = 0  # pour rendre les topic_ids uniques entre dialogues

    for dial in all_dialogues:
        dial_id   = dial.get('dial_id', f'dial_{dial_offset}')
        turns     = dial.get('turns', [])

        if not turns:
            continue

        for turn in turns:
            text     = turn.get('utterance', '').strip()
            topic_id = turn.get('topic_id', 0)

            # topic_id unique global = dial_id + topic_id
            global_topic_id = f"{dial_id}__topic_{topic_id}"

            utterances.append(text)
            y_true.append(global_topic_id)
            artifacts.append(FakeArtifact(timestamp=current_time, text=text))

            current_time += timedelta(minutes=gap_within_minutes)

        # Saut de 24h entre dialogues
        current_time += timedelta(minutes=gap_between_minutes)
        dial_offset  += 1

    return utterances, y_true, artifacts


# ================================================================== #
# Chargement TXT (format alternatif)                                  #
# ================================================================== #

def load_superseg_txt(
    txt_path: str,
    max_dialogues: Optional[int] = None,
    gap_within_minutes: float = 1.0,
    gap_between_minutes: float = 1440.0,
) -> Tuple[List[str], List[str], List[FakeArtifact]]:
    """
    Charge le format TXT de SuperDialseg (plus léger que JSON).

    Format TXT :
        call_id_0
        0 utterance text here
        1 another utterance (topic 1)
        ...
        call_id_1
        ...
    """
    utterances: List[str]          = []
    y_true:     List[str]          = []
    artifacts:  List[FakeArtifact] = []

    current_time = datetime(2020, 1, 1, 0, 0, 0)
    dial_id      = None
    n_dialogues  = 0

    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue

            if line.startswith('call_id_'):
                if max_dialogues and n_dialogues >= max_dialogues:
                    break
                dial_id = line.strip()
                n_dialogues += 1
                current_time += timedelta(minutes=gap_between_minutes)
                continue

            if dial_id is None:
                continue

            parts    = line.split(' ', 1)
            if len(parts) < 2:
                continue
            topic_id = parts[0]
            text     = parts[1].strip()

            global_topic_id = f"{dial_id}__topic_{topic_id}"

            utterances.append(text)
            y_true.append(global_topic_id)
            artifacts.append(FakeArtifact(timestamp=current_time, text=text))
            current_time += timedelta(minutes=gap_within_minutes)

    return utterances, y_true, artifacts


# ================================================================== #
# Statistiques                                                         #
# ================================================================== #

def print_stats(y_true: List[str], label: str = '') -> None:
    """Affiche les statistiques du dataset chargé."""
    from boundary_detector_tcn import extract_boundary_labels
    import numpy as np

    n       = len(y_true)
    labels  = extract_boundary_labels(y_true, n)
    n_pos   = int(labels.sum())
    n_segs  = len(set(y_true))

    print(f'{label}')
    print(f'  Messages   : {n:,}')
    print(f'  Segments   : {n_segs:,}')
    print(f'  Frontières : {n_pos:,}  ({100*n_pos/n:.1f}%)')
    print(f'  Msgs/seg   : {n/max(n_segs,1):.1f}')
