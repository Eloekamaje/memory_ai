"""
boundary_detector.py — Détecteur de frontières d'épisodes (Stage 1).

MLP léger entraîné sur les embeddings et les gaps temporels.
Prédit P(frontière) pour chaque message — O(n) parallèle sur GPU.

Rôle dans l'architecture hybride :
    Stage 1 (ce fichier) : "est-ce que le topic change ici ?"
    Stage 2 (AttachScore) : "nouvel épisode ou réactivation ?"
    Stage 3 (Lifecycle)   : EMA, DORMANT, ACTIVE

Entraînement : ~2 min CPU, ~20 sec GPU sur le gold tune.
Inference    : O(n) batch — toutes les frontières en parallèle.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional

from models import Artifact


# ================================================================== #
# Features                                                             #
# ================================================================== #

FEATURE_DIM = 770  # 768 (diff) + 1 (sim) + 1 (gap_log)


def extract_features(embeddings: np.ndarray,
                     artifacts: List[Artifact]) -> np.ndarray:
    """
    Extrait les features de frontière pour tous les messages.

    Features par message t :
      diff    = emb_t - emb_{t-1}             (768,) direction du changement
      sim     = cosine_sim(emb_{t-1}, emb_t)  (1,)   continuité sémantique
      gap_log = log1p(gap_min) / log1p(1440)  (1,)   gap normalisé par 24h

    Returns
    -------
    X : np.ndarray (n, 770)
    """
    n, d = embeddings.shape
    X = np.zeros((n, d + 2), dtype=np.float32)

    # Norme pour cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_norm = embeddings / norms

    for i in range(n):
        if i == 0:
            # Premier message : pas de prédécesseur
            diff    = np.zeros(d, dtype=np.float32)
            sim     = 0.0
            gap_log = 1.0  # gap maximal par convention
        else:
            diff    = (embeddings[i] - embeddings[i - 1]).astype(np.float32)
            sim     = float(np.dot(emb_norm[i], emb_norm[i - 1]))
            gap_min = abs(
                (artifacts[i].timestamp - artifacts[i - 1].timestamp
                 ).total_seconds() / 60.0
            )
            gap_log = float(np.log1p(gap_min) / np.log1p(1440.0))

        X[i, :d]   = diff
        X[i, d]    = sim
        X[i, d + 1] = gap_log

    return X


def extract_labels(y_true: List[Optional[int]], n: int) -> np.ndarray:
    """
    Extrait les labels binaires de frontière depuis y_true.

    Un message est une frontière si son episode_id diffère du précédent.

    Returns
    -------
    y : np.ndarray (n,) float32 — 1.0 = frontière, 0.0 = continuation
    """
    y = np.zeros(n, dtype=np.float32)
    prev_ep = None
    for i in range(n):
        ep = y_true[i]
        if ep is not None and ep != prev_ep and prev_ep is not None:
            y[i] = 1.0
        if ep is not None:
            prev_ep = ep
    return y


# ================================================================== #
# Modèle                                                               #
# ================================================================== #

class BoundaryMLP(nn.Module):
    """
    MLP léger pour la détection de frontières.

    Architecture : 770 → 128 → ReLU → Dropout → 32 → ReLU → 1
    Sortie       : logit non-sigmoïde (utiliser sigmoid pour la probabilité)
    """

    def __init__(self, input_dim: int = FEATURE_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,)


# ================================================================== #
# Entraînement                                                         #
# ================================================================== #

class BoundaryDetector:
    """
    Interface haut niveau : entraînement + inference du boundary detector.

    Usage
    -----
    detector = BoundaryDetector(device='cuda')
    detector.fit(X_tune, y_tune, n_epochs=30)
    probs = detector.predict_proba(X)        # (n,) float — P(frontière)
    boundaries = detector.predict(X)         # (n,) bool
    """

    def __init__(self, device: Optional[str] = None,
                 threshold: float = 0.5):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device    = device
        self.threshold = threshold
        self.model     = BoundaryMLP().to(device)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            n_epochs: int = 30,
            lr: float = 1e-3,
            batch_size: int = 512,
            verbose: bool = True) -> 'BoundaryDetector':
        """
        Entraîne le MLP sur les features et labels extraits.

        Gère le déséquilibre de classes via pos_weight dans BCEWithLogitsLoss.
        """
        n_pos   = int(y.sum())
        n_neg   = len(y) - n_pos
        pos_w   = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32,
                               device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        x_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)

        self.model.train()
        for epoch in range(n_epochs):
            # Mini-batch shuffle
            perm   = torch.randperm(len(x_t), device=self.device)
            losses = []
            for start in range(0, len(x_t), batch_size):
                idx    = perm[start:start + batch_size]
                logits = self.model(x_t[idx])
                loss   = criterion(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                mean_loss = np.mean(losses)
                print(f'  Epoch {epoch + 1:3d}/{n_epochs} | loss={mean_loss:.4f}')

        self.model.eval()
        return self

    def optimize_threshold(self,
                           X: np.ndarray,
                           y: np.ndarray) -> float:
        """
        Trouve le seuil optimal (max F1) sur un ensemble de validation.
        Appeler sur le TUNE — jamais sur le TEST.
        """
        probs = self.predict_proba(X)
        best_f1, best_thr = 0.0, 0.5

        for thr in np.arange(0.1, 0.9, 0.05):
            preds   = (probs >= thr).astype(int)
            tp      = int(((preds == 1) & (y == 1)).sum())
            fp      = int(((preds == 1) & (y == 0)).sum())
            fn      = int(((preds == 0) & (y == 1)).sum())
            prec    = tp / (tp + fp + 1e-8)
            rec     = tp / (tp + fn + 1e-8)
            f1      = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        self.threshold = best_thr
        print(f'  Seuil optimal : {best_thr:.2f} (F1={best_f1:.4f})')
        return best_thr

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne P(frontière) pour chaque message. Shape (n,)."""
        self.model.eval()
        x_t    = torch.from_numpy(X).to(self.device)
        logits = self.model(x_t)
        return torch.sigmoid(logits).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne les indices des frontières détectées."""
        probs = self.predict_proba(X)
        return np.nonzero(probs >= self.threshold)[0]

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            'state_dict': self.model.state_dict(),
            'threshold':  self.threshold,
        }, path)
        print(f'✓ Boundary detector sauvegardé → {path}')

    def load(self, path: str) -> 'BoundaryDetector':
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.threshold = ckpt['threshold']
        self.model.eval()
        print(f'✓ Boundary detector chargé depuis {path}')
        return self
