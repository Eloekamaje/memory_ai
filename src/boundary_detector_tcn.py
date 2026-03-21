"""
boundary_detector_tcn.py — TCN Boundary Detector (Stage 1 v2).

Architecture inspirée de TransNet V2, adaptée aux séquences d'embeddings textuels.

Principe clé (TransNet → texte) :
    Vidéo  : [batch, 100 frames,  27, 48, 3]  → 3D Conv (spatial + temporel)
    Texte  : [batch, 100 msgs,    d_emb + 1]  → 1D Conv dilated (temporel)

    Chaque position prédit P(frontière_t) en voyant 50 messages avant ET après.
    Les convolutions dilatées (1, 2, 4, 8) capturent des contextes multi-échelle
    sans exploser les paramètres → champ réceptif >> 2 msgs du MLP actuel.

Architecture V1 (conservative — dataset ~9K msgs) :
    Projection  : (d_emb+1) → 128
    3× DilatedBlock : 4 branches dilation(1,2,4,8), concat, fusion 1×1, résidu
    Head        : 128 → 64 → 1 (sigmoid)
    Total       : ~320K paramètres

Interface compatible BoundaryDetector :
    - predict_proba_sequence(embeddings, artifacts) → (n,) float — nouvelle API
    - predict_proba(X)                              → (n,) float — compat MLP
    - fit_sequence / save / load / optimize_threshold

Références :
    Souček & Lokoč (2020). TransNet V2. arXiv:2008.04838
    Bai et al. (2018). An Empirical Evaluation of Generic CNNs vs RNNs. arXiv:1803.01271
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from models import Artifact


# ================================================================== #
# Features (séquence)                                                 #
# ================================================================== #

INPUT_DIM = 769  # 768 (embedding) + 1 (log1p gap normalisé)


def _sequence_features(embeddings: np.ndarray,
                        artifacts: List[Artifact]) -> np.ndarray:
    """
    Prépare les features par message pour le TCN.

    Features par message t :
        emb_t    (768,) : embedding brut — le réseau apprend lui-même le contexte
        gap_log  (1,)   : log1p(gap_min) / log1p(1440), normalisé par 24h

    Pas de cos_sim pré-calculé — éviter le raccourci "faible sim = frontière"
    que le MLP actuel a appris (et qui plafonne la précision à 0.47).

    Returns
    -------
    X : np.ndarray (n, 769)
    """
    n, d = embeddings.shape
    X = np.zeros((n, d + 1), dtype=np.float32)
    X[:, :d] = embeddings.astype(np.float32)
    for i in range(n):
        if i == 0:
            X[i, d] = 1.0  # premier message : gap maximal par convention
        else:
            gap_min = abs(
                (artifacts[i].timestamp - artifacts[i - 1].timestamp
                 ).total_seconds() / 60.0
            )
            X[i, d] = float(np.log1p(gap_min) / np.log1p(1440.0))
    return X


def extract_boundary_labels(y_true: List[Optional[str]], n: int) -> np.ndarray:
    """
    Extrait les labels binaires de frontière depuis y_true.

    Label = 1 au PREMIER message d'un nouvel épisode (pas au dernier du précédent).

    Returns
    -------
    y : np.ndarray (n,) float32
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
# Modèle TCN                                                          #
# ================================================================== #

class DilatedBlock(nn.Module):
    """
    Bloc résiduel à convolutions dilatées multi-échelle.

    4 branches parallèles (dilation 1, 2, 4, 8) → concat → fusion 1×1
    → résidu + LayerNorm (post-norm).

    Champ réceptif par bloc : kernel=3, dilations 1..8
        dilation=1 : 3 messages
        dilation=2 : 5 messages
        dilation=4 : 9 messages
        dilation=8 : 17 messages
    → un bloc voit jusqu'à 17 messages de contexte.
    3 blocs empilés avec résidus → couverture effective ~40-50 messages.
    """

    DILATIONS = (1, 2, 4, 8)

    def __init__(self, channels: int, kernel_size: int = 3,
                 dropout: float = 0.15):
        super().__init__()
        branch_ch = channels // len(self.DILATIONS)  # 32 si channels=128

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    channels, branch_ch, kernel_size,
                    padding=d * (kernel_size // 2), dilation=d,
                    bias=False,
                ),
                nn.GELU(),
            )
            for d in self.DILATIONS
        ])

        # Fusion 1×1 : remet à channels pour le résidu
        self.fusion = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm   = nn.LayerNorm(channels)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, length, channels)
        h = x.permute(0, 2, 1)                            # → (B, C, L)
        h = torch.cat([b(h) for b in self.branches], dim=1)  # → (B, C, L)
        h = self.fusion(h)                                 # → (B, C, L)
        h = h.permute(0, 2, 1)                            # → (B, L, C)
        h = self.drop(h)
        return self.norm(x + h)                            # résidu + norm


class TCNModel(nn.Module):
    """
    Backbone TCN pour la détection de frontières par position.

    Input  : (batch, window, d_input=769)
    Output : (batch, window) — logits non-signés par position
    """

    def __init__(self,
                 d_input:    int = INPUT_DIM,
                 channels:   int = 128,
                 n_blocks:   int = 3,
                 kernel_size: int = 3,
                 dropout:    float = 0.15):
        super().__init__()
        self.proj   = nn.Linear(d_input, channels)
        self.blocks = nn.ModuleList([
            DilatedBlock(channels, kernel_size, dropout)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, d_input)
        h = self.proj(x)          # (B, L, channels)
        for block in self.blocks:
            h = block(h)           # (B, L, channels)
        return self.head(h).squeeze(-1)  # (B, L) logits


# ================================================================== #
# Focal Loss                                                          #
# ================================================================== #

class FocalLoss(nn.Module):
    """
    Focal Loss pour classification déséquilibrée.

    FL(p_t) = -(1 - p_t)^γ · log(p_t)

    γ=2 est le choix standard (Lin et al., 2017).
    pos_weight scale le gradient sur les positifs (comme BCE pos_weight).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # poids asymétrique : les positifs comptent pos_weight× plus
        weight = targets * self.pos_weight + (1 - targets)
        bce    = F.binary_cross_entropy_with_logits(logits, targets,
                                                     reduction='none')
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1 - probs) * (1 - targets)
        loss   = weight * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ================================================================== #
# Fenêtres d'entraînement                                             #
# ================================================================== #

def make_windows(features: np.ndarray,
                 labels:   np.ndarray,
                 window_size: int = 100,
                 stride_min:  int = 10,
                 stride_max:  int = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Découpe la séquence en fenêtres chevauchantes avec stride aléatoire
    pour l'augmentation de données.

    features : (n, d_input)
    labels   : (n,) float32

    Returns
    -------
    X_wins : (n_windows, window_size, d_input)
    y_wins : (n_windows, window_size)
    """
    n = len(features)
    stride = random.randint(stride_min, stride_max)
    starts = list(range(0, max(1, n - window_size + 1), stride))
    if not starts:
        starts = [0]

    X_wins, y_wins = [], []
    for s in starts:
        e = s + window_size
        win_x = np.zeros((window_size, features.shape[1]), dtype=np.float32)
        win_y = np.zeros(window_size, dtype=np.float32)
        actual = min(window_size, n - s)
        win_x[:actual] = features[s:e]
        win_y[:actual] = labels[s:e]
        X_wins.append(win_x)
        y_wins.append(win_y)

    return np.stack(X_wins), np.stack(y_wins)


# ================================================================== #
# Détecteur haut-niveau                                               #
# ================================================================== #

class TCNBoundaryDetector:
    """
    Interface haut-niveau du TCN Boundary Detector.

    Compatible avec HybridEpisodeSegmenter via predict_proba_sequence().

    Usage typique
    -------------
    det = TCNBoundaryDetector(device='cuda')
    det.fit_sequence(embeddings_tune, artifacts_tune, y_true_tune,
                     n_epochs=50, window_size=100)
    det.optimize_threshold_sequence(embeddings_val, artifacts_val, y_true_val)
    probs = det.predict_proba_sequence(embeddings_test, artifacts_test)
    det.save('boundary_detector_tcn.pt')
    """

    def __init__(self,
                 device:      Optional[str] = None,
                 threshold:   float = 0.5,
                 channels:    int   = 128,
                 n_blocks:    int   = 3,
                 kernel_size: int   = 3,
                 dropout:     float = 0.15,
                 window_size: int   = 100):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device      = device
        self.threshold   = threshold
        self.window_size = window_size
        self.model = TCNModel(
            d_input=INPUT_DIM,
            channels=channels,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(device)

    # ------------------------------------------------------------------ #
    # Entraînement                                                         #
    # ------------------------------------------------------------------ #

    def fit_sequence(self,
                     embeddings:  np.ndarray,
                     artifacts:   List[Artifact],
                     y_true:      list,
                     n_epochs:    int   = 50,
                     lr:          float = 1e-3,
                     weight_decay: float = 1e-4,
                     batch_size:  int   = 32,
                     focal_gamma: float = 2.0,
                     stride_min:  int   = 10,
                     stride_max:  int   = 25,
                     val_split:   float = 0.1,
                     verbose:     bool  = True) -> 'TCNBoundaryDetector':
        """
        Entraîne le TCN sur la séquence complète.

        Augmentation : stride aléatoire [stride_min, stride_max] à chaque epoch.
        Loss : Focal Loss (γ=2, pos_weight=n_neg/n_pos).
        """
        n = len(embeddings)
        features = _sequence_features(embeddings, artifacts)
        labels   = extract_boundary_labels(y_true, n)

        n_pos = max(int(labels.sum()), 1)
        n_neg = n - n_pos
        pos_weight = n_neg / n_pos
        if verbose:
            print(f'  Positifs : {n_pos} ({100*n_pos/n:.1f}%)  '
                  f'pos_weight={pos_weight:.1f}')

        # Split val
        n_val  = max(1, int(n * val_split))
        n_train = n - n_val
        feat_tr, feat_val = features[:n_train], features[n_train:]
        lbl_tr,  lbl_val  = labels[:n_train],   labels[n_train:]

        criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.05)

        best_val_f1  = 0.0
        best_state   = None

        self.model.train()
        for epoch in range(n_epochs):
            # Augmentation : stride aléatoire
            X_win, y_win = make_windows(
                feat_tr, lbl_tr, self.window_size, stride_min, stride_max)
            idx_shuf = np.random.permutation(len(X_win))

            epoch_loss = []
            for start in range(0, len(X_win), batch_size):
                idx_b = idx_shuf[start:start + batch_size]
                x_t = torch.from_numpy(X_win[idx_b]).to(self.device)
                y_t = torch.from_numpy(y_win[idx_b]).to(self.device)
                logits = self.model(x_t)
                loss   = criterion(logits, y_t)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss.append(loss.item())

            scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                val_f1 = self._eval_f1(feat_val, lbl_val)
                mean_loss = np.mean(epoch_loss)
                print(f'  Epoch {epoch+1:3d}/{n_epochs} | '
                      f'loss={mean_loss:.4f} | val_F1={val_f1:.4f}')
                if val_f1 > best_val_f1:
                    best_val_f1  = val_f1
                    best_state   = {k: v.clone()
                                    for k, v in self.model.state_dict().items()}

        # Restaurer le meilleur état
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f'  ✓ Meilleur modèle restauré (val F1={best_val_f1:.4f})')
        self.model.eval()
        return self

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_proba_sequence(self,
                                embeddings: np.ndarray,
                                artifacts:  List[Artifact]) -> np.ndarray:
        """
        Retourne P(frontière_t) pour chaque message — API principale.

        Utilise une fenêtre glissante avec stride=window//2 et moyenne
        des prédictions sur les zones chevauchantes.

        Returns
        -------
        probs : np.ndarray (n,) float32
        """
        self.model.eval()
        n = len(embeddings)
        if n == 0:
            return np.array([], dtype=np.float32)

        features = _sequence_features(embeddings, artifacts)  # (n, 769)
        W        = self.window_size
        half     = W // 2

        scores = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)

        stride = half
        for start in range(0, n, stride):
            end      = min(start + W, n)
            actual   = end - start

            chunk = np.zeros((W, features.shape[1]), dtype=np.float32)
            chunk[:actual] = features[start:end]

            x_t    = torch.from_numpy(chunk[None]).to(self.device)  # (1, W, 769)
            logits = self.model(x_t)                                 # (1, W)
            probs  = torch.sigmoid(logits)[0].cpu().numpy()         # (W,)

            scores[start:end] += probs[:actual]
            counts[start:end] += 1.0

        return scores / np.maximum(counts, 1.0)

    # ------------------------------------------------------------------ #
    # Calibration du seuil                                                 #
    # ------------------------------------------------------------------ #

    def optimize_threshold_sequence(self,
                                     embeddings: np.ndarray,
                                     artifacts:  List[Artifact],
                                     y_true:     list,
                                     min_recall: float = 0.80) -> float:
        """
        Trouve le seuil qui maximise F1 sous contrainte recall ≥ min_recall.
        Appeler sur TUNE_EARLY — jamais sur TEST.
        """
        n      = len(embeddings)
        labels = extract_boundary_labels(y_true, n)
        probs  = self.predict_proba_sequence(embeddings, artifacts)

        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.05, 0.90, 0.025):
            preds = (probs >= thr).astype(int)
            tp    = int(((preds == 1) & (labels == 1)).sum())
            fp    = int(((preds == 1) & (labels == 0)).sum())
            fn    = int(((preds == 0) & (labels == 1)).sum())
            rec   = tp / (tp + fn + 1e-8)
            prec  = tp / (tp + fp + 1e-8)
            f1    = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1 and rec >= min_recall:
                best_f1, best_thr = f1, thr

        self.threshold = best_thr
        n_pos = int(labels.sum())
        preds  = (probs >= best_thr).astype(int)
        tp     = int(((preds == 1) & (labels == 1)).sum())
        fp     = int(((preds == 1) & (labels == 0)).sum())
        fn     = int(((preds == 0) & (labels == 1)).sum())
        rec    = tp / (tp + fn + 1e-8)
        prec   = tp / (tp + fp + 1e-8)
        print(f'  Seuil optimal : {best_thr:.3f} | '
              f'F1={best_f1:.4f} prec={prec:.4f} rec={rec:.4f} | '
              f'pred={tp+fp}  gold={n_pos}')
        return best_thr

    # ------------------------------------------------------------------ #
    # Evaluation interne                                                   #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _eval_f1(self, features: np.ndarray, labels: np.ndarray) -> float:
        """F1 sur la séquence val (sans sliding window — direct)."""
        self.model.eval()
        W = self.window_size
        n = len(features)

        chunk = np.zeros((W, features.shape[1]), dtype=np.float32)
        chunk[:n] = features
        x_t    = torch.from_numpy(chunk[None]).to(self.device)
        logits = self.model(x_t)[0].cpu().numpy()[:n]
        probs  = 1 / (1 + np.exp(-logits))
        preds  = (probs >= 0.5).astype(int)
        tp     = int(((preds == 1) & (labels == 1)).sum())
        fp     = int(((preds == 1) & (labels == 0)).sum())
        fn     = int(((preds == 0) & (labels == 1)).sum())
        prec   = tp / (tp + fp + 1e-8)
        rec    = tp / (tp + fn + 1e-8)
        self.model.train()
        return 2 * prec * rec / (prec + rec + 1e-8)

    # ------------------------------------------------------------------ #
    # Persistance                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save({
            'state_dict':  self.model.state_dict(),
            'threshold':   self.threshold,
            'window_size': self.window_size,
            'model_cfg': {
                'd_input':    INPUT_DIM,
                'channels':   self.model.proj.out_features,
                'n_blocks':   len(self.model.blocks),
                'kernel_size': self.model.blocks[0].branches[0][0].kernel_size[0],
                'dropout':     self.model.blocks[0].drop.p,
            },
        }, path)
        print(f'✓ TCN boundary detector sauvegardé → {path}')

    def load(self, path: str) -> 'TCNBoundaryDetector':
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg  = ckpt.get('model_cfg', {})
        if cfg:
            self.model = TCNModel(**cfg).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.threshold   = ckpt['threshold']
        self.window_size = ckpt.get('window_size', self.window_size)
        self.model.eval()
        print(f'✓ TCN boundary detector chargé (seuil={self.threshold:.3f})')
        return self
