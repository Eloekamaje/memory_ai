"""
boundary_detector_transformer.py — Transformer Boundary Detector (Stage 1 v3).

Avantage clé vs TCN :
    TCN  : champ réceptif ~42 positions (3 blocs, dilations 1-8, kernel=3)
    TFM  : self-attention globale → champ = toute la fenêtre (100 msgs)

    Pour détecter les ruptures de sujet longue portée (bifurcations, retours
    de topic), le contexte global est critique — le TCN est structurellement
    limité ici.

Architecture v1 :
    Projection  : Linear(769 → d_model=128)   — 768 emb + gap_log
    2× TransformerEncoderLayer : 4 heads, FFN 4×d_model, pre-LN, dropout=0.15
    Head        : d_model → 32 → 1 (sigmoid via BCEWithLogits)
    Total       : ~500K paramètres

Nouvelles features (issues de l'évaluation qualitative par Claude) :
    msg_length_log : log1p(len(content)) / log1p(500)
        → capte la verbosité : débat=long, partage médias=court
    is_media : 1 si URL détectée ou message < 15 chars
        → cible les faux positifs dans les phases de partage de liens

Interface identique à TCNBoundaryDetector :
    - fit_sequence(embeddings, artifacts, y_true, ...)
    - predict_proba_sequence(embeddings, artifacts) → (n,) float
    - optimize_threshold_sequence(embeddings, artifacts, y_true, min_recall)
    - save(path) / load(path)

Sauvegarde : boundary_detector_tfm.pt (ne touche pas boundary_detector_tcn.pt)

Références :
    Vaswani et al. (2017). Attention is All You Need. NeurIPS.
    Zhang et al. (2021). DialSeg — dialogue segmentation via Transformer.
"""

from __future__ import annotations

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from models import Artifact

# Réutiliser les helpers du TCN (évite duplication)
from boundary_detector_tcn import (
    FocalLoss,
    make_windows,
    extract_boundary_labels,
    _sequence_features,  # 769d : 768 emb + gap_log
    INPUT_DIM,           # 769
)

# ================================================================== #
# Features discursives (v2)                                           #
# ================================================================== #

INPUT_DIM_V2 = 771   # 768 emb + gap_log + msg_length_log + is_media

_URL_RE = re.compile(r'https?://|www\.|bit\.ly|youtu\.be|fb\.com|tinyurl', re.IGNORECASE)


def _sequence_features_v2(embeddings: np.ndarray,
                           artifacts:  List[Artifact]) -> np.ndarray:
    """
    Features étendues pour le Transformer (771d).

    Features par message t :
        emb_t          (768,) : embedding mE5
        gap_log        (  1,) : log1p(gap_min) / log1p(1440) — rupture temporelle
        msg_length_log (  1,) : log1p(len(content)) / log1p(500) — verbosité
        is_media       (  1,) : 1 si URL ou message < 15 chars — partage passif

    Motivation (évaluation qualitative Claude, 22/03/2026) :
        FP systématiques dans les phases de partage de médias :
            → msg_length_log ≈ 0, is_media = 1
        FN systématiques aux ruptures de registre (passif → actif) :
            → msg_length_log augmente brusquement après la frontière
    """
    n, d = embeddings.shape
    X = np.zeros((n, INPUT_DIM_V2), dtype=np.float32)
    X[:, :d] = embeddings.astype(np.float32)

    for i in range(n):
        content = artifacts[i].content or ''

        # gap_log
        if i == 0:
            X[i, d] = 1.0
        else:
            t1 = artifacts[i - 1].timestamp
            t2 = artifacts[i].timestamp
            if t1 is not None and t2 is not None:
                gap_min = abs((t2 - t1).total_seconds() / 60.0)
                X[i, d] = float(np.log1p(gap_min) / np.log1p(1440.0))

        # msg_length_log : verbosité normalisée (log car distribution très skewed)
        X[i, d + 1] = float(np.log1p(len(content)) / np.log1p(500.0))

        # is_media : lien URL ou message très court
        is_url   = bool(_URL_RE.search(content))
        is_short = len(content.strip()) < 15
        X[i, d + 2] = 1.0 if (is_url or is_short) else 0.0

    return X


# ================================================================== #
# Modèle Transformer                                                  #
# ================================================================== #

class TransformerModel(nn.Module):
    """
    Transformer Encoder sur séquences d'embeddings mE5.

    Input  : (batch, window, 769) — embedding 768d + gap_log 1d
    Output : (batch, window)      — logit de frontière par position

    Design :
    - pre-LN (norm_first=True) : plus stable que post-LN, lr plus élevé OK
    - FFN 4× d_model            : standard Transformer
    - Pas de positional encoding explicite : le gap temporel le remplace
    """

    def __init__(self,
                 d_input:  int   = INPUT_DIM,   # 769
                 d_model:  int   = 128,
                 n_heads:  int   = 4,
                 n_layers: int   = 2,
                 dropout:  float = 0.15):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model={d_model} doit être divisible par n_heads={n_heads}"

        # Projection embeddings + gap → d_model
        self.proj = nn.Linear(d_input, d_model)

        # Transformer Encoder (pre-LN pour stabilité)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,   # pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
        )

        # Tête de prédiction par position
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self,
                x:            torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x            : (B, L, d_input)
        padding_mask : (B, L) bool — True = position paddée, à ignorer
        Returns      : (B, L) logits
        """
        h = self.proj(x)                                           # (B, L, d_model)
        h = self.transformer(h, src_key_padding_mask=padding_mask) # (B, L, d_model)
        return self.head(h).squeeze(-1)                            # (B, L)


# ================================================================== #
# Détecteur haut-niveau                                               #
# ================================================================== #

class TransformerBoundaryDetector:
    """
    Interface haut-niveau — compatible HybridEpisodeSegmenter.

    Usage
    -----
    det = TransformerBoundaryDetector(device='cuda')
    det.fit_sequence(embeddings_tune, artifacts_tune, y_true_tune, n_epochs=60)
    det.optimize_threshold_sequence(embeddings_tune, artifacts_tune, y_true_tune)
    probs = det.predict_proba_sequence(embeddings_test, artifacts_test)
    det.save('boundary_detector_tfm.pt')

    det2 = TransformerBoundaryDetector(device='cuda').load('boundary_detector_tfm.pt')
    """

    def __init__(self,
                 device:    Optional[str] = None,
                 threshold: float = 0.5,
                 d_model:   int   = 128,
                 n_heads:   int   = 4,
                 n_layers:  int   = 2,
                 dropout:   float = 0.15,
                 window_size: int = 100):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device      = device
        self.threshold   = threshold
        self.window_size = window_size
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.dropout     = dropout

        # v1 : projection jointe — INPUT_DIM=769 (emb 768 + gap_log)
        self.model = TransformerModel(
            d_input  = INPUT_DIM,
            d_model  = d_model,
            n_heads  = n_heads,
            n_layers = n_layers,
            dropout  = dropout,
        ).to(device)

    # ------------------------------------------------------------------ #
    # Entraînement                                                         #
    # ------------------------------------------------------------------ #

    def fit_sequence(self,
                     embeddings:   np.ndarray,
                     artifacts:    List[Artifact],
                     y_true:       list,
                     n_epochs:     int   = 60,
                     lr:           float = 5e-4,
                     weight_decay: float = 1e-4,
                     batch_size:   int   = 32,
                     focal_gamma:  float = 2.0,
                     stride_min:   int   = 10,
                     stride_max:   int   = 25,
                     val_split:    float = 0.10,
                     verbose:      bool  = True) -> 'TransformerBoundaryDetector':
        """
        Entraîne le Transformer sur la séquence complète (tune_early).

        LR plus faible que TCN (5e-4 vs 1e-3) : les transformers convergent
        plus lentement mais vers de meilleures solutions.
        Warmup : 5% des steps pour stabiliser l'attention.
        """
        n        = len(embeddings)
        features = _sequence_features(embeddings, artifacts)
        labels   = extract_boundary_labels(y_true, n)

        n_pos       = max(int(labels.sum()), 1)
        n_neg       = n - n_pos
        pos_weight  = n_neg / n_pos
        if verbose:
            print(f'  Positifs    : {n_pos} ({100*n_pos/n:.1f}%)  pos_weight={pos_weight:.1f}')

        # Split validation
        n_val   = max(1, int(n * val_split))
        n_train = n - n_val
        feat_tr, feat_val = features[:n_train], features[n_train:]
        lbl_tr,  lbl_val  = labels[:n_train],   labels[n_train:]

        criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = lr,
            weight_decay = weight_decay,
            betas        = (0.9, 0.98),   # beta2=0.98 standard pour transformers
        )

        # Warmup linéaire sur 5% des epochs, puis cosine
        warmup_steps = max(1, int(n_epochs * 0.05))
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_steps:
                return (epoch + 1) / warmup_steps
            progress = (epoch - warmup_steps) / max(1, n_epochs - warmup_steps)
            return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_f1 = 0.0
        best_state  = None

        self.model.train()
        for epoch in range(n_epochs):
            X_win, y_win = make_windows(
                feat_tr, lbl_tr, self.window_size, stride_min, stride_max)
            idx_shuf = np.random.permutation(len(X_win))

            epoch_loss = []
            for start in range(0, len(X_win), batch_size):
                idx_b = idx_shuf[start:start + batch_size]
                x_t   = torch.from_numpy(X_win[idx_b]).to(self.device)
                y_t   = torch.from_numpy(y_win[idx_b]).to(self.device)

                # Padding mask : positions zeros (padding) → True = ignoré
                pad_mask = (x_t.abs().sum(-1) == 0)   # (B, L)

                logits = self.model(x_t, padding_mask=pad_mask)
                loss   = criterion(logits, y_t)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss.append(loss.item())

            scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                val_f1    = self._eval_f1(feat_val, lbl_val)
                mean_loss = float(np.mean(epoch_loss))
                lr_curr   = scheduler.get_last_lr()[0]
                print(f'  Epoch {epoch+1:3d}/{n_epochs} | '
                      f'loss={mean_loss:.4f} | val_F1={val_f1:.4f} | lr={lr_curr:.2e}')
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state  = {k: v.clone()
                                   for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f'  ✓ Meilleur modèle restauré (val_F1={best_val_f1:.4f})')
        self.model.eval()
        return self

    # ------------------------------------------------------------------ #
    # Inférence                                                            #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_proba_sequence(self,
                                embeddings: np.ndarray,
                                artifacts:  List[Artifact]) -> np.ndarray:
        """
        Retourne P(frontière_t) pour chaque message.
        Fenêtre glissante stride=window//2, moyenne sur chevauchements.

        Returns
        -------
        probs : np.ndarray (n,) float32
        """
        self.model.eval()
        n = len(embeddings)
        if n == 0:
            return np.array([], dtype=np.float32)

        features = _sequence_features(embeddings, artifacts)
        W        = self.window_size
        half     = W // 2

        scores = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)

        for start in range(0, n, half):
            end    = min(start + W, n)
            actual = end - start

            chunk = np.zeros((W, features.shape[1]), dtype=np.float32)
            chunk[:actual] = features[start:end]

            x_t      = torch.from_numpy(chunk[None]).to(self.device)  # (1, W, 769)
            pad_mask = torch.zeros(1, W, dtype=torch.bool, device=self.device)
            if actual < W:
                pad_mask[0, actual:] = True

            logits = self.model(x_t, padding_mask=pad_mask)          # (1, W)
            probs  = torch.sigmoid(logits)[0].cpu().numpy()          # (W,)

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
                                     min_recall: float = 0.85) -> float:
        """
        Seuil optimal sur TUNE_EARLY — jamais sur TEST.
        Maximise F1 sous contrainte recall ≥ min_recall.
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
        preds = (probs >= best_thr).astype(int)
        tp    = int(((preds == 1) & (labels == 1)).sum())
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        rec   = tp / (tp + fn + 1e-8)
        prec  = tp / (tp + fp + 1e-8)
        n_pos = int(labels.sum())
        print(f'  Seuil optimal : {best_thr:.3f} | '
              f'F1={best_f1:.4f} prec={prec:.4f} rec={rec:.4f} | '
              f'pred={tp+fp}  gold={n_pos}')
        return best_thr

    # ------------------------------------------------------------------ #
    # Evaluation interne                                                   #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _eval_f1(self, features: np.ndarray, labels: np.ndarray) -> float:
        """F1 sur le split val (sliding window, seuil=0.5)."""
        n    = len(features)
        W    = self.window_size
        half = W // 2

        scores = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)

        self.model.eval()
        for start in range(0, n, half):
            end    = min(start + W, n)
            actual = end - start

            chunk = np.zeros((W, features.shape[1]), dtype=np.float32)
            chunk[:actual] = features[start:end]

            x_t      = torch.from_numpy(chunk[None]).to(self.device)
            pad_mask = torch.zeros(1, W, dtype=torch.bool, device=self.device)
            if actual < W:
                pad_mask[0, actual:] = True

            logits = self.model(x_t, padding_mask=pad_mask)[0].cpu().numpy()
            probs  = 1 / (1 + np.exp(-logits))
            scores[start:end] += probs[:actual]
            counts[start:end] += 1.0

        self.model.train()

        probs = scores / np.maximum(counts, 1.0)
        preds = (probs >= 0.5).astype(int)
        tp    = int(((preds == 1) & (labels == 1)).sum())
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        prec  = tp / (tp + fp + 1e-8)
        rec   = tp / (tp + fn + 1e-8)
        return 2 * prec * rec / (prec + rec + 1e-8)

    # ------------------------------------------------------------------ #
    # Persistance                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save({
            'model_type': 'transformer',
            'state_dict': self.model.state_dict(),
            'threshold':  self.threshold,
            'model_cfg': {
                'd_input':     INPUT_DIM,
                'd_model':     self.d_model,
                'n_heads':     self.n_heads,
                'n_layers':    self.n_layers,
                'dropout':     self.dropout,
            },
            'window_size': self.window_size,
        }, path)
        print(f'✓ Transformer boundary detector sauvegardé → {path}')

    def load(self, path: str) -> 'TransformerBoundaryDetector':
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg  = ckpt.get('model_cfg', {})
        self.d_model     = cfg.get('d_model',  self.d_model)
        self.n_heads     = cfg.get('n_heads',  self.n_heads)
        self.n_layers    = cfg.get('n_layers', self.n_layers)
        self.dropout     = cfg.get('dropout',  self.dropout)
        self.model = TransformerModel(
            d_input  = cfg.get('d_input', INPUT_DIM),
            d_model  = self.d_model,
            n_heads  = self.n_heads,
            n_layers = self.n_layers,
            dropout  = self.dropout,
        ).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.threshold   = ckpt['threshold']
        self.window_size = ckpt.get('window_size', self.window_size)
        self.model.eval()
        print(f'✓ Transformer boundary detector chargé (seuil={self.threshold:.3f})')
        return self
