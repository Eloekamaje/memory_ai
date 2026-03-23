"""
boundary_detector_llm.py — LLM-based Boundary Detector (Stage 1 v4).

Utilise un small LLM (Qwen2.5-1.5B fine-tuné LoRA-Seg) pour détecter
les frontières d'épisodes en lisant le TEXTE des conversations.

Avantage clé vs Transformer embeddings :
    Le LLM capte la pragmatique conversationnelle :
    - "Bon allez, à demain" → clôture (frontière après)
    - "D'ailleurs..."       → digression (frontière avant)
    - Changement de registre, code-switching FR/EN, actes de langage

Architecture :
    - Modèle : Qwen2.5-1.5B-Instruct fine-tuné LoRA (boundary classification)
    - Format : GGUF Q4_K_M (~1.1 GB) via llama-cpp-python
    - Inférence : CPU only, ~3-5s par fenêtre de messages

Modes d'utilisation :
    1. Standalone : remplace le Transformer Stage 1
    2. Hybride    : ne traite que les cas ambigus du Transformer (zone grise)

Interface compatible avec TransformerBoundaryDetector :
    - predict_proba_sequence(embeddings, artifacts) → (n,) float
    - predict_proba(X) — non applicable, utilise predict_proba_sequence

Usage :
    from boundary_detector_llm import LLMBoundaryDetector

    # Standalone
    detector = LLMBoundaryDetector("models/qwen2.5-boundary-q4_k_m.gguf")
    probs = detector.predict_proba_sequence(embeddings, artifacts)

    # Hybride avec Transformer
    detector = LLMBoundaryDetector(
        "models/qwen2.5-boundary-q4_k_m.gguf",
        fallback_detector=transformer_detector,
        ambiguity_low=0.20,
        ambiguity_high=0.80,
    )
    probs = detector.predict_proba_sequence(embeddings, artifacts)
"""

from __future__ import annotations

import re
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from models import Artifact


# ================================================================== #
# Prompt engineering                                                   #
# ================================================================== #

SYSTEM_PROMPT = """\
Tu es un expert en analyse conversationnelle. Ta tâche est de déterminer \
si le dernier message d'une conversation commence un NOUVEAU sujet/épisode \
ou s'il CONTINUE le sujet en cours.

Réponds UNIQUEMENT par un JSON : {"boundary": true/false, "confidence": 0.0-1.0}

Critères de frontière (nouveau sujet) :
- Changement de thème explicite (nouveau sujet sans lien)
- Reprise après une longue pause avec un nouveau sujet
- Marqueurs de clôture suivis d'un nouveau sujet ("bon allez", "bref", "sinon")
- Changement de registre significatif (formel↔informel)

Critères de continuation (même sujet) :
- Réponse directe à un message précédent
- Réaction/emoji en rapport avec le contexte
- Approfondissement du même sujet
- Partage de médias liés au sujet en cours"""


def _format_message(artifact: Artifact, idx: int) -> str:
    """Formate un message pour le prompt."""
    ts = artifact.timestamp.strftime("%d/%m %H:%M")
    author = artifact.author or "?"
    content = artifact.content[:200]  # tronquer les messages longs
    return f"[{ts} {author}] {content}"


def _format_gap(art_prev: Artifact, art_next: Artifact) -> str:
    """Ajoute un marqueur de gap temporel si significatif."""
    gap_min = abs(
        (art_next.timestamp - art_prev.timestamp).total_seconds() / 60.0
    )
    if gap_min < 5:
        return ""
    if gap_min < 60:
        return f"  --- {gap_min:.0f} min de pause ---\n"
    if gap_min < 1440:
        return f"  --- {gap_min / 60:.1f}h de pause ---\n"
    return f"  --- {gap_min / 1440:.1f} jours de pause ---\n"


def build_prompt(
    artifacts: List[Artifact],
    target_idx: int,
    context_before: int = 15,
    context_after: int = 0,
) -> str:
    """
    Construit le prompt pour classifier un message comme frontière.

    Parameters
    ----------
    artifacts      : liste complète des artefacts
    target_idx     : index du message à classifier
    context_before : nombre de messages de contexte avant
    context_after  : nombre de messages de contexte après (pour bidirectionnel)
    """
    start = max(0, target_idx - context_before)
    end = min(len(artifacts), target_idx + context_after + 1)

    lines = []
    for i in range(start, end):
        # Ajouter un marqueur de gap
        if i > start:
            gap = _format_gap(artifacts[i - 1], artifacts[i])
            if gap:
                lines.append(gap)

        prefix = ">>>" if i == target_idx else "   "
        lines.append(f"{prefix} {_format_message(artifacts[i], i)}")

    messages_text = "\n".join(lines)

    return f"""Voici un extrait de conversation WhatsApp. Le message marqué >>> est celui à analyser.

{messages_text}

Le message marqué >>> commence-t-il un nouveau sujet/épisode ?
Réponds UNIQUEMENT en JSON : {{"boundary": true/false, "confidence": 0.0-1.0}}"""


# ================================================================== #
# Parsing de la réponse LLM                                           #
# ================================================================== #

_JSON_RE = re.compile(r'\{[^}]+\}')


def parse_llm_response(text: str) -> Tuple[bool, float]:
    """
    Parse la réponse JSON du LLM.

    Returns (is_boundary, confidence)
    Fallback robuste si le JSON est malformé.
    """
    # Chercher un objet JSON dans la réponse
    match = _JSON_RE.search(text)
    if match:
        import json
        try:
            data = json.loads(match.group())
            boundary = bool(data.get("boundary", False))
            confidence = float(data.get("confidence", 0.5))
            return boundary, min(max(confidence, 0.0), 1.0)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback : chercher des mots-clés
    text_lower = text.lower()
    if "true" in text_lower or "oui" in text_lower or "nouveau" in text_lower:
        return True, 0.6
    if "false" in text_lower or "non" in text_lower or "continue" in text_lower:
        return False, 0.6

    # Par défaut : pas de frontière
    return False, 0.3


# ================================================================== #
# LLM Boundary Detector                                               #
# ================================================================== #

class LLMBoundaryDetector:
    """
    Détecteur de frontières basé sur un small LLM (GGUF via llama-cpp-python).

    Parameters
    ----------
    model_path       : chemin vers le fichier GGUF
    n_ctx            : taille du contexte (tokens)
    context_messages : nombre de messages de contexte avant le message cible
    temperature      : température de génération (0.0 = déterministe)
    fallback_detector: détecteur de fallback (Transformer) pour le mode hybride
    ambiguity_low    : seuil bas de la zone d'ambiguïté (mode hybride)
    ambiguity_high   : seuil haut de la zone d'ambiguïté (mode hybride)
    verbose          : afficher la progression
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        context_messages: int = 15,
        temperature: float = 0.0,
        fallback_detector: Optional[object] = None,
        ambiguity_low: float = 0.20,
        ambiguity_high: float = 0.80,
        verbose: bool = True,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.context_messages = context_messages
        self.temperature = temperature
        self.fallback_detector = fallback_detector
        self.ambiguity_low = ambiguity_low
        self.ambiguity_high = ambiguity_high
        self.verbose = verbose
        self.threshold = 0.50  # seuil de décision par défaut

        self._llm = None  # lazy loading

    def _load_model(self):
        """Charge le modèle GGUF (lazy)."""
        if self._llm is not None:
            return

        from llama_cpp import Llama

        if self.verbose:
            print(f"  Chargement LLM : {self.model_path}")

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=4,       # i5-1235U : 4 P-cores
            n_gpu_layers=0,    # CPU only
            verbose=False,
        )

        if self.verbose:
            print(f"  ✓ LLM chargé ({Path(self.model_path).stat().st_size / 1e9:.1f} GB)")

    def _predict_one(self, artifacts: List[Artifact], target_idx: int) -> float:
        """
        Prédit P(frontière) pour un seul message via le LLM.

        Returns
        -------
        float : probabilité de frontière [0, 1]
        """
        self._load_model()

        prompt = build_prompt(
            artifacts, target_idx,
            context_before=self.context_messages,
        )

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=64,
            temperature=self.temperature,
        )

        text = response["choices"][0]["message"]["content"]
        is_boundary, confidence = parse_llm_response(text)

        # Convertir en probabilité
        if is_boundary:
            return confidence
        else:
            return 1.0 - confidence

    def predict_proba_sequence(
        self,
        embeddings: np.ndarray,
        artifacts: List[Artifact],
        candidate_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Prédit P(frontière) pour chaque message de la séquence.

        En mode hybride (fallback_detector fourni) :
            1. Le Transformer prédit d'abord sur toute la séquence
            2. Le LLM ne traite que les messages dans la zone d'ambiguïté
               (ambiguity_low < P_transformer < ambiguity_high)

        En mode standalone :
            Le LLM traite tous les messages (ou seulement candidate_indices).

        Parameters
        ----------
        embeddings        : (n, d) embeddings mE5 — nécessaire pour le mode hybride
        artifacts         : liste des artefacts
        candidate_indices : si fourni, ne traiter que ces indices (optimisation)

        Returns
        -------
        probs : (n,) float32 — probabilités de frontière
        """
        n = len(artifacts)
        probs = np.zeros(n, dtype=np.float32)

        # ── Mode hybride ──────────────────────────────────────────
        if self.fallback_detector is not None:
            if self.verbose:
                print("  LLM Boundary : mode hybride (Transformer + LLM)")

            # Stage 1a : Transformer sur toute la séquence
            tfm_probs = self.fallback_detector.predict_proba_sequence(
                embeddings, artifacts
            )
            probs[:] = tfm_probs

            # Identifier les messages ambigus
            ambiguous = np.where(
                (tfm_probs > self.ambiguity_low) &
                (tfm_probs < self.ambiguity_high)
            )[0]

            if self.verbose:
                print(f"  Transformer : {n} messages, "
                      f"{int((tfm_probs > self.ambiguity_high).sum())} clairs-boundary, "
                      f"{int((tfm_probs < self.ambiguity_low).sum())} clairs-continuation, "
                      f"{len(ambiguous)} ambigus → LLM")

            # Stage 1b : LLM sur les cas ambigus
            for i, idx in enumerate(ambiguous):
                if idx == 0:
                    continue
                probs[idx] = self._predict_one(artifacts, idx)
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"    LLM : {i + 1}/{len(ambiguous)} messages traités")

            return probs

        # ── Mode standalone ───────────────────────────────────────
        if self.verbose:
            print(f"  LLM Boundary : mode standalone ({n} messages)")

        indices = candidate_indices if candidate_indices is not None else range(1, n)
        total = len(list(indices)) if candidate_indices else n - 1
        t0 = time.time()

        for i, idx in enumerate(indices):
            if idx == 0:
                continue
            probs[idx] = self._predict_one(artifacts, idx)
            if self.verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"    {i + 1}/{total} "
                      f"({rate:.1f} msg/s, ETA {eta / 60:.1f} min)")

        if self.verbose:
            elapsed = time.time() - t0
            print(f"  ✓ {total} messages en {elapsed:.1f}s "
                  f"({total / elapsed:.1f} msg/s)")

        return probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compat API — non applicable pour le LLM (besoin du texte)."""
        raise NotImplementedError(
            "LLMBoundaryDetector nécessite le texte des messages. "
            "Utilisez predict_proba_sequence(embeddings, artifacts) à la place."
        )

    def optimize_threshold(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        min_recall: float = 0.85,
    ) -> float:
        """
        Optimise le seuil de décision pour un rappel minimum donné.

        Parameters
        ----------
        probs      : probabilités prédites (output de predict_proba_sequence)
        y_true     : labels binaires gold
        min_recall : rappel minimum requis

        Returns
        -------
        threshold optimal
        """
        best_thr, best_f1 = 0.5, 0.0

        for thr in np.arange(0.10, 0.90, 0.01):
            preds = (probs >= thr).astype(int)
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())

            rec = tp / (tp + fn + 1e-8)
            if rec < min_recall:
                continue

            prec = tp / (tp + fp + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        self.threshold = best_thr
        if self.verbose:
            print(f"  Seuil optimal : {best_thr:.3f} (F1={best_f1:.4f})")
        return best_thr
