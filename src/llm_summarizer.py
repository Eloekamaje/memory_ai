"""
llm_summarizer.py — Résumés abstractifs via Qwen2.5-1.5B-Instruct GGUF.

Remplace l'EpisodeSummarizer extractif (TF-IDF + KeyBERT-like)
par un vrai résumé génératif grâce au LLM, tout en gardant une
interface compatible drop-in.

Avantages vs extractif :
    - Résumés naturels et fluides (pas du copier-coller de phrases)
    - Labels thématiques plus pertinents (le LLM comprend le contexte)
    - Capture les intentions et les conclusions implicites
    - Multilingue natif (FR/EN mélangé OK)

Limites :
    - Plus lent (~2-5s par épisode sur CPU)
    - Fenêtre de contexte limitée (2048 tokens → troncature)

Interface compatible :
    - summarize_all(episodes, artifacts, embeddings) → List[EpisodeSummary]
    - summarize_one(episode, artifacts, embeddings) → EpisodeSummary

Usage :
    from llm_summarizer import LLMSummarizer

    summarizer = LLMSummarizer("path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    summaries = summarizer.summarize_all(episodes, artifacts, embeddings)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import Artifact, Episode
from episode_summarizer import EpisodeSummary, attach_summaries


# ================================================================== #
# Prompt engineering — Summarization                                   #
# ================================================================== #

SYSTEM_PROMPT_SUMMARY = """\
Tu es un assistant qui résume des conversations. Pour chaque épisode \
de conversation, tu dois produire :
1. Un LABEL court (5-8 mots max) qui capture le sujet principal
2. Un RÉSUMÉ de 2-3 phrases décrivant ce qui s'est passé
3. Les MOTS-CLÉS principaux (3-5 mots discriminants)
4. La TONALITÉ (positive, negative, neutral, mixed)

Réponds UNIQUEMENT par un JSON valide :
{"label": "...", "summary": "...", "keywords": ["..."], "mood": "..."}

Règles :
- Le label doit être informatif et spécifique (pas "Conversation générale")
- Le résumé doit être naturel et concret
- Les keywords sont des mots ou expressions clés discriminants
- Mood parmi : positive, negative, neutral, mixed"""


def _format_episode_for_prompt(
    artifacts: List[Artifact],
    max_messages: int = 30,
    max_chars_per_msg: int = 150,
) -> str:
    """
    Formate un épisode pour le prompt du LLM.

    Tronque intelligemment les épisodes longs :
    - Garde les 10 premiers messages (contexte d'ouverture)
    - Garde les 10 derniers messages (conclusion)
    - Échantillonne 10 messages du milieu
    """
    n = len(artifacts)

    if n <= max_messages:
        selected = artifacts
    else:
        # Head + middle sample + tail
        head = artifacts[:10]
        tail = artifacts[-10:]
        middle_pool = artifacts[10:-10]
        step = max(1, len(middle_pool) // 10)
        middle = middle_pool[::step][:10]
        selected = head + middle + tail

    lines = []
    for art in selected:
        ts = art.timestamp.strftime("%d/%m %H:%M") if art.timestamp else "??:??"
        author = art.author or "?"
        content = art.content[:max_chars_per_msg].replace("\n", " ").strip()
        lines.append(f"[{ts} {author}] {content}")

    return "\n".join(lines)


# ================================================================== #
# Parsing robuste du JSON                                              #
# ================================================================== #

_JSON_RE = re.compile(r'\{[^{}]*"label"\s*:.*?\}', re.DOTALL)


def _parse_summary_response(text: str) -> Dict:
    """
    Parse la réponse du LLM en dict de résumé.

    Robuste face aux réponses mal formatées.
    """
    text = text.strip()

    # 1. Essai direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "label" in obj:
            return _validate_summary(obj)
    except json.JSONDecodeError:
        pass

    # 2. Regex extraction
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "label" in obj:
                return _validate_summary(obj)
        except json.JSONDecodeError:
            pass

    # 3. Fallback
    return {
        "label": "Conversation",
        "summary": text[:200] if text else "Résumé indisponible.",
        "keywords": [],
        "mood": "neutral",
    }


def _validate_summary(obj: dict) -> Dict:
    """Valide et normalise un résumé parsé."""
    label = str(obj.get("label", "Conversation")).strip()
    summary = str(obj.get("summary", "")).strip()
    keywords = obj.get("keywords", [])
    mood = str(obj.get("mood", "neutral")).lower().strip()

    # Tronquer le label
    words = label.split()
    if len(words) > 10:
        label = " ".join(words[:8]) + "…"

    # Valider keywords
    if isinstance(keywords, list):
        keywords = [str(k).strip() for k in keywords if k][:6]
    else:
        keywords = []

    # Valider mood
    if mood not in {"positive", "negative", "neutral", "mixed"}:
        mood = "neutral"

    return {
        "label": label,
        "summary": summary,
        "keywords": keywords,
        "mood": mood,
    }


# ================================================================== #
# Time label helper (repris de episode_summarizer)                     #
# ================================================================== #

_DAYS_FR = {0: "lun", 1: "mar", 2: "mer", 3: "jeu", 4: "ven", 5: "sam", 6: "dim"}
_MONTHS_FR = {
    1: "jan", 2: "fév", 3: "mar", 4: "avr", 5: "mai", 6: "jun",
    7: "jul", 8: "aoû", 9: "sep", 10: "oct", 11: "nov", 12: "déc",
}


def _time_label(episode: Episode) -> Tuple[str, float]:
    """Génère un label temporel lisible."""
    if episode.time_interval is None:
        return "date inconnue", 0.0

    t_start, t_end = episode.time_interval
    duration = (t_end - t_start).total_seconds() / 3600.0

    day_name = _DAYS_FR.get(t_start.weekday(), "?")
    month_name = _MONTHS_FR.get(t_start.month, "?")

    hour = t_start.hour
    if hour < 6:
        period = "nuit"
    elif hour < 12:
        period = "matin"
    elif hour < 18:
        period = "après-midi"
    elif hour < 22:
        period = "soirée"
    else:
        period = "nuit"

    label = f"{day_name} {t_start.day} {month_name}, {period}"
    return label, duration


# ================================================================== #
# LLM Summarizer                                                      #
# ================================================================== #

# Instance LLM globale (partagée avec llm_ner si même modèle)
_llm_instance: Optional[object] = None


class LLMSummarizer:
    """
    Génère des résumés abstractifs d'épisodes via Qwen2.5 GGUF.

    Drop-in replacement pour EpisodeSummarizer :
        summarize_all(episodes, artifacts, embeddings) → List[EpisodeSummary]

    Parameters
    ----------
    model_path : str | Path
        Chemin vers le GGUF (qwen2.5-1.5b-instruct-q4_k_m.gguf).
    n_ctx : int
        Taille du contexte.
    n_threads : int
        Nombre de threads CPU.
    temperature : float
        Température d'échantillonnage.
    max_tokens : int
        Tokens max pour la réponse.
    verbose : bool
        Afficher les logs de progression.
    """

    def __init__(
        self,
        model_path: str | Path = "",
        n_ctx: int = 2048,
        n_threads: int = 4,
        temperature: float = 0.3,
        max_tokens: int = 300,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Pour compat avec le flag entity_first du pipeline
        self.entity_first = False

        self._llm = None

    # ---------------------------------------------------------- #
    # Chargement lazy                                              #
    # ---------------------------------------------------------- #

    def _ensure_loaded(self):
        """Charge le modèle GGUF (partage l'instance globale si possible)."""
        global _llm_instance
        if self._llm is not None:
            return

        # Essayer de partager avec llm_ner
        try:
            from llm_ner import _llm_instance as ner_instance
            if ner_instance is not None:
                self._llm = ner_instance
                _llm_instance = ner_instance
                if self.verbose:
                    print("  ♻️  Résumé : réutilise l'instance LLM du NER")
                return
        except ImportError:
            pass

        if _llm_instance is not None:
            self._llm = _llm_instance
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle GGUF introuvable : {self.model_path}\n"
                f"Téléchargez qwen2.5-1.5b-instruct-q4_k_m.gguf"
            )

        from llama_cpp import Llama

        t0 = time.time()
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
            n_gpu_layers=0,
        )
        _llm_instance = self._llm

        if self.verbose:
            dt = time.time() - t0
            print(f"  ⚡ LLM Summarizer chargé en {dt:.1f}s ({self.model_path.name})")

    # ---------------------------------------------------------- #
    # Pipeline principal                                           #
    # ---------------------------------------------------------- #

    def summarize_all(
        self,
        episodes: List[Episode],
        artifacts: List[Artifact],
        embeddings: np.ndarray,
    ) -> List[EpisodeSummary]:
        """
        Résume tous les épisodes via le LLM.

        Interface identique à EpisodeSummarizer.summarize_all().
        """
        self._ensure_loaded()

        summaries = []
        t0 = time.time()

        for idx, ep in enumerate(episodes):
            summary = self.summarize_one(ep, artifacts, embeddings)
            summaries.append(summary)

            if self.verbose and (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(episodes) - idx - 1) / rate
                print(
                    f"  📝 Résumés : {idx + 1}/{len(episodes)} "
                    f"({rate:.1f} ep/s, ETA {eta:.0f}s)"
                )

        if self.verbose:
            dt = time.time() - t0
            print(f"  ✅ {len(summaries)} résumés générés en {dt:.1f}s")

        return summaries

    def summarize_one(
        self,
        episode: Episode,
        artifacts: List[Artifact],
        embeddings: np.ndarray,
        tfidf_keywords: List[str] = None,  # ignoré, compat interface
    ) -> EpisodeSummary:
        """Résume un seul épisode via le LLM."""

        ep_artifacts = [artifacts[i] for i in episode.artifact_indices]
        n_msgs = len(ep_artifacts)

        # --- Contexte temporel ---
        time_label, duration_hours = _time_label(episode)

        # --- Entités connues (du pipeline précédent) ---
        top_entities = sorted(
            episode.entity_weights.keys(),
            key=lambda k: episode.entity_weights[k],
            reverse=True,
        )[:5]

        # --- Formater l'épisode pour le prompt ---
        conversation_text = _format_episode_for_prompt(ep_artifacts)

        # --- Construire le prompt ---
        context_info = f"Épisode de {n_msgs} messages, {time_label}"
        if top_entities:
            context_info += f", participants/entités : {', '.join(top_entities[:3])}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
            # Few-shot example
            {"role": "user", "content": (
                "Contexte : Épisode de 8 messages, lun 12 fév, soirée\n\n"
                "[12/02 20:15 Marie] T'as vu le match ce soir ?\n"
                "[12/02 20:16 Thomas] Ouais c'était dingue, le 3ème but !\n"
                "[12/02 20:17 Marie] Mbappé il est trop fort\n"
                "[12/02 20:18 Thomas] Carrément, meilleur joueur du monde\n"
                "[12/02 20:20 Marie] On devrait aller au stade la prochaine fois\n"
                "[12/02 20:21 Thomas] Grave, je check les places"
            )},
            {"role": "assistant", "content": json.dumps({
                "label": "Match de foot — Mbappé et le stade",
                "summary": (
                    "Marie et Thomas discutent du match de football du soir, "
                    "impressionnés par la performance de Mbappé et son troisième but. "
                    "Ils envisagent d'aller au stade pour un prochain match."
                ),
                "keywords": ["match", "Mbappé", "stade", "football", "places"],
                "mood": "positive"
            }, ensure_ascii=False)},
            # Texte cible
            {"role": "user", "content": f"Contexte : {context_info}\n\n{conversation_text}"},
        ]

        # --- Appel LLM ---
        raw = self._call_llm(messages)
        parsed = _parse_summary_response(raw)

        # --- Construire EpisodeSummary ---
        return EpisodeSummary(
            episode_id=episode.id,
            label=parsed["label"],
            summary=parsed["summary"],
            keywords=parsed["keywords"],
            top_entities=top_entities[:4],
            representative_quotes=self._pick_quotes(ep_artifacts, embeddings,
                                                     episode),
            mood=parsed["mood"],
            time_label=time_label,
            artifact_count=n_msgs,
            duration_hours=duration_hours,
        )

    # ---------------------------------------------------------- #
    # Appel LLM                                                    #
    # ---------------------------------------------------------- #

    def _call_llm(self, messages: list) -> str:
        """Appel au modèle via chat completion."""
        try:
            resp = self._llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=["```", "\n\n\n"],
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  LLM Summary error: {e}")
            return '{"label": "Conversation", "summary": "Résumé indisponible.", "keywords": [], "mood": "neutral"}'

    # ---------------------------------------------------------- #
    # Citations (KeyBERT-like, repris de l'extractif)              #
    # ---------------------------------------------------------- #

    def _pick_quotes(
        self,
        artifacts: List[Artifact],
        all_embeddings: np.ndarray,
        episode: Episode,
        max_quotes: int = 3,
        min_length: int = 15,
    ) -> List[str]:
        """
        Sélectionne les messages les plus représentatifs via similarité
        cosinus au centroïde de l'épisode.

        (Ici on garde l'approche embedding-based — plus efficace que
        demander au LLM de citer.)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import math

        centroid = episode.centroid
        if centroid is None or len(artifacts) == 0:
            return []

        ep_embeddings = all_embeddings[episode.artifact_indices]
        sims = cosine_similarity(
            ep_embeddings, centroid.reshape(1, -1)
        ).flatten()

        scores = []
        for i, art in enumerate(artifacts):
            content = art.content.strip()
            if len(content) < min_length:
                continue
            length_bonus = math.log1p(len(content))
            score = sims[i] * 0.7 + (length_bonus / 10.0) * 0.3
            scores.append((score, content))

        scores.sort(key=lambda x: x[0], reverse=True)

        quotes = []
        for _, text in scores:
            preview = text[:120].strip()
            # Éviter les doublons
            if not any(_jaccard_overlap(preview, q) > 0.6 for q in quotes):
                quotes.append(preview)
            if len(quotes) >= max_quotes:
                break

        return quotes

    # ---------------------------------------------------------- #
    # Nettoyage                                                    #
    # ---------------------------------------------------------- #

    def unload(self):
        """Libère le modèle de la mémoire."""
        global _llm_instance
        self._llm = None
        _llm_instance = None


# ================================================================== #
# Utilitaires                                                          #
# ================================================================== #

def _jaccard_overlap(a: str, b: str) -> float:
    """Overlap Jaccard entre les mots de deux textes."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ================================================================== #
# Convenience factory                                                  #
# ================================================================== #

DEFAULT_GGUF_PATH = Path(__file__).parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
_ALT_GGUF_PATH = Path.home() / "ProjetsPerso" / "mivias" / "backend" / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def get_default_gguf_path() -> Path:
    """Retourne le premier chemin GGUF existant."""
    if DEFAULT_GGUF_PATH.exists():
        return DEFAULT_GGUF_PATH
    if _ALT_GGUF_PATH.exists():
        return _ALT_GGUF_PATH
    return DEFAULT_GGUF_PATH


def create_llm_summarizer(
    model_path: Optional[str | Path] = None,
    verbose: bool = False,
    **kwargs,
) -> LLMSummarizer:
    """Factory pour créer un summarizer LLM avec des defaults raisonnables."""
    if model_path is None:
        model_path = get_default_gguf_path()
    return LLMSummarizer(model_path=model_path, verbose=verbose, **kwargs)
