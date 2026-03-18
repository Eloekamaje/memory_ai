"""
episode_summarizer.py — Résumé automatique et labelling thématique des épisodes.

Phase G du framework : donner un nom et un résumé lisible à chaque épisode
sans recourir à un LLM externe.

Stratégie multi-signal :
  1. KeyBERT-like : artefacts les plus proches du centroïde → phrases-clés
  2. TF-IDF local/global : mots discriminants de l'épisode vs corpus
  3. Entités saillantes (entity_weights)
  4. Modèle temporel (quand, durée, fréquence)

Sorties :
  - EpisodeSummary : label court (5-8 mots), résumé (2-3 phrases),
    keywords, dominant_topic, mood estimation
"""

from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from models import Artifact, Episode


# ================================================================== #
# Résultat                                                             #
# ================================================================== #

@dataclass
class EpisodeSummary:
    """Résumé structuré d'un épisode."""

    episode_id: str
    label: str                                     # label court (5-8 mots)
    summary: str                                   # résumé 2-3 phrases
    keywords: List[str] = field(default_factory=list)  # top-5 mots-clés discriminants
    top_entities: List[str] = field(default_factory=list)
    representative_quotes: List[str] = field(default_factory=list)  # 1-3 phrases-clés
    mood: str = "neutral"                          # positif / négatif / neutre / mixte
    time_label: str = ""                           # ex: "soirée du 14 fév"
    artifact_count: int = 0
    duration_hours: float = 0.0

    def __repr__(self):
        return f"📝 {self.episode_id} — {self.label}"

    def to_text(self) -> str:
        """Format texte lisible."""
        lines = [
            f"━━━ {self.episode_id} ━━━",
            f"  📌 {self.label}",
            f"  🕐 {self.time_label} ({self.artifact_count} messages, {self.duration_hours:.1f}h)",
        ]
        if self.keywords:
            lines.append(f"  🔑 {', '.join(self.keywords)}")
        if self.top_entities:
            lines.append(f"  👤 {', '.join(self.top_entities)}")
        if self.mood != "neutral":
            mood_icon = {"positive": "😊", "negative": "😟", "mixed": "🔀"}.get(self.mood, "😐")
            lines.append(f"  {mood_icon} Tonalité: {self.mood}")
        if self.summary:
            lines.append(f"  💬 {self.summary}")
        if self.representative_quotes:
            lines.append("  ── Citations ──")
            for q in self.representative_quotes:
                lines.append(f"     « {q} »")
        return "\n".join(lines)


# ================================================================== #
# Stopwords légers FR/EN                                               #
# ================================================================== #

_STOPWORDS = {
    # FR — articles, pronoms, prépositions
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "est", "que", "qui", "dans", "pour", "pas", "sur", "ce", "avec",
    "il", "elle", "je", "tu", "nous", "vous", "ils", "elles", "on",
    "ne", "se", "sa", "son", "ses", "au", "aux", "ma", "mon", "mes",
    "ta", "ton", "tes", "mais", "ou", "où", "donc", "ni", "car",
    "si", "ya", "ça", "là", "ai", "as", "bien", "très", "plus",
    "aussi", "quoi", "comment", "quand", "moi", "toi", "lui", "eux",
    "cette", "ces", "celui", "celle", "ceux", "dont", "par", "tout",
    "même", "alors", "depuis", "avant", "après", "pendant", "entre",
    "sous", "vers", "chez", "sans", "lors", "dès", "lors",
    # FR — verbes courants (infinitifs + formes conjuguées)
    "être", "avoir", "faire", "aller", "venir", "voir", "vouloir",
    "pouvoir", "savoir", "devoir", "falloir", "prendre", "partir",
    "mettre", "dire", "donner", "passer", "rester", "trouver",
    "arriver", "revenir", "rentrer", "sortir", "fuir", "coucher",
    "dormir", "manger", "boire", "aider", "parler", "penser",
    "vas", "vais", "viens", "vient", "peux", "peut", "sais", "sait",
    "veut", "veux", "faut", "fais", "fait", "dit", "mis", "pris",
    "dors", "dort", "mange", "suis", "êtes", "sont",
    # FR — participes passés / adjectifs verbaux courants
    "allé", "parti", "venu", "vu", "fait", "dit", "mis", "pris",
    "fuit", "couché", "dormi", "mangé", "passé", "arrivé", "rentré",
    "sorti", "resté", "trouvé", "donné", "parlé", "pensé", "aidé",
    # FR — adverbes et mots temporels très fréquents
    "fois", "dernière", "dernier", "derniers", "toujours", "encore",
    "déjà", "jamais", "moment", "jours", "jour", "heure", "heures",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
    "dimanche", "matin", "midi", "soir", "nuit", "hier", "demain",
    "maintenant", "bientôt", "tôt", "tard",
    # FR — mots génériques de conversation
    "chose", "truc", "genre", "façon", "côté", "sorte", "type",
    "peu", "beaucoup", "moins", "longtemps", "souvent", "parfois",
    "vraiment", "plutôt", "assez", "tellement", "rien", "quelque",
    # EN
    "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "a", "an", "and", "but", "or", "nor", "not", "no",
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "about", "into", "over", "after",
    "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "so", "if", "then", "than", "too", "very", "just", "don",
    "now", "here", "there", "when", "where", "how", "all",
    "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "am",
    # Chat noise
    "oui", "non", "ok", "lol", "mdr", "haha", "bah", "euh",
    "ah", "oh", "hey", "yo", "bon", "ben", "hein",
    "null", "yes", "no", "yeah", "nah", "wow",
}

# Patterns pour URLs et emojis
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F\U0000200D"
    "]+", re.UNICODE
)


def _clean_text(text: str) -> str:
    """Nettoyage léger pour extraction de mots-clés."""
    text = _URL_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"[^\w\s''-]", " ", text)
    return text.strip().lower()


def _tokenize(text: str) -> List[str]:
    """Tokenize + filtre stopwords et mots trop courts."""
    words = _clean_text(text).split()
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


# ================================================================== #
# Mood Detection (lexique-based, sans ML)                              #
# ================================================================== #

_POSITIVE_WORDS = {
    "love", "happy", "beautiful", "wonderful", "amazing", "great",
    "good", "best", "excellent", "pretty", "sweet", "nice", "enjoy",
    "amour", "aimer", "belle", "beau", "magnifique", "super",
    "génial", "bien", "meilleur", "bonheur", "heureux", "content",
    "merci", "bisous", "câlin", "doux", "tendresse", "joie",
    "😍", "❤", "😘", "🥰", "😊", "💕", "🤗", "💖", "💗",
}

_NEGATIVE_WORDS = {
    "sad", "angry", "bad", "terrible", "horrible", "hate", "upset",
    "worry", "problem", "issue", "wrong", "sorry", "miss",
    "triste", "fâché", "colère", "problème", "mauvais", "peur",
    "inquiet", "désolé", "manque", "pleure", "mal", "douleur",
    "pénible", "difficile", "stress", "anxieux", "ennui", "ennuyeux",
    "😢", "😭", "😡", "😤", "😰", "😥", "💔", "😞",
}


def _detect_mood(texts: List[str]) -> str:
    """Détection basique du mood via comptage lexical."""
    all_text = " ".join(texts).lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in all_text)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in all_text)
    total = pos + neg
    if total == 0:
        return "neutral"
    ratio = pos / total
    if ratio > 0.7:
        return "positive"
    elif ratio < 0.3:
        return "negative"
    else:
        return "mixed"


# ================================================================== #
# Episode Summarizer                                                   #
# ================================================================== #

class EpisodeSummarizer:
    """
    Génère des résumés structurés pour chaque épisode.

    Méthodes :
      - summarize_all(episodes, artifacts, embeddings)
        → List[EpisodeSummary]
      - summarize_one(episode, artifacts, embeddings, tfidf_keywords)
        → EpisodeSummary
      - print_all(summaries) → affichage formaté
    """

    def __init__(
        self,
        max_keywords: int = 6,
        max_quotes: int = 3,
        min_quote_length: int = 15,
        label_max_words: int = 8,
        entity_first: bool = False,
    ):
        self.max_keywords = max_keywords
        self.max_quotes = max_quotes
        self.min_quote_length = min_quote_length
        self.label_max_words = label_max_words
        self.entity_first = entity_first

    # ============================================================== #
    # Pipeline principal                                               #
    # ============================================================== #

    def summarize_all(
        self,
        episodes: List[Episode],
        artifacts: List[Artifact],
        embeddings: np.ndarray,
    ) -> List[EpisodeSummary]:
        """
        Résume tous les épisodes.

        Utilise TF-IDF corpus-wide pour les mots-clés discriminants,
        puis KeyBERT-like per-episode pour les citations représentatives.
        """
        # 1. Construire les « documents » par épisode pour TF-IDF
        ep_docs = []
        for ep in episodes:
            texts = [artifacts[i].content for i in ep.artifact_indices]
            doc = " ".join(_clean_text(t) for t in texts)
            ep_docs.append(doc)

        # 2. TF-IDF global
        tfidf_keywords = self._extract_tfidf_keywords(ep_docs)

        # 3. Résumer chaque épisode
        summaries = []
        for idx, ep in enumerate(episodes):
            kw = tfidf_keywords[idx] if idx < len(tfidf_keywords) else []
            summary = self.summarize_one(ep, artifacts, embeddings, kw)
            summaries.append(summary)

        return summaries

    def summarize_one(
        self,
        episode: Episode,
        artifacts: List[Artifact],
        embeddings: np.ndarray,
        tfidf_keywords: List[str] = None,
    ) -> EpisodeSummary:
        """Résume un seul épisode."""

        ep_artifacts = [artifacts[i] for i in episode.artifact_indices]
        ep_embeddings = embeddings[episode.artifact_indices]
        texts = [a.content for a in ep_artifacts]

        # --- Temps ---
        time_label, duration_hours = self._time_label(episode)

        # --- Entités top ---
        top_entities = sorted(
            episode.entity_weights.keys(),
            key=lambda k: episode.entity_weights[k],
            reverse=True,
        )[:4]

        # --- Mots-clés (TF-IDF + fréquence locale) ---
        keywords = self._merge_keywords(texts, tfidf_keywords)

        # --- Citations représentatives (KeyBERT-like) ---
        quotes = self._representative_quotes(
            ep_artifacts, ep_embeddings, episode.centroid
        )

        # --- Mood ---
        mood = _detect_mood(texts)

        # --- Label court ---
        label = self._generate_label(keywords, top_entities, quotes)

        # --- Résumé narratif ---
        summary = self._generate_summary(
            ep_artifacts, keywords, top_entities, time_label, mood
        )

        return EpisodeSummary(
            episode_id=episode.id,
            label=label,
            summary=summary,
            keywords=keywords[:self.max_keywords],
            top_entities=top_entities,
            representative_quotes=quotes,
            mood=mood,
            time_label=time_label,
            artifact_count=len(ep_artifacts),
            duration_hours=duration_hours,
        )

    # ============================================================== #
    # TF-IDF Keywords (discriminants vs corpus)                        #
    # ============================================================== #

    def _extract_tfidf_keywords(
        self, documents: List[str], top_n: int = 10
    ) -> List[List[str]]:
        """
        Extrait les mots les plus discriminants par document via TF-IDF.

        Retourne une liste de listes de keywords (une par document).
        """
        if not documents or all(len(d.strip()) == 0 for d in documents):
            return [[] for _ in documents]

        try:
            vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words=list(_STOPWORDS),
                min_df=1,
                max_df=0.85,
                token_pattern=r"(?u)\b[a-zA-ZàâäéèêëïîôùûüÿçÀ-ÿ]{3,}\b",
            )
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            return [[] for _ in documents]

        keywords_per_doc = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray().flatten()
            top_indices = row.argsort()[-top_n:][::-1]
            kw = [feature_names[j] for j in top_indices if row[j] > 0]
            keywords_per_doc.append(kw)

        return keywords_per_doc

    # ============================================================== #
    # Merge keywords : TF-IDF + fréquence locale                      #
    # ============================================================== #

    def _merge_keywords(
        self,
        texts: List[str],
        tfidf_kw: List[str] = None,
    ) -> List[str]:
        """Fusionne TF-IDF keywords et mots fréquents locaux."""
        # Fréquence locale
        all_tokens = []
        for t in texts:
            all_tokens.extend(_tokenize(t))

        freq = Counter(all_tokens)
        local_top = [w for w, _ in freq.most_common(10)]

        # Fusion : TF-IDF en priorité, complété par fréquents locaux
        merged = []
        seen = set()

        if tfidf_kw:
            for kw in tfidf_kw:
                if kw not in seen:
                    merged.append(kw)
                    seen.add(kw)

        for kw in local_top:
            if kw not in seen and len(merged) < self.max_keywords:
                merged.append(kw)
                seen.add(kw)

        return merged[:self.max_keywords]

    # ============================================================== #
    # Citations représentatives (KeyBERT-like)                         #
    # ============================================================== #

    def _representative_quotes(
        self,
        artifacts: List[Artifact],
        embeddings: np.ndarray,
        centroid: Optional[np.ndarray],
    ) -> List[str]:
        """
        Sélectionne les artefacts les plus proches du centroïde
        et les plus longs (informatifs) comme citations.
        """
        if centroid is None or len(artifacts) == 0:
            return []

        # Similarité au centroïde
        sims = cosine_similarity(
            embeddings, centroid.reshape(1, -1)
        ).flatten()

        # Score combiné : similarité × log(longueur)
        scores = []
        for i, art in enumerate(artifacts):
            clean = _clean_text(art.content)
            if len(clean) < self.min_quote_length:
                continue
            length_bonus = math.log1p(len(clean))
            score = sims[i] * 0.7 + (length_bonus / 10.0) * 0.3
            scores.append((score, i, art.content))

        scores.sort(key=lambda x: x[0], reverse=True)

        # Sélectionner les top quotes, en évitant la redondance
        quotes = []
        for _, _, text in scores:
            # Tronquer à 120 chars
            preview = text[:120].strip()
            if preview.endswith("|"):
                preview = preview.rsplit("|", 1)[0].strip()
            # Vérifier que ce n'est pas trop similaire à une quote déjà choisie
            if not any(self._overlap(preview, q) > 0.6 for q in quotes):
                quotes.append(preview)
            if len(quotes) >= self.max_quotes:
                break

        return quotes

    @staticmethod
    def _overlap(a: str, b: str) -> float:
        """Overlap Jaccard entre les mots de deux textes."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    # ============================================================== #
    # Label court                                                      #
    # ============================================================== #

    @staticmethod
    def _keywords_informative(keywords: List[str]) -> bool:
        """True si la liste contient au moins un mot de longueur > 4 hors stopwords."""
        return any(len(k) > 4 for k in keywords)

    def _label_from_quote(self, quotes: List[str]) -> str:
        """Extrait les premiers mots utiles de la meilleure citation."""
        for quote in quotes:
            words = [w for w in quote.split() if len(w) > 2 and w.lower() not in _STOPWORDS]
            if len(words) >= 3:
                return " ".join(w.capitalize() for w in words[:5]) + "…"
        return ""

    def _build_label_parts(self, keywords: List[str], entities: List[str]) -> str:
        """Assemble entités + keywords en chaîne brute (sans troncature)."""
        if self.entity_first and entities:
            ent_part = entities[:3]
            ent_set = {e.lower() for e in ent_part}
            kw_part = [k for k in keywords if k.lower() not in ent_set][:2]
            base = ", ".join(e.capitalize() for e in ent_part)
            if kw_part and self._keywords_informative(kw_part):
                base += " — " + " ".join(w.capitalize() for w in kw_part)
            return base

        kw_part = keywords[:3] if keywords else []
        kw_set = {k.lower() for k in kw_part}
        ent_part = [e for e in entities[:2] if e.lower() not in kw_set]
        parts = []
        if kw_part and self._keywords_informative(kw_part):
            parts.append(" ".join(w.capitalize() for w in kw_part))
        if ent_part:
            parts.append("(" + ", ".join(ent_part) + ")")
        return " ".join(parts)

    def _generate_label(
        self,
        keywords: List[str],
        entities: List[str],
        quotes: Optional[List[str]] = None,
    ) -> str:
        """
        Génère un label court et informatif.

        Priorité : entités/keywords → citation → "Conversation générale".
        """
        label = self._build_label_parts(keywords, entities)

        if not label or not self._keywords_informative(label.split()):
            label = self._label_from_quote(quotes or []) or "Conversation générale"

        words = label.split()
        if len(words) > self.label_max_words:
            label = " ".join(words[: self.label_max_words]) + "…"

        return label

    # ============================================================== #
    # Résumé narratif                                                  #
    # ============================================================== #

    def _generate_summary(
        self,
        artifacts: List[Artifact],
        keywords: List[str],
        entities: List[str],
        time_label: str,
        mood: str,
    ) -> str:
        """
        Génère un résumé de 2-3 phrases décrivant l'épisode.
        """
        n = len(artifacts)

        # Auteurs
        authors = list({a.author for a in artifacts})
        if len(authors) == 1:
            who = authors[0]
        elif len(authors) == 2:
            who = f"{authors[0]} et {authors[1]}"
        else:
            who = f"{authors[0]} et {len(authors) - 1} autres"

        # Phrase 1 : qui, quand, combien
        phrases = [
            f"Conversation entre {who} ({time_label}, {n} messages)."
        ]

        # Phrase 2 : sujets
        subjects = keywords[:3] if keywords else []
        if entities:
            # Ajouter les entités non redondantes
            for e in entities[:2]:
                if e.lower() not in {s.lower() for s in subjects}:
                    subjects.append(e)

        if subjects:
            phrases.append(f"Sujets abordés : {', '.join(subjects)}.")

        # Phrase 3 : tonalité
        mood_phrases = {
            "positive": "Tonalité globalement positive et chaleureuse.",
            "negative": "Tonalité plutôt tendue ou préoccupée.",
            "mixed": "Tonalité variée, mêlant moments légers et sérieux.",
        }
        if mood in mood_phrases:
            phrases.append(mood_phrases[mood])

        return " ".join(phrases)

    # ============================================================== #
    # Time label                                                       #
    # ============================================================== #

    @staticmethod
    def _time_label(episode: Episode) -> Tuple[str, float]:
        """
        Génère un label temporel lisible.

        Retourne (label, duration_hours).
        """
        if episode.time_interval is None:
            return "date inconnue", 0.0

        t_start, t_end = episode.time_interval
        duration = (t_end - t_start).total_seconds() / 3600.0

        # Jour de la semaine FR
        days_fr = {
            0: "lun", 1: "mar", 2: "mer", 3: "jeu",
            4: "ven", 5: "sam", 6: "dim",
        }
        months_fr = {
            1: "jan", 2: "fév", 3: "mar", 4: "avr", 5: "mai", 6: "jun",
            7: "jul", 8: "aoû", 9: "sep", 10: "oct", 11: "nov", 12: "déc",
        }

        day_name = days_fr.get(t_start.weekday(), "?")
        month_name = months_fr.get(t_start.month, "?")

        # Format selon la durée
        if duration < 0.1:
            # Instant
            label = f"{day_name} {t_start.day} {month_name} à {t_start.strftime('%Hh%M')}"
        elif duration < 24:
            # Même jour
            period = _time_period(t_start.hour)
            label = f"{period} du {day_name} {t_start.day} {month_name}"
        elif t_start.date() == t_end.date() or duration < 48:
            # 1-2 jours
            label = f"{day_name} {t_start.day}–{t_end.day} {month_name}"
        else:
            # Plusieurs jours
            if t_start.month == t_end.month:
                label = f"{t_start.day}–{t_end.day} {month_name}"
            else:
                m2 = months_fr.get(t_end.month, "?")
                label = f"{t_start.day} {month_name} – {t_end.day} {m2}"

        return label, duration


def _time_period(hour: int) -> str:
    """Retourne la période de la journée."""
    if hour < 6:
        return "Nuit"
    elif hour < 12:
        return "Matin"
    elif hour < 14:
        return "Midi"
    elif hour < 18:
        return "Après-midi"
    elif hour < 22:
        return "Soirée"
    else:
        return "Nuit"


# ================================================================== #
# Utilitaire : enrichir les épisodes avec les summaries                #
# ================================================================== #

def attach_summaries(
    episodes: List[Episode],
    summaries: List[EpisodeSummary],
) -> Dict[str, EpisodeSummary]:
    """
    Crée un mapping episode_id → EpisodeSummary.

    Stocke aussi le label directement sur l'épisode (via attribut dynamique).
    """
    mapping = {}
    for s in summaries:
        mapping[s.episode_id] = s

    for ep in episodes:
        if ep.id in mapping:
            ep._summary = mapping[ep.id]  # type: ignore

    return mapping
