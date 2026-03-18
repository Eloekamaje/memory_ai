"""
conversation_preprocessor.py — Prétraitement des conversations réelles.

Problèmes spécifiques aux chats WhatsApp :
  1. Messages bruit (emoji seuls, "ok", "oui") → polluent les centroïdes
  2. Noms des participants omniprésents → entity overlap = 100% sans signal
  3. Liens URL bruts → entités parasites
  4. Messages très courts → embeddings peu informatifs

Stratégie :
  - Marquer les artifacts "noise" (is_noise=True) → poids réduit dans le centroïde
  - Retirer les noms des participants des entités
  - Regrouper les messages adjacents très rapprochés (burst merging)
"""

import re
from typing import List, Set, Tuple
from dataclasses import dataclass
from models import Artifact


# ---------------------------------------------------------------------------
# Détection de bruit
# ---------------------------------------------------------------------------

# Patterns emoji (couvre la majorité des emoji Unicode)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symboles & pictographes
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental
    "\U0001FA00-\U0001FA6F"  # chess
    "\U0001FA70-\U0001FAFF"  # extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # ZWJ
    "\U0000200B-\U0000200F"  # zero-width
    "]+", re.UNICODE
)

# Mots de remplissage (chat noise) — FR/EN
_FILLER_WORDS = {
    "ok", "oui", "non", "okok", "hmm", "hm", "ah", "oh", "ohh", "mdr",
    "lol", "lool", "loool", "ptdr", "haha", "hahaha", "hihi", "yes", "no",
    "yeah", "yep", "nope", "nah", "wow", "aww", "awww", "yo", "hey",
    "sûr", "sure", "cool", "nice", "merci", "thanks", "thx", "svp",
    "pls", "please", "bon", "bah", "bof", "quoi", "hein", "euh",
    "genre", "wsh", "wallah", "jsp", "idk", "btw", "omg", "ikr",
    "stp", "dac", "dacc", "d'accord", "d'ac", "ouais", "mouais",
    "mmh", "mhm", "okk", "okkk", "okkkk",
}

# URL pattern
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def is_noise_message(text: str, min_substantive_chars: int = 4) -> bool:
    """
    Détermine si un message est du bruit conversationnel.

    Bruit = emoji seuls, fillers, très courts, ou ponctuation seule.
    """
    # Retirer emojis et espaces
    stripped = _EMOJI_RE.sub("", text).strip()

    # Retirer URLs
    stripped = _URL_RE.sub("", stripped).strip()

    # Retirer ponctuation et espaces
    alpha_only = re.sub(r"[^a-zA-ZàâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ]", "", stripped)

    # Vide après nettoyage → bruit
    if len(alpha_only) < min_substantive_chars:
        return True

    # Mot filler exact
    normalized = stripped.lower().strip("!?.,… ")
    if normalized in _FILLER_WORDS:
        return True

    # Répétition de la même lettre (ex: "oooooh", "laaaaa")
    if len(set(alpha_only.lower())) <= 2 and len(alpha_only) < 10:
        return True

    return False


# ---------------------------------------------------------------------------
# Nettoyage des entités (retirer participants)
# ---------------------------------------------------------------------------

def remove_participant_entities(artifacts: List[Artifact],
                                participants: Set[str] = None) -> Set[str]:
    """
    Détecte les participants (auteurs) et retire leurs noms des entités
    de chaque artifact.

    Retourne les noms retirés pour référence.
    """
    # Auto-détect participants si non fournis
    if participants is None:
        participants = {a.author.lower().strip() for a in artifacts if a.author}

    removed = set()

    for art in artifacts:
        if not art.entities:
            continue

        cleaned = []
        for ent in art.entities:
            if ent.lower().strip() in participants:
                removed.add(ent.lower().strip())
            else:
                cleaned.append(ent)

        art.entities = cleaned if cleaned else []

    return removed


# ---------------------------------------------------------------------------
# Burst merging : fusionner les messages adjacents très rapprochés
# ---------------------------------------------------------------------------

@dataclass
class BurstConfig:
    """Configuration du burst merging."""
    max_gap_seconds: float = 30.0    # gap max entre messages du même burst
    max_burst_size: int = 5          # max messages dans un burst
    same_author_only: bool = True    # fusionner seulement du même auteur
    separator: str = " | "           # séparateur entre messages fusionnés


def merge_bursts(artifacts: List[Artifact],
                 config: BurstConfig = None) -> List[Artifact]:
    """
    Fusionne les messages très rapprochés (< max_gap_seconds) du même auteur
    en un seul artifact enrichi.

    Cela réduit le nombre d'artifacts et améliore la qualité des embeddings
    (messages plus longs = embeddings plus informatifs).
    """
    if config is None:
        config = BurstConfig()

    if not artifacts:
        return artifacts

    merged = []
    current_burst = [artifacts[0]]

    for i in range(1, len(artifacts)):
        prev = artifacts[i - 1]
        curr = artifacts[i]
        gap = (curr.timestamp - prev.timestamp).total_seconds()

        same_author = (curr.author == prev.author) if config.same_author_only else True

        if (gap <= config.max_gap_seconds
                and same_author
                and len(current_burst) < config.max_burst_size):
            current_burst.append(curr)
        else:
            merged.append(_merge_burst_group(current_burst, config.separator))
            current_burst = [curr]

    # dernier burst
    if current_burst:
        merged.append(_merge_burst_group(current_burst, config.separator))

    # ré-indexer les IDs
    for i, art in enumerate(merged):
        art.id = str(i + 1)

    return merged


def _merge_burst_group(group: List[Artifact], separator: str) -> Artifact:
    """Fusionne un groupe d'artifacts en un seul."""
    if len(group) == 1:
        return group[0]

    # Fusionner les contenus (ignorer bruit pur)
    contents = []
    for art in group:
        if not is_noise_message(art.content):
            contents.append(art.content)
        elif len(contents) == 0:
            # garder au moins le premier même si bruit
            contents.append(art.content)

    merged_content = separator.join(contents) if contents else group[0].content

    # Fusionner les entités
    all_entities = []
    for art in group:
        if art.entities:
            all_entities.extend(art.entities)
    unique_entities = list(dict.fromkeys(all_entities))  # preserve order, deduplicate

    merged = Artifact(
        id=group[0].id,
        timestamp=group[0].timestamp,
        author=group[0].author,
        content=merged_content,
        entities=unique_entities if unique_entities else None,
        importance=max(a.importance for a in group),
    )

    # Conserver true_episode_id si présent (pour évaluation)
    if hasattr(group[0], 'true_episode_id') and group[0].true_episode_id:
        merged.true_episode_id = group[0].true_episode_id

    return merged


# ---------------------------------------------------------------------------
# Pipeline de prétraitement complet
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Configuration du prétraitement conversationnel."""
    remove_noise: bool = True            # marquer les messages bruit
    min_substantive_chars: int = 4       # seuil pour is_noise
    remove_participant_names: bool = True # retirer les noms des auteurs des entités
    merge_bursts_enabled: bool = True    # fusionner les bursts
    burst_config: BurstConfig = None     # config burst (défaut si None)

    def __post_init__(self):
        if self.burst_config is None:
            self.burst_config = BurstConfig()


def preprocess_conversation(artifacts: List[Artifact],
                            config: PreprocessConfig = None) -> Tuple[List[Artifact], dict]:
    """
    Pipeline de prétraitement pour conversations réelles.

    Retourne :
      - artifacts nettoyés
      - stats du prétraitement
    """
    if config is None:
        config = PreprocessConfig()

    stats = {
        "original_count": len(artifacts),
        "noise_count": 0,
        "participants_removed": set(),
        "burst_merges": 0,
    }

    # 1. Détecter les messages bruit
    if config.remove_noise:
        noise_count = 0
        for art in artifacts:
            if is_noise_message(art.content, config.min_substantive_chars):
                noise_count += 1
                # On marque mais on ne supprime PAS : le timing est important
                # pour la segmentation. On ajoute un flag.
                art._is_noise = True
            else:
                art._is_noise = False
        stats["noise_count"] = noise_count

    # 2. Burst merging (avant NER, pour avoir des textes plus riches)
    if config.merge_bursts_enabled:
        before = len(artifacts)
        artifacts = merge_bursts(artifacts, config.burst_config)
        stats["burst_merges"] = before - len(artifacts)

    # 3. Retirer les noms de participants des entités (après NER)
    if config.remove_participant_names:
        removed = remove_participant_entities(artifacts)
        stats["participants_removed"] = removed

    stats["final_count"] = len(artifacts)

    return artifacts, stats
