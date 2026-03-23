"""
entity_extractor.py — Extraction d'entités multilingue et pluggable.

Architecture :
  1. Backend NER pluggable (spaCy OU GLiNER)
  2. Regex pour les patterns structurés (emails, @handles, URLs, téléphones)
  3. Métadonnées (author) comme signal complémentaire
  4. Filtrage structurel domain-agnostic (pas de stoplist hardcodée)

Backends :
  - "spacy"  : rapide (~5s/2500 msgs), français uniquement, bruyant sur l'informel
  - "gliner" : plus lent (~150s/2500 msgs CPU), multilingue, très précis,
               types d'entités personnalisables (person, org, location, project, event)

Usage :
  extractor = EntityExtractor(backend="gliner")      # ou "spacy"
  mentions  = extractor.extract_batch(artifacts)
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from models import Artifact


# ------------------------------------------------------------------ #
# Modèle de mention                                                   #
# ------------------------------------------------------------------ #

class EntityType(Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    PLACE = "PLACE"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    HANDLE = "HANDLE"
    URL = "URL"
    PROJECT = "PROJECT"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    DATE_EVENT = "DATE_EVENT"
    DOCUMENT = "DOCUMENT"
    CONCEPT = "CONCEPT"
    UNKNOWN = "UNKNOWN"


@dataclass
class Mention:
    """Une occurrence brute d'entité dans un artefact."""
    surface_form: str
    entity_type: EntityType
    source_artifact_id: str
    confidence: float = 1.0
    extraction_method: str = "regex"   # spacy | gliner | regex | metadata


# ------------------------------------------------------------------ #
# Patterns regex (language-agnostic)                                  #
# ------------------------------------------------------------------ #

_EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)
_HANDLE_RE = re.compile(r'@([A-Za-z0-9_]{2,30})\b')
_URL_RE = re.compile(
    r'https?://[^\s<>\"\']+|www\.[^\s<>\"\']+',
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r'(?:\+\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{2,4}'
    r'[\s.-]?\d{2,4}(?:[\s.-]?\d{2,4})?'
)


# ------------------------------------------------------------------ #
# Filtrage structurel domain-agnostic                                 #
# ------------------------------------------------------------------ #

_EMOJI_RE = re.compile(
    r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
    r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
    r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F'
    r'\U00002702-\U000027B0\U0000FE0F]'
)
_REPEAT_RE = re.compile(r'(.)\1{2,}')

# Pronoms/déterminants FR+EN — pas des entités nommées
_PRONOUNS = frozenset({
    # FR
    'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
    'me', 'te', 'se', 'le', 'la', 'les', 'lui', 'leur', 'moi', 'toi',
    'ce', 'ça', 'ceci', 'cela', 'qui', 'que', 'quoi', 'dont', 'où',
    'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
    'notre', 'votre', 'nos', 'vos', 'leurs',
    # EN
    'i', 'me', 'my', 'mine', 'you', 'your', 'yours',
    'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'we', 'us', 'our', 'ours',
    'they', 'them', 'their', 'theirs',
    'this', 'that', 'these', 'those',
    'who', 'whom', 'whose', 'which', 'what',
})

# Déterminants FR+EN pour le filtre DET+NomCommun
_DETERMINERS = frozenset({
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de',
    'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
    'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
    'ce', 'cet', 'cette', 'ces',
    'the', 'a', 'an', 'my', 'your', 'his', 'her', 'our', 'their',
})

# Prépositions + interjections + pronoms indéfinis (filtre premier mot)
_FILTER_FIRST_WORDS = _PRONOUNS | _DETERMINERS | frozenset({
    # Prépositions FR
    'pour', 'dans', 'sur', 'sous', 'avec', 'sans', 'en', 'entre',
    'vers', 'par', 'avant', 'après', 'depuis', 'pendant',
    'comme', 'selon', 'contre',
    # Prépositions EN
    'for', 'in', 'on', 'at', 'with', 'without', 'from', 'to',
    'by', 'about', 'between', 'after', 'before', 'during',
    # Pronoms indéfinis
    'chacun', 'chacune', 'tout', 'tous', 'toute', 'toutes',
    'rien', 'personne', 'quelqu\'un', 'certains', 'certaines',
    'deux', 'trois', 'quatre', 'cinq', 'quelques', 'plusieurs',
    # Adj qualificatifs fréquents en premier mot
    'petit', 'petite', 'grand', 'grande', 'bon', 'bonne',
    'vrai', 'vraie', 'beau', 'belle', 'joli', 'jolie',
})

# Interjections FR+EN (pas des entités)
_INTERJECTIONS = frozenset({
    'oh', 'ah', 'oula', 'oulala', 'ohla', 'nop', 'nope',
    'hein', 'bah', 'bof', 'pfff', 'hmm', 'hum', 'mdr',
    'lol', 'wow', 'ouf', 'oups', 'ok', 'okay',
    'hello', 'bonjour', 'salut', 'coucou', 'hey', 'hi',
    'bravo', 'courage',
})

# Noms communs de personne (pas des entités nommées)
_COMMON_PERSON_NOUNS = frozenset({
    # FR — rôles familiaux
    'femme', 'homme', 'monsieur', 'madame', 'mademoiselle',
    'fille', 'garçon', 'enfant', 'bébé', 'maman', 'papa',
    'frère', 'soeur', 'sœur', 'ami', 'amie', 'copain', 'copine',
    'père', 'mère', 'fils', 'tante', 'oncle', 'cousin', 'cousine',
    'voisin', 'voisine', 'collègue', 'tonton',
    'gars', 'gar', 'mec', 'nana', 'type', 'personne', 'gens',
    # FR — métiers/rôles
    'docteur', 'professeur', 'patron', 'chef', 'nurse',
    'infirmière', 'infirmier', 'médecin', 'commandant',
    'propriétaire', 'danseur', 'danseuse', 'programmeur',
    'chauffeur', 'élève', 'empereur', 'choriste', 'pasteur',
    'prêtre', 'missionnaire', 'maîtresse', 'maître',
    # FR — noms communs mal détectés
    'hommes', 'femmes', 'personnes', 'parent', 'parents',
    # EN — roles
    'man', 'woman', 'boy', 'girl', 'child', 'baby',
    'friend', 'brother', 'sister', 'mom', 'dad', 'mother', 'father',
    'doctor', 'teacher', 'boss', 'guy', 'dude', 'people', 'person',
    'son', 'daughter', 'aunt', 'uncle', 'neighbor',
})

# Termes affectueux / surnoms informels (pas des entités nommées)
_AFFECTIONATE_TERMS = frozenset({
    # FR
    'bb', 'bébé', 'chéri', 'chérie', 'mon cœur', 'ma puce',
    'doudou', 'bébou', 'chou', 'mon ange',
    # EN
    'baby', 'babe', 'sweety', 'sweetie', 'honey', 'darling',
    'love', 'dear', 'boo', 'cutie', 'hun',
})

# Platform media markers
_MEDIA_MARKERS = frozenset({
    'médias', '<médias omis>', 'média omis',
    'image omise', 'vidéo omise', 'audio omis',
    'sticker omis', 'gif omis', 'document omis',
    'media omitted', 'image omitted', 'video omitted',
})


def _structural_filter(text: str, min_len: int = 2, max_len: int = 50) -> bool:
    """Filtrage structurel domain-agnostic. True = garder."""
    clean = text.strip()
    if len(clean) < min_len or len(clean) > max_len:
        return False
    if not any(c.isalpha() for c in clean):
        return False
    if clean.replace(' ', '').isdigit():
        return False
    if _EMOJI_RE.search(clean):
        return False
    if _REPEAT_RE.search(clean):
        return False
    if clean.lower().strip() in _MEDIA_MARKERS:
        return False
    if clean.lower().strip() in _PRONOUNS:
        return False
    # Interjection seule
    if clean.lower().strip() in _INTERJECTIONS:
        return False
    # Commence par pronom/dét/préposition/adj → "I love you", "Pour moi", "A pretty girl"
    first_word = clean.split()[0].lower() if clean.split() else ''
    if len(clean.split()) >= 2 and first_word in _FILTER_FIRST_WORDS:
        return False
    # Chaîne aléatoire (IDs YouTube, tokens) : faible ratio voyelles OU mélange lettres+chiffres
    alpha_chars = [c.lower() for c in clean if c.isalpha()]
    digit_chars = [c for c in clean if c.isdigit()]
    if alpha_chars:
        vowels = sum(1 for c in alpha_chars if c in 'aeiouyàâéèêëïîôùûü')
        if vowels / len(alpha_chars) < 0.15:
            return False
    # Mélange lettres + chiffres dans un seul mot → token aléatoire
    if len(clean.split()) == 1 and alpha_chars and digit_chars:
        return False
    # Trop de mots → fragment conversationnel
    if len(clean.split()) > 4:
        return False
    return True


def _filter_common_person(text: str, label: str) -> bool:
    """Filtrer les noms communs de personne et termes affectueux.
    True = à rejeter."""
    lower = text.lower().strip()
    if label != 'person':
        return False
    # Terme affectueux seul
    if lower in _AFFECTIONATE_TERMS:
        return True
    # Nom commun seul
    if lower in _COMMON_PERSON_NOUNS:
        return True
    # DET + quelque chose : "la femme", "mon père", "Le bb", "le commandant"
    words = lower.split()
    if len(words) >= 2 and words[0] in _DETERMINERS:
        rest = ' '.join(words[1:])
        if rest in _COMMON_PERSON_NOUNS or rest in _AFFECTIONATE_TERMS:
            return True
        # DET + n'importe quel nom commun de rôle (même dans multi-mot)
        if words[1] in _COMMON_PERSON_NOUNS:
            return True
    # Premier mot = nom commun de personne : "monsieur charismatique", "médecin traitant"
    if len(words) >= 2 and words[0] in _COMMON_PERSON_NOUNS:
        return True
    # Mot seul qui est un nom commun au pluriel ou avec suffixe
    if lower.rstrip('s') in _COMMON_PERSON_NOUNS:
        return True
    return False


# Noms communs de lieu (pas des entités nommées)
_COMMON_PLACE_NOUNS = frozenset({
    # FR
    'maison', 'village', 'terrain', 'métro', 'rue', 'avenue',
    'banquise', 'monde', 'terre', 'pays', 'ville', 'quartier',
    'foyer', 'bureau', 'école', 'hôpital', 'église', 'marché',
    # EN
    'house', 'home', 'city', 'town', 'street', 'world',
    'land', 'country', 'school', 'office', 'church',
})

# Noms communs d'organisation
_COMMON_ORG_NOUNS = frozenset({
    # FR
    'gouvernement', 'police', 'pompiers', 'comptabilité',
    'compagnie', 'entreprise', 'équipe', 'team',
    'télévision', 'radio', 'journal',
    # EN
    'government', 'police', 'company', 'television',
})


def _filter_common_noun(text: str, label: str) -> bool:
    """Filtrer les noms communs (lieu, org, produit) détectés par NER. True = rejeter."""
    lower = text.lower().strip()
    words = lower.split()

    if label == 'location':
        if lower in _COMMON_PLACE_NOUNS:
            return True
        if len(words) >= 2 and words[0] in _DETERMINERS:
            if words[1] in _COMMON_PLACE_NOUNS:
                return True

    if label == 'organization':
        if lower in _COMMON_ORG_NOUNS:
            return True
        if len(words) >= 2 and words[0] in _DETERMINERS:
            if words[1] in _COMMON_ORG_NOUNS:
                return True

    # PRODUCT : filtrer les noms communs, adjectifs, interjections
    if label == 'product':
        if lower in _COMMON_PRODUCT_NOISE:
            return True
        if lower in _INTERJECTIONS:
            return True
        if lower in _AFFECTIONATE_TERMS:
            return True
        # Mot courant d'1 seul mot tout en minuscules → probablement pas un produit
        if len(words) == 1 and lower == text.strip() and lower.isalpha():
            return True

    # DATE_EVENT : jours de la semaine seuls ne sont pas des événements
    if label == 'holiday or celebration':
        if lower in _COMMON_DAY_NAMES:
            return True

    return False


# Noms communs bruyants pour PRODUCT
_COMMON_PRODUCT_NOISE = frozenset({
    # FR — mots courants détectés comme "product"
    'vêtements', 'vêtement', 'mince', 'cool', 'boulot', 'truc',
    'chose', 'machin', 'bidule', 'pouf', 'perfecto', 'exquis',
    'mielleux', 'bonjour', 'bonheur', 'beauté',
    # EN
    'stuff', 'thing', 'cool', 'nice', 'sweet', 'great',
})

# Jours de la semaine (pas des événements calendaires)
_COMMON_DAY_NAMES = frozenset({
    'lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
})


# ------------------------------------------------------------------ #
# Backend abstrait                                                    #
# ------------------------------------------------------------------ #

class NERBackend(ABC):
    """Interface pour un backend NER."""

    @abstractmethod
    def extract_batch(self, texts: List[str]) -> List[List[dict]]:
        """
        Extraire les entités de chaque texte.

        Returns: List[List[{text, label, score}]] — une liste par texte.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du backend pour les logs."""
        ...


# ------------------------------------------------------------------ #
# Backend spaCy                                                       #
# ------------------------------------------------------------------ #

# POS tags incompatibles avec les entités nommées
_REJECT_ROOT_POS = frozenset({
    'VERB', 'ADJ', 'ADV', 'AUX', 'INTJ',
    'PUNCT', 'SYM', 'DET', 'PRON',
    'CCONJ', 'SCONJ', 'ADP', 'PART',
})

_SPACY_LABEL_MAP = {
    'PER': 'person', 'PERS': 'person',
    'LOC': 'location', 'GPE': 'location', 'FAC': 'location',
    'ORG': 'organization',
    'MISC': 'concept',
    'EVENT': 'event',
    'PRODUCT': 'concept',
    'WORK_OF_ART': 'concept',
}

# Singleton spaCy
_nlp_instance = None

def _get_nlp(model_name: str = "fr_core_news_lg"):
    """Charger le modèle spaCy en singleton."""
    global _nlp_instance
    if _nlp_instance is None:
        import spacy
        for name in [model_name, "fr_core_news_md", "fr_core_news_sm"]:
            try:
                _nlp_instance = spacy.load(name, disable=["parser", "lemmatizer"])
                _nlp_instance.max_length = 2_000_000
                break
            except OSError:
                continue
        else:
            raise RuntimeError("Aucun modèle spaCy français trouvé")
    return _nlp_instance


class SpacyBackend(NERBackend):
    """Backend spaCy avec filtrage POS + lowercase cross-check."""

    def __init__(self, model_name: str = "fr_core_news_lg",
                 batch_size: int = 256):
        self._nlp = _get_nlp(model_name)
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "spacy"

    def extract_batch(self, texts: List[str]) -> List[List[dict]]:
        results: List[List[dict]] = []
        docs = self._nlp.pipe(texts, batch_size=self._batch_size)
        for doc in docs:
            entities = []
            for ent in doc.ents:
                # Filtre POS spaCy-specific
                if not self._filter_spacy_entity(ent):
                    continue
                label = _SPACY_LABEL_MAP.get(ent.label_)
                if label is None:
                    continue
                score = self._compute_confidence(ent)
                entities.append({
                    'text': ent.text.strip(),
                    'label': label,
                    'score': score,
                })
            results.append(entities)
        return results

    def _filter_spacy_entity(self, ent) -> bool:
        """
        Filtrer une entité spaCy via ses features linguistiques (POS).
        Domain-agnostic — pas de blacklist ad hoc.
        """
        text = ent.text.strip()
        label = ent.label_

        # Checks structurels rapides
        if len(text) < 2 or len(text) > 50:
            return False
        if not any(c.isalpha() for c in text):
            return False
        if text.replace(' ', '').isdigit():
            return False
        if _EMOJI_RE.search(text):
            return False
        if _REPEAT_RE.search(text):
            return False
        if text.lower().strip() in _MEDIA_MARKERS:
            return False

        n_words = len(text.split())
        if n_words > 4:
            return False

        # POS-based
        tokens = [t for t in ent if not t.is_punct and not t.is_space]
        if not tokens:
            return False

        root = ent.root
        if root.pos_ in _REJECT_ROOT_POS:
            return False
        if all(t.is_stop or len(t.text) <= 1 for t in tokens):
            return False

        has_propn = any(t.pos_ == 'PROPN' for t in tokens)
        has_noun = any(t.pos_ == 'NOUN' for t in tokens)

        if n_words == 1 and not has_propn:
            return False
        if n_words >= 2 and not has_propn and not has_noun:
            return False

        # MISC stricter
        if label == 'MISC':
            if not has_propn:
                return False
            if n_words == 1 and len(text) < 4:
                return False
        if label in ('PER', 'PERS') and len(text) < 3:
            return False
        if label == 'ORG' and len(text) < 3:
            return False

        return True

    def _compute_confidence(self, ent) -> float:
        """Confidence POS-based + lowercase cross-check."""
        root = ent.root
        label = ent.label_
        has_propn = any(t.pos_ == 'PROPN' for t in ent if not t.is_punct)
        n_tokens = len([t for t in ent if not t.is_punct and not t.is_space])

        # Lowercase cross-check pour single-word PROPN
        if n_tokens == 1 and root.pos_ == 'PROPN':
            doc_lower = self._nlp(root.text.lower())
            if doc_lower[0].pos_ == 'PROPN':
                return 0.90
            if label in ('PER', 'PERS'):
                return 0.60
            if label in ('LOC', 'GPE', 'FAC'):
                return 0.35
            if label == 'ORG':
                return 0.45
            return 0.30

        if root.pos_ == 'PROPN':
            return 0.90
        if has_propn:
            return 0.80
        if root.pos_ == 'NOUN':
            return 0.65
        if root.pos_ == 'X':
            return 0.55
        return 0.50


# ------------------------------------------------------------------ #
# Backend GLiNER                                                      #
# ------------------------------------------------------------------ #

class GLiNERBackend(NERBackend):
    """Backend GLiNER : NER zero-shot multilingue basé sur un transformer."""

    DEFAULT_LABELS = [
        'person', 'organization', 'location', 'project', 'event',
        'product', 'holiday or celebration', 'official document',
    ]

    def __init__(self,
                 model_name: str = "urchade/gliner_multi-v2.1",
                 labels: Optional[List[str]] = None,
                 threshold: float = 0.6,
                 batch_size: int = 32):
        from gliner import GLiNER
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = GLiNER.from_pretrained(model_name)
        self._labels = labels or self.DEFAULT_LABELS
        self._threshold = threshold
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "gliner"

    def extract_batch(self, texts: List[str]) -> List[List[dict]]:
        import warnings
        # GLiNER retourne [{text, label, score}, ...] par document
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._model.batch_predict_entities(
                texts,
                self._labels,
                threshold=self._threshold,
                batch_size=self._batch_size,
            )
        return raw


# ------------------------------------------------------------------ #
# Mapping label → EntityType                                          #
# ------------------------------------------------------------------ #

_LABEL_TO_ENTITY_TYPE = {
    'person': EntityType.PERSON,
    'organization': EntityType.ORG,
    'location': EntityType.PLACE,
    'project': EntityType.PROJECT,
    'event': EntityType.EVENT,
    'product': EntityType.PRODUCT,
    'holiday or celebration': EntityType.DATE_EVENT,
    'official document': EntityType.DOCUMENT,
    'concept': EntityType.CONCEPT,
}


# ------------------------------------------------------------------ #
# Extracteur principal                                                #
# ------------------------------------------------------------------ #

class EntityExtractor:
    """
    Extraction d'entités multilingue et pluggable.

    Parameters
    ----------
    backend : "gliner" | "spacy"
        Backend NER à utiliser. Default="spacy" pour backward compat.
    min_confidence : float
        Confiance minimum pour garder une entité NER.
    extract_from_content : bool
        Extraire des entités du contenu textuel.
    extract_from_metadata : bool
        Extraire l'auteur comme entité PERSON.

    Backward-compatible kwargs (ignorés si non-spacy):
        use_spacy, batch_size
    """

    def __init__(self,
                 backend: str = "spacy",
                 min_confidence: float = 0.50,
                 extract_from_content: bool = True,
                 extract_from_metadata: bool = True,
                 # Backward compat params
                 use_spacy: bool = True,
                 batch_size: int = 256,
                 **backend_kwargs):
        self.min_confidence = min_confidence
        self.extract_from_content = extract_from_content
        self.extract_from_metadata = extract_from_metadata

        if backend == "gliner":
            self._backend = GLiNERBackend(**backend_kwargs)
        elif backend == "spacy":
            self._backend = SpacyBackend(batch_size=batch_size)
        elif backend == "llm":
            from llm_ner import create_llm_ner
            self._backend = create_llm_ner(**backend_kwargs)
        else:
            raise ValueError(f"Backend inconnu : {backend!r}. Choix : 'gliner', 'spacy', 'llm'")

        self._backend_name = self._backend.name

    def extract(self, artifact: Artifact) -> List[Mention]:
        """Extraire les mentions d'un seul artefact."""
        return self.extract_batch([artifact])

    def extract_batch(self, artifacts: List[Artifact]) -> List[Mention]:
        """
        Extraire les mentions de tous les artefacts.

        Pipeline :
          1. Métadonnées (author) + regex (email, url, handle, phone) + legacy
          2. Backend NER (spaCy ou GLiNER) avec filtrage structurel partagé
        """
        all_mentions: List[Mention] = []

        # Phase 1 : métadonnées + regex + legacy
        for artifact in artifacts:
            self._extract_author(artifact, all_mentions)
            if self.extract_from_content and artifact.content:
                all_mentions.extend(
                    self._extract_structured(artifact.content, artifact.id)
                )
            all_mentions.extend(self._from_existing_entities(artifact))

        # Phase 2 : NER via le backend choisi
        if self.extract_from_content:
            texts, aids = [], []
            for artifact in artifacts:
                if artifact.content and len(artifact.content.strip()) > 2:
                    texts.append(artifact.content)
                    aids.append(artifact.id)

            if texts:
                batch_results = self._backend.extract_batch(texts)
                for doc_ents, aid in zip(batch_results, aids):
                    for ent in doc_ents:
                        text = ent['text'].strip()
                        label = ent['label']
                        score = ent.get('score', 0.85)

                        # Filtre structurel partagé
                        if not _structural_filter(text):
                            continue
                        # Filtre noms communs de personne
                        if _filter_common_person(text, label):
                            continue
                        # Filtre noms communs (lieu, org)
                        if _filter_common_noun(text, label):
                            continue
                        # Filtre confiance
                        if score < self.min_confidence:
                            continue

                        etype = _LABEL_TO_ENTITY_TYPE.get(
                            label, EntityType.UNKNOWN
                        )
                        all_mentions.append(Mention(
                            surface_form=text,
                            entity_type=etype,
                            source_artifact_id=aid,
                            confidence=score,
                            extraction_method=self._backend_name,
                        ))

        return all_mentions

    # ---- Extracteurs complémentaires ----

    def _extract_author(self, artifact: Artifact, mentions: List[Mention]):
        """Extraire l'auteur comme entité PERSON."""
        if not self.extract_from_metadata or not artifact.author:
            return
        author = artifact.author.strip()
        if author and author.lower() not in {'unknown', 'système', 'system', ''}:
            mentions.append(Mention(
                surface_form=author,
                entity_type=EntityType.PERSON,
                source_artifact_id=artifact.id,
                confidence=0.95,
                extraction_method="metadata",
            ))

    def _extract_structured(self, text: str, aid: str) -> List[Mention]:
        """Extraire emails, URLs, @handles, téléphones via regex."""
        mentions: List[Mention] = []

        for m in _EMAIL_RE.finditer(text):
            mentions.append(Mention(
                surface_form=m.group(), entity_type=EntityType.EMAIL,
                source_artifact_id=aid, confidence=0.99,
                extraction_method="regex",
            ))

        for m in _HANDLE_RE.finditer(text):
            mentions.append(Mention(
                surface_form=f"@{m.group(1)}", entity_type=EntityType.HANDLE,
                source_artifact_id=aid, confidence=0.90,
                extraction_method="regex",
            ))

        for m in _URL_RE.finditer(text):
            mentions.append(Mention(
                surface_form=m.group(), entity_type=EntityType.URL,
                source_artifact_id=aid, confidence=0.99,
                extraction_method="regex",
            ))

        for m in _PHONE_RE.finditer(text):
            phone = m.group().strip()
            digits = re.sub(r'\D', '', phone)
            if 7 <= len(digits) <= 15:
                if re.match(r'^20\d{6}$', digits):
                    continue
                mentions.append(Mention(
                    surface_form=phone, entity_type=EntityType.PHONE,
                    source_artifact_id=aid, confidence=0.80,
                    extraction_method="regex",
                ))

        return mentions

    def _from_existing_entities(self, artifact: Artifact) -> List[Mention]:
        """Backward compat : conserver les entités déjà dans l'artefact."""
        if not artifact.entities:
            return []
        return [
            Mention(
                surface_form=ent.strip(),
                entity_type=EntityType.CONCEPT,
                source_artifact_id=artifact.id,
                confidence=0.70,
                extraction_method="legacy",
            )
            for ent in artifact.entities
            if ent.strip()
        ]
