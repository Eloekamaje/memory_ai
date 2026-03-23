"""
llm_ner.py — NER unifié via Qwen2.5-1.5B-Instruct GGUF (few-shot).

Remplace le duo GLiNER (batch, 2.2 GB) + spaCy (query, 200 MB)
par un seul modèle GGUF (~1.1 GB) utilisé en few-shot prompting.

Avantages :
    - NER identique batch ↔ query → plus de mismatch d'entités
    - Multilingue natif (FR/EN mélangé OK)
    - Types d'entités personnalisables dans le prompt
    - Zéro dépendance lourde (pas de spaCy, pas de GLiNER)

Types extraits : PERSON, ORG, PLACE, PROJECT, EVENT, PRODUCT,
                 DATE_EVENT, DOCUMENT

Interface :
    - Compatible NERBackend (extract_batch)
    - Compatible EntityExtractor (drop-in backend="llm")
    - extract_query_entities(query) → Set[str] pour RecallEngine

Usage :
    from llm_ner import LLMNERBackend

    backend = LLMNERBackend("path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    results = backend.extract_batch(["Hier j'ai vu Marie à Paris"])
    # → [[{'text': 'Marie', 'label': 'person', 'score': 0.9},
    #      {'text': 'Paris', 'label': 'location', 'score': 0.9}]]
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from models import Artifact

# Lazy import — llama_cpp peut être absent en CI
_llm_instance: Optional[object] = None


# ================================================================== #
# Prompt engineering — Few-shot NER                                    #
# ================================================================== #

SYSTEM_PROMPT = """\
Tu es un extracteur d'entités nommées expert. Pour chaque texte, \
extrais les entités nommées et retourne-les en JSON.

Types d'entités à extraire :
- person : nom de personne (prénom, nom complet, surnom)
- organization : entreprise, association, institution
- location : ville, pays, lieu physique
- project : nom de projet, d'app, de repo
- event : événement nommé, conférence, fête
- product : produit, service, outil nommé
- date_event : fête calendaire, anniversaire nommé
- document : document officiel, contrat, rapport nommé

Règles STRICTES :
- N'extrais que les NOMS PROPRES ou entités nommées spécifiques
- Ignore les noms communs, pronoms, verbes, adjectifs
- Ignore les mots trop courts (< 3 lettres)
- Ignore les URLs, emails, @handles (ils sont extraits séparément)
- Si aucune entité, retourne {"entities": []}

Réponds UNIQUEMENT par un JSON valide, rien d'autre :
{"entities": [{"text": "...", "type": "...", "confidence": 0.0-1.0}, ...]}"""

# Few-shot exemples couvrant FR, EN, et mélangé
FEW_SHOT_EXAMPLES = [
    {
        "input": "Hier j'ai eu un call avec Marie et Thomas du projet Ivias.",
        "output": '{"entities": [{"text": "Marie", "type": "person", "confidence": 0.95}, {"text": "Thomas", "type": "person", "confidence": 0.95}, {"text": "Ivias", "type": "project", "confidence": 0.90}]}'
    },
    {
        "input": "On se retrouve à la Gare de Lyon samedi pour le meetup Python.",
        "output": '{"entities": [{"text": "Gare de Lyon", "type": "location", "confidence": 0.90}, {"text": "meetup Python", "type": "event", "confidence": 0.85}]}'
    },
    {
        "input": "J'ai envoyé le rapport Q3 à Google France ce matin.",
        "output": '{"entities": [{"text": "Google France", "type": "organization", "confidence": 0.95}, {"text": "rapport Q3", "type": "document", "confidence": 0.80}]}'
    },
    {
        "input": "Ok cool, je check ça et je te dis",
        "output": '{"entities": []}'
    },
    {
        "input": "Sophie va bosser sur le refacto de TransNet avec l'équipe de Bordeaux",
        "output": '{"entities": [{"text": "Sophie", "type": "person", "confidence": 0.95}, {"text": "TransNet", "type": "project", "confidence": 0.90}, {"text": "Bordeaux", "type": "location", "confidence": 0.90}]}'
    },
]


# ================================================================== #
# Parsing robuste du JSON LLM                                         #
# ================================================================== #

# Regex pour extraire un JSON même entouré de texte
_JSON_RE = re.compile(r'\{[^{}]*"entities"\s*:\s*\[.*?\]\s*\}', re.DOTALL)


def _parse_ner_response(text: str) -> List[Dict]:
    """
    Parse la réponse du LLM en liste d'entités.

    Robuste face aux réponses non-conformes :
      - Essaie json.loads direct
      - Sinon cherche un JSON via regex
      - Sinon retourne []
    """
    text = text.strip()

    # 1. Essai direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "entities" in obj:
            return _validate_entities(obj["entities"])
    except json.JSONDecodeError:
        pass

    # 2. Regex extraction
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "entities" in obj:
                return _validate_entities(obj["entities"])
        except json.JSONDecodeError:
            pass

    # 3. Tentative array direct
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return _validate_entities(arr)
    except json.JSONDecodeError:
        pass

    return []


def _validate_entities(entities: list) -> List[Dict]:
    """Valide et normalise les entités extraites."""
    valid = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        text = ent.get("text", "").strip()
        etype = ent.get("type", "").lower().strip()
        conf = ent.get("confidence", 0.85)

        if not text or len(text) < 2:
            continue
        if not etype:
            continue

        # Normaliser les types
        etype = _normalize_type(etype)
        if etype is None:
            continue

        # Confiance numérique
        try:
            conf = float(conf)
        except (ValueError, TypeError):
            conf = 0.85

        valid.append({"text": text, "label": etype, "score": min(max(conf, 0.0), 1.0)})

    return valid


_TYPE_ALIASES = {
    "person": "person",
    "per": "person",
    "personne": "person",
    "nom": "person",
    "organization": "organization",
    "org": "organization",
    "organisation": "organization",
    "entreprise": "organization",
    "location": "location",
    "loc": "location",
    "lieu": "location",
    "place": "location",
    "ville": "location",
    "pays": "location",
    "project": "project",
    "projet": "project",
    "repo": "project",
    "event": "event",
    "événement": "event",
    "evenement": "event",
    "product": "product",
    "produit": "product",
    "outil": "product",
    "date_event": "date_event",
    "holiday": "date_event",
    "fete": "date_event",
    "fête": "date_event",
    "celebration": "date_event",
    "document": "document",
    "doc": "document",
    "rapport": "document",
    "concept": "concept",
}


def _normalize_type(raw: str) -> Optional[str]:
    """Normalise un type d'entité brut du LLM."""
    raw = raw.lower().strip().replace(" ", "_")
    return _TYPE_ALIASES.get(raw)


# ================================================================== #
# Backend LLM NER                                                     #
# ================================================================== #

class LLMNERBackend:
    """
    Backend NER utilisant Qwen2.5-1.5B-Instruct GGUF en few-shot.

    Compatible avec l'interface NERBackend de entity_extractor.py :
        extract_batch(texts: List[str]) → List[List[dict]]
        où chaque dict = {"text": ..., "label": ..., "score": ...}

    Parameters
    ----------
    model_path : str | Path
        Chemin vers le GGUF (qwen2.5-1.5b-instruct-q4_k_m.gguf).
    n_ctx : int
        Taille du contexte (2048 suffisant pour NER few-shot).
    n_threads : int
        Nombre de threads CPU.
    temperature : float
        Température d'échantillonnage (basse pour du JSON déterministe).
    max_tokens : int
        Tokens max pour la réponse NER (court → rapide).
    batch_concat : int
        Nombre de messages à concaténer par appel LLM (batching sémantique).
    verbose : bool
        Afficher les logs de chargement.
    """

    def __init__(
        self,
        model_path: str | Path = "",
        n_ctx: int = 2048,
        n_threads: int = 4,
        temperature: float = 0.1,
        max_tokens: int = 512,
        batch_concat: int = 5,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_concat = batch_concat
        self.verbose = verbose

        self._llm = None
        self._load_count = 0

    @property
    def name(self) -> str:
        return "llm"

    # ---------------------------------------------------------- #
    # Chargement lazy du modèle                                    #
    # ---------------------------------------------------------- #

    def _ensure_loaded(self):
        """Charge le modèle GGUF si pas encore fait."""
        global _llm_instance
        if self._llm is not None:
            return

        # Réutiliser l'instance globale si même modèle
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
            verbose=self.verbose,
            n_gpu_layers=0,  # CPU only
        )
        _llm_instance = self._llm

        if self.verbose:
            dt = time.time() - t0
            print(f"  ⚡ LLM NER chargé en {dt:.1f}s ({self.model_path.name})")

    # ---------------------------------------------------------- #
    # Build messages for chat completion                           #
    # ---------------------------------------------------------- #

    def _build_messages(self, text: str) -> list:
        """Construit la conversation few-shot + texte à analyser."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Few-shot examples
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})

        # Texte cible
        messages.append({"role": "user", "content": text})
        return messages

    # ---------------------------------------------------------- #
    # Extraction d'un batch de textes                              #
    # ---------------------------------------------------------- #

    def extract_batch(self, texts: List[str]) -> List[List[dict]]:
        """
        Extraire les entités de chaque texte.

        Pour optimiser le débit CPU, on concatène `batch_concat` messages
        en un seul prompt (séparés par des lignes numérotées), puis on
        parse les résultats.

        Returns: List[List[{text, label, score}]] — une liste par texte.
        """
        self._ensure_loaded()

        results: List[List[dict]] = []

        # Traiter par mini-batches concaténés
        for i in range(0, len(texts), self.batch_concat):
            chunk = texts[i : i + self.batch_concat]
            chunk_results = self._extract_chunk(chunk)
            results.extend(chunk_results)

        return results

    def _extract_chunk(self, texts: List[str]) -> List[List[dict]]:
        """Extrait les entités d'un chunk de textes concaténés."""
        if len(texts) == 1:
            return [self._extract_single(texts[0])]

        # Concaténer avec numéros de ligne pour guider le LLM
        numbered = "\n".join(
            f"[MSG{j+1}] {t[:300]}" for j, t in enumerate(texts)
        )

        prompt = (
            f"Extrais les entités de CHAQUE message ci-dessous.\n"
            f"Retourne un JSON avec une clé par message :\n"
            f'{{"MSG1": {{"entities": [...]}}, "MSG2": {{"entities": [...]}}, ...}}\n\n'
            f"{numbered}"
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Un seul few-shot pour le multi
        messages.append({"role": "user", "content": (
            "[MSG1] Marie part à Lyon demain\n"
            "[MSG2] Ok ça marche, bonne route"
        )})
        messages.append({"role": "assistant", "content": (
            '{"MSG1": {"entities": [{"text": "Marie", "type": "person", "confidence": 0.95}, '
            '{"text": "Lyon", "type": "location", "confidence": 0.90}]}, '
            '"MSG2": {"entities": []}}'
        )})
        messages.append({"role": "user", "content": prompt})

        raw = self._call_llm(messages)

        # Parse multi-message JSON
        return self._parse_multi_response(raw, len(texts))

    def _extract_single(self, text: str) -> List[dict]:
        """Extrait les entités d'un seul texte."""
        if not text or len(text.strip()) < 3:
            return []

        messages = self._build_messages(text[:500])  # tronquer les très longs
        raw = self._call_llm(messages)
        return _parse_ner_response(raw)

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
                print(f"  ⚠️  LLM NER error: {e}")
            return '{"entities": []}'

    # ---------------------------------------------------------- #
    # Parse multi-message response                                 #
    # ---------------------------------------------------------- #

    def _parse_multi_response(
        self, raw: str, expected_count: int
    ) -> List[List[dict]]:
        """Parse une réponse multi-message JSON."""
        raw = raw.strip()

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                results = []
                for i in range(expected_count):
                    key = f"MSG{i+1}"
                    msg_data = obj.get(key, {})
                    if isinstance(msg_data, dict):
                        entities = msg_data.get("entities", [])
                        results.append(_validate_entities(entities))
                    else:
                        results.append([])
                return results
        except json.JSONDecodeError:
            pass

        # Fallback : traiter tout comme un seul résultat
        entities = _parse_ner_response(raw)
        result = [entities] + [[] for _ in range(expected_count - 1)]
        return result

    # ---------------------------------------------------------- #
    # Extraction pour les requêtes (RecallEngine)                  #
    # ---------------------------------------------------------- #

    def extract_query_entities(self, query: str) -> Set[str]:
        """
        Extrait les entités d'une requête utilisateur.

        Retourne un Set[str] de noms d'entités en minuscules,
        directement utilisable par RecallEngine._extract_query_entities().

        C'est la méthode clé qui élimine le mismatch GLiNER ↔ spaCy :
        le même modèle traite batch ET query.
        """
        entities = self._extract_single(query)
        return {ent["text"].lower().strip() for ent in entities if ent.get("text")}

    # ---------------------------------------------------------- #
    # Nettoyage                                                    #
    # ---------------------------------------------------------- #

    def unload(self):
        """Libère le modèle de la mémoire."""
        global _llm_instance
        self._llm = None
        _llm_instance = None


# ================================================================== #
# Convenience : créer un backend avec le chemin par défaut             #
# ================================================================== #

DEFAULT_GGUF_PATH = Path(__file__).parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Chemin alternatif (dépôt mivias)
_ALT_GGUF_PATH = Path.home() / "ProjetsPerso" / "mivias" / "backend" / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def get_default_gguf_path() -> Path:
    """Retourne le premier chemin GGUF existant."""
    if DEFAULT_GGUF_PATH.exists():
        return DEFAULT_GGUF_PATH
    if _ALT_GGUF_PATH.exists():
        return _ALT_GGUF_PATH
    return DEFAULT_GGUF_PATH  # sera levé en FileNotFoundError au load


def create_llm_ner(
    model_path: Optional[str | Path] = None,
    verbose: bool = False,
    **kwargs,
) -> LLMNERBackend:
    """Factory pour créer un backend NER LLM avec des defaults raisonnables."""
    if model_path is None:
        model_path = get_default_gguf_path()
    return LLMNERBackend(model_path=model_path, verbose=verbose, **kwargs)
