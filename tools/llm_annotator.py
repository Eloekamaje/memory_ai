"""
llm_annotator.py — Pré-annotation LLM des frontières d'épisodes.

Usage :
    python tools/llm_annotator.py data/mon_groupe.txt --out data/annotation_silver.json

    # Avec Claude API :
    ANTHROPIC_API_KEY=sk-... python tools/llm_annotator.py data/mon_groupe.txt

    # Avec Ollama local (gratuit) :
    python tools/llm_annotator.py data/mon_groupe.txt --backend ollama --model mistral

Sortie :
    annotation_silver.json — à passer ensuite à annotation_review.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsers.whatsapp_parser import parse_whatsapp_chat
from models import Artifact


# ══════════════════════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Tu es un expert en segmentation de conversations.
Tu dois identifier les frontières entre épisodes thématiques dans une conversation WhatsApp.

Un épisode = un sujet ou une activité cohérente et continue.
Exemples d'épisodes : planification d'un événement, discussion d'un projet, blague/humour, \
logistique quotidienne, nouvelles personnelles, décision collective.

Une frontière apparaît quand :
- Le sujet change clairement
- Une nouvelle activité démarre
- Un long silence sépare deux échanges de sujets différents
- Le registre change (sérieux → humour, personnel → professionnel)

Réponds UNIQUEMENT avec un JSON valide. Aucun texte avant ou après.
"""

USER_PROMPT_TEMPLATE = """\
Voici {n} messages consécutifs d'une conversation WhatsApp (groupe, français).
Les messages sont numérotés de {start} à {end}.

{messages}

Identifie les frontières d'épisodes dans cette séquence.
Une frontière à l'indice N signifie : "après le message N, le sujet change".

Réponds avec ce JSON exactement :
{{
  "boundaries": [liste des indices N où il y a une frontière],
  "explanations": {{
    "N": "explication en une phrase de pourquoi il y a une frontière ici"
  }}
}}

Si aucune frontière dans ce bloc : {{"boundaries": [], "explanations": {{}}}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Backends LLM
# ══════════════════════════════════════════════════════════════════════════════

def _build_claude_fn(model: str = "claude-sonnet-4-6"):
    try:
        import anthropic
    except ImportError:
        print("  pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(system: str, user: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    return call


def _build_ollama_fn(model: str = "mistral"):
    try:
        import ollama
    except ImportError:
        print("  pip install ollama", file=sys.stderr)
        sys.exit(1)

    def call(system: str, user: str) -> str:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response["message"]["content"]

    return call


# ══════════════════════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_artifacts(
    artifacts: List[Artifact],
    size: int = 50,
    overlap: int = 5,
) -> List[Tuple[int, int, List[Artifact]]]:
    """Découpe la liste en chunks (start_idx, end_idx, artifacts)."""
    chunks = []
    i = 0
    while i < len(artifacts):
        end = min(i + size, len(artifacts))
        chunks.append((i, end - 1, artifacts[i:end]))
        if end == len(artifacts):
            break
        i += size - overlap
    return chunks


def _format_messages(start: int, artifacts: List[Artifact]) -> str:
    lines = []
    for j, art in enumerate(artifacts):
        idx = start + j
        ts  = art.timestamp.strftime("%d/%m %H:%M")
        lines.append(f"[{idx}] {ts} {art.author}: {art.content}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Parsing de la réponse LLM
# ══════════════════════════════════════════════════════════════════════════════

def _parse_response(text: str) -> Tuple[List[int], dict]:
    """Extrait boundaries + explanations depuis la réponse LLM."""
    # Chercher le bloc JSON même si le LLM a ajouté du texte autour
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return [], {}
    try:
        data = json.loads(match.group())
        boundaries    = [int(b) for b in data.get("boundaries", [])]
        explanations  = {str(k): v for k, v in data.get("explanations", {}).items()}
        return boundaries, explanations
    except (json.JSONDecodeError, ValueError):
        return [], {}


# ══════════════════════════════════════════════════════════════════════════════
# Annotateur principal
# ══════════════════════════════════════════════════════════════════════════════

def annotate(
    artifacts: List[Artifact],
    llm_fn,
    chunk_size: int = 50,
    overlap: int = 5,
    delay: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Pré-annote la conversation avec le LLM.

    Returns
    -------
    dict avec :
      "artifacts"    : métadonnées des artefacts (index, timestamp, author, content)
      "boundaries"   : liste d'indices de frontières (triés, dédupliqués)
      "explanations" : {str(idx): explication}
    """
    chunks = _chunk_artifacts(artifacts, size=chunk_size, overlap=overlap)
    all_boundaries: List[int] = []
    all_explanations: dict = {}

    if verbose:
        print(f"  {len(artifacts)} messages → {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

    for ci, (start, end, chunk) in enumerate(chunks, 1):
        if verbose:
            print(f"  Chunk {ci}/{len(chunks)} (msgs {start}–{end}) ...", end=" ", flush=True)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            n=len(chunk),
            start=start,
            end=end,
            messages=_format_messages(start, chunk),
        )

        try:
            response = llm_fn(SYSTEM_PROMPT, user_prompt)
            boundaries, explanations = _parse_response(response)
            # Filtrer les frontières hors de la fenêtre courante
            boundaries = [b for b in boundaries if start <= b <= end]
            all_boundaries.extend(boundaries)
            all_explanations.update(explanations)
            if verbose:
                print(f"{len(boundaries)} frontières")
        except Exception as e:
            if verbose:
                print(f"ERREUR: {e}")

        if delay > 0:
            time.sleep(delay)

    # Dédupliquer et trier
    all_boundaries = sorted(set(all_boundaries))

    # Sérialiser les artefacts (pour annotation_review)
    artifacts_data = [
        {
            "idx":       i,
            "timestamp": art.timestamp.isoformat(),
            "author":    art.author,
            "content":   art.content,
        }
        for i, art in enumerate(artifacts)
    ]

    return {
        "artifacts":    artifacts_data,
        "boundaries":   all_boundaries,
        "explanations": all_explanations,
        "meta": {
            "n_messages":  len(artifacts),
            "n_boundaries": len(all_boundaries),
            "chunk_size":  chunk_size,
            "overlap":     overlap,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pré-annotation LLM des frontières d'épisodes (WhatsApp)"
    )
    parser.add_argument("input",  help="Fichier WhatsApp .txt")
    parser.add_argument("--out",  default=None, help="Fichier JSON de sortie (défaut: <input>_silver.json)")
    parser.add_argument("--backend", choices=["claude", "ollama"], default="claude")
    parser.add_argument("--model",   default=None, help="Nom du modèle (ex: mistral, claude-sonnet-4-6)")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--overlap",    type=int, default=5)
    parser.add_argument("--delay",      type=float, default=0.5, help="Pause entre chunks (secondes)")
    args = parser.parse_args()

    # LLM
    if args.backend == "claude":
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("Erreur : ANTHROPIC_API_KEY non définie.", file=sys.stderr)
            sys.exit(1)
        model = args.model or "claude-sonnet-4-6"
        llm_fn = _build_claude_fn(model)
        print(f"Backend : Claude ({model})")
    else:
        model = args.model or "mistral"
        llm_fn = _build_ollama_fn(model)
        print(f"Backend : Ollama ({model})")

    # Parse
    print(f"Parsing {args.input} ...")
    artifacts = parse_whatsapp_chat(args.input)
    print(f"  {len(artifacts)} messages chargés")

    # Annotation
    print("Annotation en cours ...")
    result = annotate(
        artifacts,
        llm_fn,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        delay=args.delay,
    )

    # Sauvegarde
    out_path = args.out or Path(args.input).stem + "_silver.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nAnnotation silver sauvegardée : {out_path}")
    print(f"  {result['meta']['n_messages']} messages")
    print(f"  {result['meta']['n_boundaries']} frontières proposées")
    print(f"\nProchaine étape :")
    print(f"  python tools/annotation_review.py {out_path}")


if __name__ == "__main__":
    main()
