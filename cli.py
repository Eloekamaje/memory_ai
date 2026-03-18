#!/usr/bin/env python3
"""
cli.py — Interface ligne de commande pour le Memory AI Lab.

Commandes :
  build   <data>   Construit la mémoire depuis un fichier de données
  recall  <query>  Rappelle les épisodes pertinents
  summary          Affiche un résumé de la mémoire
  chat             Session interactive avec LLM (nécessite ANTHROPIC_API_KEY)

Exemples :
  python cli.py build data/chat.txt
  python cli.py build data/chat.txt --store memory/ --archive-after 90
  python cli.py recall "GEDDVIT budget"
  python cli.py recall "segmentation" --top-k 3 --verbose
  python cli.py summary
  python cli.py chat
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

_STORE_HELP = "Répertoire de persistance (défaut: memory/)"
_MEMORY_FILE = "episodes.json"


# ================================================================== #
# Commande : build                                                    #
# ================================================================== #

def cmd_build(args):
    from memory_pipeline import build_pipeline

    if not Path(args.data).exists():
        print(f"Erreur : fichier introuvable — {args.data}", file=sys.stderr)
        sys.exit(1)

    print(f"\nConstruction de la mémoire depuis {args.data}")
    print("=" * 50)

    build_pipeline(
        data_path=args.data,
        store_dir=args.store,
        time_threshold_minutes=args.time_threshold,
        attach_threshold=args.attach_threshold,
        archive_after_days=args.archive_after,
        verbose=True,
    )

    print(f"\nMémoire sauvegardée dans {args.store}/")
    print("Utilise `python cli.py summary` pour voir l'état.")


# ================================================================== #
# Commande : recall                                                   #
# ================================================================== #

def cmd_recall(args):
    from memory_pipeline import load_pipeline

    store_path = Path(args.store)
    if not (store_path / _MEMORY_FILE).exists():
        print(f"Erreur : aucune mémoire dans {args.store}. Lance d'abord `build`.", file=sys.stderr)
        sys.exit(1)

    state = load_pipeline(store_dir=args.store, verbose=False)
    engine = state["engine"]

    results = engine.recall(args.query, top_k=args.top_k)

    if not results:
        print("Aucun épisode pertinent trouvé.")
        return

    print(f"\nRésultats pour : « {args.query} »")
    print("=" * 50)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {engine.explain(r, verbose=args.verbose)}")


# ================================================================== #
# Commande : summary                                                  #
# ================================================================== #

def cmd_summary(args):
    from memory_pipeline import load_pipeline

    store_path = Path(args.store)
    if not (store_path / _MEMORY_FILE).exists():
        print(f"Erreur : aucune mémoire dans {args.store}. Lance d'abord `build`.", file=sys.stderr)
        sys.exit(1)

    state = load_pipeline(store_dir=args.store, verbose=False)
    engine = state["engine"]
    store  = state["store"]
    graph  = state["graph"]

    print()
    print(store.summary(state["episodes"]))
    print()
    print(graph.summary())
    print()
    print(engine.active_memories_summary(top_k=10))


# ================================================================== #
# Commande : chat                                                     #
# ================================================================== #

def cmd_chat(args):
    from memory_pipeline import load_pipeline

    store_path = Path(args.store)
    if not (store_path / _MEMORY_FILE).exists():
        print(f"Erreur : aucune mémoire dans {args.store}. Lance d'abord `build`.", file=sys.stderr)
        sys.exit(1)

    # ── Construire llm_fn ──────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Erreur : variable d'environnement ANTHROPIC_API_KEY non définie.", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("Erreur : pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    model = args.model

    def llm_fn(system_prompt: str, user_message: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return msg.content[0].text

    # ── Charger pipeline ───────────────────────────────────────────
    print(f"\nChargement de la mémoire depuis {args.store}/ …")
    state = load_pipeline(store_dir=args.store, llm_fn=llm_fn, verbose=False)
    agent = state["agent"]
    n_ep = len(state["episodes"])

    print(f"Mémoire prête — {n_ep} épisodes indexés.")
    print(f"Modèle : {model}")
    print("Tape 'quit' ou Ctrl+C pour quitter. 'explain' pour voir les derniers souvenirs utilisés.\n")

    # ── Boucle interactive ─────────────────────────────────────────
    while True:
        try:
            user_input = input("Vous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir.")
            break
        if user_input.lower() == "explain":
            print(agent.explain_last())
            print()
            continue

        try:
            response = agent.respond(user_input)
            print(f"\nAssistant : {response}\n")
        except Exception as e:
            print(f"Erreur LLM : {e}", file=sys.stderr)


# ================================================================== #
# Parsing CLI                                                         #
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Memory AI Lab — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build ──────────────────────────────────────────────────────
    p_build = subparsers.add_parser("build", help="Construire la mémoire depuis les données")
    p_build.add_argument("data", help="Fichier source (.txt WhatsApp ou .csv)")
    p_build.add_argument("--store", default="memory", help=_STORE_HELP)
    p_build.add_argument("--time-threshold", type=float, default=120,
                         help="Gap temporel (min) pour la segmentation (défaut: 120)")
    p_build.add_argument("--attach-threshold", type=float, default=0.30,
                         help="Score minimum pour rattacher un artefact (défaut: 0.30)")
    p_build.add_argument("--archive-after", type=int, default=0,
                         help="Archiver les épisodes > N jours (0 = désactivé)")

    # ── recall ─────────────────────────────────────────────────────
    p_recall = subparsers.add_parser("recall", help="Rappeler des épisodes")
    p_recall.add_argument("query", help="Requête en langage naturel")
    p_recall.add_argument("--store", default="memory", help=_STORE_HELP)
    p_recall.add_argument("--top-k", type=int, default=5, help="Nombre de résultats (défaut: 5)")
    p_recall.add_argument("--verbose", action="store_true", help="Afficher les artefacts pertinents")

    # ── summary ────────────────────────────────────────────────────
    p_summary = subparsers.add_parser("summary", help="Afficher l'état de la mémoire")
    p_summary.add_argument("--store", default="memory", help=_STORE_HELP)

    # ── chat ───────────────────────────────────────────────────────
    p_chat = subparsers.add_parser("chat", help="Session interactive avec LLM")
    p_chat.add_argument("--store", default="memory", help=_STORE_HELP)
    p_chat.add_argument("--model", default="claude-sonnet-4-6",
                        help="Modèle Anthropic (défaut: claude-sonnet-4-6)")

    args = parser.parse_args()

    dispatch = {
        "build":   cmd_build,
        "recall":  cmd_recall,
        "summary": cmd_summary,
        "chat":    cmd_chat,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
