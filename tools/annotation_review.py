"""
annotation_review.py — Interface de review humaine des annotations LLM.

Usage :
    python tools/annotation_review.py data/mon_groupe_silver.json

Navigation :
    ENTRÉE        valider la frontière proposée
    n + ENTRÉE    supprimer la frontière (pas de coupure ici)
    a + ENTRÉE    ajouter une frontière avant ce message
    q + ENTRÉE    quitter et sauvegarder

Les frontières avec un gap temporel >= 2h sont auto-approuvées.
Seules les frontières ambiguës (même jour) sont présentées à l'humain.

Sortie :
    <input>_gold.json  — annotation finale prête pour l'entraînement
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Affichage
# ══════════════════════════════════════════════════════════════════════════════

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _green(t):  return _color(t, "92")
def _yellow(t): return _color(t, "93")
def _red(t):    return _color(t, "91")
def _cyan(t):   return _color(t, "96")
def _bold(t):   return _color(t, "1")
def _dim(t):    return _color(t, "2")


def _format_ts(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m %H:%M")
    except ValueError:
        return iso[:16]


def _print_message(idx: int, msg: dict, highlight: bool = False):
    ts      = _format_ts(msg["timestamp"])
    author  = msg["author"][:15].ljust(15)
    content = msg["content"]
    if len(content) > 120:
        content = content[:117] + "..."
    prefix = _bold(f"[{idx:4d}]") if highlight else _dim(f"[{idx:4d}]")
    print(f"  {prefix} {_dim(ts)} {_cyan(author)} {content}")


def _print_boundary_marker(explanation: Optional[str]):
    bar   = "─" * 70
    label = _green("▶ FRONTIÈRE PROPOSÉE")
    if explanation:
        print(f"\n  {label}")
        print(f"  {_yellow('Raison :')} {_dim(explanation)}")
        print(f"  {_green(bar)}\n")
    else:
        print(f"\n  {label} {_green(bar)}\n")


def _print_confirmed_boundary(idx: int):
    print(f"  {_green('✓ FRONTIÈRE CONFIRMÉE')} après message {idx}")
    print(f"  {'─' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
# Logique de confiance temporelle
# ══════════════════════════════════════════════════════════════════════════════

def _gap_hours(artifacts: list, boundary: int) -> float:
    if boundary + 1 >= len(artifacts):
        return 999.0
    try:
        t1 = datetime.fromisoformat(artifacts[boundary]["timestamp"])
        t2 = datetime.fromisoformat(artifacts[boundary + 1]["timestamp"])
        return (t2 - t1).total_seconds() / 3600
    except (ValueError, KeyError):
        return 0.0


def _split_by_confidence(
    artifacts: list,
    boundaries: list,
    auto_threshold_hours: float = 2.0,
) -> tuple:
    """
    Sépare les frontières en :
    - auto_approve : gap >= seuil (quasi-certaines, pas besoin de review)
    - to_review    : gap <  seuil (décision thématique ambiguë)
    """
    auto_approve, to_review = [], []
    for b in boundaries:
        if _gap_hours(artifacts, b) >= auto_threshold_hours:
            auto_approve.append(b)
        else:
            to_review.append(b)
    return auto_approve, to_review


# ══════════════════════════════════════════════════════════════════════════════
# Logique de review
# ══════════════════════════════════════════════════════════════════════════════

def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "q"


def _add_manual_boundary(confirmed: List[int], artifacts: list, ctx_start: int, boundary: int):
    n = len(artifacts)
    raw = _ask(f"  Après quel message ajouter une frontière ? ({ctx_start}–{boundary}) : ")
    try:
        manual = int(raw)
        if 0 <= manual < n and manual not in confirmed:
            confirmed.append(manual)
            confirmed.sort()
            print(f"  {_green('✓')} Frontière manuelle ajoutée après message {manual}")
        else:
            print("  Indice invalide ou déjà présent.")
    except ValueError:
        print("  Index invalide, ignoré.")


def _collect_extra_boundaries(confirmed: List[int], n: int):
    more = _ask("  Ajouter des frontières manuelles supplémentaires ? (o/n) : ")
    while more == "o":
        raw = _ask(f"  Après quel message (0–{n-2}) ? : ")
        try:
            manual = int(raw)
            if 0 <= manual < n - 1 and manual not in confirmed:
                confirmed.append(manual)
                confirmed.sort()
                print(f"  {_green('✓')} Frontière ajoutée après {manual}")
            else:
                print("  Indice invalide ou déjà présent.")
        except ValueError:
            print("  Invalide.")
        more = _ask("  Encore une ? (o/n) : ")


def _review_one(
    boundary: int,
    i: int,
    total: int,
    artifacts: list,
    explanations: dict,
    confirmed: List[int],
) -> str:
    """Affiche une frontière et retourne la décision ('q', 'n', 'a', ou 'ok')."""
    expl = explanations.get(str(boundary))
    n    = len(artifacts)

    _clear()
    print(_bold(f"\n  Frontière {i+1}/{total} — après message {boundary}"))
    print(_dim(f"  Confirmées jusqu'ici : {len(confirmed)}"))
    print()

    ctx_start = max(0, boundary - 4)
    for j in range(ctx_start, boundary + 1):
        _print_message(j, artifacts[j], highlight=(j == boundary))

    _print_boundary_marker(expl)

    ctx_end = min(n, boundary + 6)
    for j in range(boundary + 1, ctx_end):
        _print_message(j, artifacts[j], highlight=(j == boundary + 1))

    print()
    print(_bold("  → ") + "ENTRÉE=valider  n=supprimer  a=ajouter avant  q=quitter")
    ans = _ask("  Choix : ")

    if ans == "a":
        _add_manual_boundary(confirmed, artifacts, ctx_start, boundary)
        return "a"
    return ans


def review_boundaries(data: dict, auto_threshold_hours: float = 2.0) -> dict:
    """
    Interface interactive pour valider/corriger les frontières LLM.

    Les frontières avec gap >= auto_threshold_hours sont auto-approuvées.
    Seules les frontières ambiguës sont présentées à l'humain.
    """
    artifacts    = data["artifacts"]
    proposed     = sorted(set(data["boundaries"]))
    explanations = data.get("explanations", {})
    n            = len(artifacts)

    auto_approved, to_review = _split_by_confidence(artifacts, proposed, auto_threshold_hours)
    confirmed: List[int] = list(auto_approved)

    print(_bold("\n  === ANNOTATION REVIEW ==="))
    print(f"  {n} messages · {len(proposed)} frontières proposées")
    print(f"  {_green(str(len(auto_approved)))} auto-approuvées (gap >= {auto_threshold_hours}h)")
    print(f"  {_yellow(str(len(to_review)))} à reviewer (ambiguës)\n")
    print("  Commandes : ENTRÉE=valider · n=supprimer · a=ajouter avant · q=quitter\n")
    input("  Appuie sur ENTRÉE pour commencer...")

    i = 0
    while i < len(to_review):
        boundary = to_review[i]
        ans = _review_one(boundary, i, len(to_review), artifacts, explanations, confirmed)

        if ans == "q":
            print("\n  Sauvegarde en cours...")
            break
        elif ans == "n":
            i += 1
        elif ans == "a":
            pass  # frontière ajoutée, on re-présente la même
        else:
            confirmed.append(boundary)
            _print_confirmed_boundary(boundary)
            i += 1

    print()
    _collect_extra_boundaries(confirmed, n)

    confirmed = sorted(set(confirmed))
    episodes  = _boundaries_to_episodes(artifacts, confirmed)

    return {
        "artifacts":  artifacts,
        "boundaries": confirmed,
        "episodes":   episodes,
        "meta": {
            **data.get("meta", {}),
            "n_boundaries_proposed":   len(proposed),
            "n_boundaries_auto":       len(auto_approved),
            "n_boundaries_confirmed":  len(confirmed),
            "n_episodes":              len(episodes),
            "auto_threshold_hours":    auto_threshold_hours,
            "reviewed":                True,
        },
    }


def _boundaries_to_episodes(artifacts: List[dict], boundaries: List[int]) -> List[dict]:
    cuts   = sorted(set(boundaries))
    starts = [0] + [b + 1 for b in cuts]
    ends   = cuts + [len(artifacts) - 1]

    episodes = []
    for ep_id, (s, e) in enumerate(zip(starts, ends)):
        if s > e:
            continue
        episodes.append({
            "episode_id": ep_id,
            "start_idx":  s,
            "end_idx":    e,
            "n_messages": e - s + 1,
            "start_ts":   artifacts[s]["timestamp"],
            "end_ts":     artifacts[e]["timestamp"],
        })
    return episodes


# ══════════════════════════════════════════════════════════════════════════════
# Statistiques finales
# ══════════════════════════════════════════════════════════════════════════════

def _print_stats(gold: dict):
    episodes = gold["episodes"]
    n_ep     = len(episodes)
    sizes    = [e["n_messages"] for e in episodes]
    avg_size = sum(sizes) / n_ep if n_ep else 0

    print(_bold("\n  ═══ ANNOTATION TERMINÉE ═══"))
    print(f"  Messages   : {gold['meta'].get('n_messages', '?')}")
    print(f"  Frontières : {gold['meta']['n_boundaries_confirmed']} confirmées "
          f"(sur {gold['meta']['n_boundaries_proposed']} proposées)")
    print(f"  Auto       : {gold['meta']['n_boundaries_auto']} (gap >= {gold['meta']['auto_threshold_hours']}h)")
    print(f"  Épisodes   : {n_ep}")
    print(f"  Taille moy : {avg_size:.1f} messages/épisode")
    print(f"  Min/Max    : {min(sizes) if sizes else 0} / {max(sizes) if sizes else 0} messages")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage : python tools/annotation_review.py <silver.json> [--out gold.json] [--auto-hours N]")
        sys.exit(1)

    silver_path = sys.argv[1]

    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    else:
        out_path = silver_path.replace("_silver.json", "_gold.json")
        if out_path == silver_path:
            out_path = Path(silver_path).stem + "_gold.json"

    auto_hours = 2.0
    if "--auto-hours" in sys.argv:
        auto_hours = float(sys.argv[sys.argv.index("--auto-hours") + 1])

    with open(silver_path, encoding="utf-8") as f:
        data = json.load(f)

    gold = review_boundaries(data, auto_threshold_hours=auto_hours)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

    _print_stats(gold)
    print(f"\n  Annotation gold sauvegardée : {_green(out_path)}")
    print()
    print("  Prochaines étapes :")
    print("    → Ce fichier sert de ground truth pour évaluer l'ARI sur données réelles")
    print("    → Il peut aussi servir à fine-tuner le bi-encoder (H2a)")


if __name__ == "__main__":
    main()
