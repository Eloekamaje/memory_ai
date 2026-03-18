"""
memory_agent.py — Interface LLM avec mémoire épisodique (Phase B6).

Le MemoryAgent orchestre :
  1. recall(user_message)     — récupère les épisodes pertinents
  2. _build_context(results)  — formate le contexte mémoire (tiered)
  3. llm_fn(system, user_msg) — appel LLM injecté (Anthropic, OpenAI, local…)
  4. _store_turn(msg, reply)  — optionnel : persiste l'échange comme Artifact

Le LLM est entièrement injectable via `llm_fn(system_prompt, user_message) → str`.
Aucune dépendance à un SDK spécifique.

Usage rapide :
    import anthropic
    client = anthropic.Anthropic()

    def llm_fn(system, user):
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    agent = MemoryAgent(recall_engine=engine, llm_fn=llm_fn)
    response = agent.respond("Où en est-on avec GEDDVIT ?")
    print(response)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from recall_engine import RecallEngine, RecallResult


# ================================================================== #
# Format du contexte mémoire                                          #
# ================================================================== #

SYSTEM_TEMPLATE = """\
Tu es un assistant doté d'une mémoire épisodique structurée.
Tu as accès aux souvenirs les plus pertinents pour répondre à l'utilisateur.

{memory_context}

---
Instructions :
- Appuie-toi sur les souvenirs ci-dessus quand ils sont pertinents.
- Si aucun souvenir n'est pertinent, réponds normalement sans les mentionner.
- Ne cite pas les identifiants techniques (EP_XXXX).
- Parle des souvenirs de façon naturelle ("la dernière fois qu'on a parlé de X…").
"""


# ================================================================== #
# Résultat d'un tour de conversation                                  #
# ================================================================== #

@dataclass
class AgentTurn:
    user_message: str
    agent_response: str
    recalled_episodes: List[str] = field(default_factory=list)
    context_tokens_estimate: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# ================================================================== #
# MemoryAgent                                                          #
# ================================================================== #

class MemoryAgent:
    """
    Agent conversationnel avec mémoire épisodique.

    Parameters
    ----------
    recall_engine : RecallEngine configuré (avec épisodes + embeddings)
    llm_fn : callable(system_prompt: str, user_message: str) -> str
             Injecte n'importe quel LLM (Anthropic, OpenAI, Ollama…)
    top_k_recall : nombre d'épisodes rappelés par requête
    max_context_chars : budget caractères pour le contexte mémoire
                        (~4 chars/token, 8000 chars ≈ 2000 tokens)
    system_template : override du template système si besoin
    """

    def __init__(
        self,
        recall_engine: RecallEngine,
        llm_fn: Callable[[str, str], str],
        top_k_recall: int = 7,
        max_context_chars: int = 8_000,
        system_template: str = SYSTEM_TEMPLATE,
    ):
        self.engine = recall_engine
        self.llm_fn = llm_fn
        self.top_k_recall = top_k_recall
        self.max_context_chars = max_context_chars
        self.system_template = system_template

        self.history: List[AgentTurn] = []

    # -------------------------------------------------------------- #
    # Interface principale                                             #
    # -------------------------------------------------------------- #

    def respond(self, user_message: str) -> str:
        """
        Répond au message utilisateur en injectant la mémoire pertinente.

        Returns
        -------
        Réponse textuelle du LLM.
        """
        results = self.engine.recall(user_message, top_k=self.top_k_recall)
        context = self._build_context(results)
        system_prompt = self.system_template.format(memory_context=context)

        response = self.llm_fn(system_prompt, user_message)

        self.history.append(AgentTurn(
            user_message=user_message,
            agent_response=response,
            recalled_episodes=[r.episode_id for r in results],
            context_tokens_estimate=len(context) // 4,
        ))

        return response

    # -------------------------------------------------------------- #
    # Construction du contexte mémoire (tiered)                       #
    # -------------------------------------------------------------- #

    def _build_context(self, results: List[RecallResult]) -> str:
        """
        Formate les épisodes rappelés en contexte pour le LLM.

        Stratégie tiered (budget max_context_chars) :
          Tier 1 — top 3 épisodes : résumé complet (label + summary + entités + quotes)
          Tier 2 — épisodes 4-7   : une ligne (label + date + entités)
          Tier 3 — reste           : ligne agrégée "autres souvenirs : X, Y, Z"

        Si aucun épisode n'est rappelé, retourne un message neutre.
        """
        if not results:
            return "## Mémoire\nAucun souvenir pertinent trouvé pour cette requête."

        blocks: List[str] = []
        budget = self.max_context_chars

        for i, r in enumerate(results):
            summary = self.engine.summary_map.get(r.episode_id)

            if i < 3:
                block = self._format_full(r, summary)
            elif i < 7:
                block = self._format_oneliner(r, summary)
            else:
                # Tier 3 : agréger le reste en une seule ligne
                remaining = results[i:]
                labels = []
                for rem in remaining:
                    s = self.engine.summary_map.get(rem.episode_id)
                    labels.append(s.label if s else rem.episode_id)
                blocks.append(f"*Autres souvenirs pertinents : {', '.join(labels)}.*")
                break

            if budget - len(block) < 200:
                # Budget épuisé : résumé minimal
                blocks.append(self._format_oneliner(r, summary))
            else:
                blocks.append(block)
                budget -= len(block)

        return "## Mémoire épisodique\n\n" + "\n\n".join(blocks)

    def _format_full(self, r: RecallResult, summary) -> str:
        """Bloc complet pour un épisode (Tier 1)."""
        lines = []

        # En-tête : label ou id
        label = summary.label if summary else r.episode_id
        date_str = r.time_interval[0].strftime("%d %b %Y") if r.time_interval else "?"
        lines.append(f"**{label}** ({date_str})")

        # Résumé narratif
        if summary and summary.summary:
            lines.append(summary.summary)

        # Entités saillantes
        if r.matching_entities:
            lines.append(f"Personnes/sujets : {', '.join(r.matching_entities[:5])}")
        elif summary and summary.top_entities:
            lines.append(f"Personnes/sujets : {', '.join(summary.top_entities[:4])}")

        # Citations représentatives (si dispo dans le summary)
        if summary and summary.representative_quotes:
            quote = summary.representative_quotes[0][:120]
            lines.append(f"› *{quote}*")

        # Fallback : top artifact preview
        elif r.top_artifacts:
            preview = r.top_artifacts[0][1][:120]
            lines.append(f"› *{preview}*")

        return "\n".join(lines)

    def _format_oneliner(self, r: RecallResult, summary) -> str:
        """Ligne unique pour un épisode (Tier 2)."""
        label = summary.label if summary else r.episode_id
        date_str = r.time_interval[0].strftime("%d %b %Y") if r.time_interval else "?"
        entities = ", ".join(r.matching_entities[:3]) if r.matching_entities else ""
        entity_part = f" — {entities}" if entities else ""
        return f"- **{label}** ({date_str}){entity_part}"

    # -------------------------------------------------------------- #
    # Debug / inspection                                               #
    # -------------------------------------------------------------- #

    def explain_last(self) -> str:
        """Explique les souvenirs utilisés lors du dernier tour."""
        if not self.history:
            return "Aucun tour de conversation."

        turn = self.history[-1]
        lines = [
            f"Tour : {turn.timestamp.strftime('%H:%M:%S')}",
            f"Message : {turn.user_message[:80]}",
            f"Épisodes rappelés ({len(turn.recalled_episodes)}) :",
        ]
        for ep_id in turn.recalled_episodes:
            s = self.engine.summary_map.get(ep_id)
            label = s.label if s else ep_id
            lines.append(f"  - {ep_id} : {label}")
        lines.append(f"Contexte estimé : ~{turn.context_tokens_estimate} tokens")
        return "\n".join(lines)
