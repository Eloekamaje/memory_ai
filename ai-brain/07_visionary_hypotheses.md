# 07 — Hypothèses visionnaires : Memory Engine & Memory Agent

> Document spéculatif et fondateur. Pas de contraintes d'implémentation immédiate.
> C'est la boussole du projet Ivias/Mivias.

---

## La vérité architecturale

```
Memory Engine = State
Agent          = Policy
Decision Engine = Bridge
```

Ce n'est pas une métaphore. C'est la définition formelle du système.

Le Memory Engine seul, c'est un dashboard. L'Agent seul, c'est un LLM sans contexte. La valeur émerge de la boucle fermée entre les deux — et le Decision Engine est ce qui la ferme.

> **Un Memory Engine ne stocke pas des données. Il fournit l'état sur lequel une policy peut agir.**

---

## La boucle agentique fermée

```
1. OBSERVE
   Nouvel artefact, message, événement

2. UPDATE MEMORY
   → segmentation épisode
   → extraction entités
   → mise à jour graphe

3. EVALUATE MEMORY
   → importance
   → cohérence
   → nouveauté

4. DECIDE  ←── Decision Engine
   → KEEP / LINK / COMPRESS / ACTIVATE

5. ACT
   → répondre
   → suggérer
   → rappeler
   → déclencher action

6. LOOP  ←── retour à 1
```

Le Memory Engine intervient aux étapes 2, 3 et alimente 4.
L'Agent exécute 5.
Le Decision Engine est le cerveau de 4.

C'est une boucle fermée, pas un pipeline. **Memory → Agent → Memory → Agent → ...**

---

## Les deux côtés

### Memory Engine (cerveau)

Répond à :
- Qu'est-ce que je sais ?
- Qu'est-ce qui s'est passé ?
- Qu'est-ce qui est important ?

Il stocke, structure, relie, évolue, évalue. Principalement déterministe : ML + règles + scoring.

### Agent (système nerveux)

Répond à :
- Que dois-je faire maintenant ?

Il observe, raisonne, décide, agit. Principalement génératif : LLM + raisonnement + action.

> **LLM ≠ mémoire. LLM = interprétation / décision / génération.**

Le LLM ne remplace pas le Memory Engine. Il le consomme.

---

## Le Decision Engine — le vrai point de jonction

C'est lui qui transforme un état mémoire en action.

```
Memory state
  ├── Episode: "Lancer une app"
  ├── Entities: [budget, AWS, MVP]
  ├── Importance: élevé
  └── Last activity: récent

      ↓  Decision Engine

ACTION: ACTIVATE

      ↓  Agent

"Tu veux qu'on construise un MVP low-cost ensemble ?"
```

Sans Decision Engine, la mémoire reste passive. Sans mémoire, le Decision Engine est aveugle.

---

## Hypothèses architecturales

---

### H0 — Le Decision Engine doit être implémenté

**C'est la pièce manquante du pipeline actuel.**

Aujourd'hui : `MemoryAgent` reçoit une requête utilisateur → cherche dans la mémoire → LLM génère une réponse. C'est réactif.

Le Decision Engine rend le système proactif :

```python
class DecisionEngine:
    def evaluate(self, memory_state: MemoryState) -> List[Action]:
        """
        Transforme un état mémoire en décisions.
        Règles : seuils d'importance, patterns temporels, entités récurrentes.
        LLM : pour les cas ambigus, la nuance, la priorisation.
        """
```

Actions possibles : `ACTIVATE` (rappeler à l'utilisateur), `LINK` (connecter deux épisodes), `COMPRESS` (résumer un épisode vieux), `ARCHIVE`, `FLAG` (marquer comme non résolu).

*Implémentation minimale :* règles déterministes sur les scores d'importance + recency. Pas de GPU requis.

---

### H1 — L'épisode est une entité vivante

**H1.a — Un épisode a une trajectoire narrative**

Un épisode a un début (tension ou question ouverte), un développement, une fin (décision, abandon). Le système devrait détecter cette structure.

*Mécanisme :* variance des embeddings dans la fenêtre temporelle = signal de progression narrative. Pic de variance = rupture ou résolution.

**H1.b — Les épisodes se restructurent rétrospectivement**

Quand un nouvel épisode partage >2 entités avec un épisode ancien dont le label est faible, déclencher une révision backward. La mémoire réinterprète le passé à la lumière du présent — comme la mémoire humaine.

*Trigger :* `memory_graph.detect_retroactive_links(new_episode)` → révision des labels et connexions.

---

### H2 — Embeddings personnalisés, pas génériques

**H2.a — Bi-encoder fine-tuné sur nos données**

LaBSE est multilingue mais générique. Un bi-encoder entraîné avec contrastive loss sur des paires `(message, épisode correct)` apprendrait une métrique de similarité adaptée à notre domaine.

Gain estimé : **+15 à +25% ARI**.

*Coût GPU :* 3-5h sur RTX 3060 (Lambda Labs, ~10€). Dataset requis : ~500 paires annotées (4-6h de travail humain).

**H2.b — Fenêtre contextuelle : le message n'est pas isolé**

"Ok" signifie quelque chose de différent selon les 4 messages qui précèdent. Encoder une fenêtre de 5 messages au lieu d'un message isolé.

*Coût :* zéro GPU, changement de preprocessing uniquement.

---

### H3 — Segmentation apprise, pas calibrée

**H3.a — MLP AttachScore**

Les poids α, β, γ, δ, ρ sont calibrés à la main sur synthetic_200. Un MLP 5→16→1 entraîné sur des paires `(features, attach/no-attach)` apprendrait ces poids sans surapprentissage par construction.

*Entraînable sur CPU en minutes.* Dataset : annotés depuis les épisodes réels (H6.a).

**H3.b — Soft assignment probabiliste**

Le seuil `attach_threshold=0.30` est arbitraire. Un modèle probabiliste produirait `P(a ∈ e_k)` pour chaque épisode candidat. Les artefacts ambigus créent des connexions faibles, pas des ruptures forcées.

*Bénéfice :* le recall retourne des épisodes partiellement liés avec un score de confiance réel.

**H3.c — Segmentation comme classification de frontières**

Reformuler : "est-ce que le message $t$ est une frontière d'épisode ?" — tâche binaire, apprenante, évaluable en F1. Adapter l'approche BERTSeg avec nos features spécifiques (entités, temporalité, goals).

---

### H4 — Le Memory Agent comme partenaire cognitif actif

**H4.a — Recall proactif**

Aujourd'hui : l'utilisateur pose une question → le système cherche. Demain : le système *propose* sans qu'on lui demande.

```
Nouveau message entrant
  → index search sur mémoire existante
  → si score > seuil : "Ce message ressemble à une conversation
     du 14 mars sur le même budget AWS. Tu veux récupérer le contexte ?"
```

*Implémentation :* trigger sur `memory_index.search()` à chaque ingestion. CPU only.

**H4.b — Mémoire sémantique émergente**

Au-delà des épisodes (mémoire épisodique), des *faits stables* émergent des répétitions : "Jean travaille chez GEDDVIT", "le budget X est bloqué depuis mars". Ces faits persistent même quand les épisodes qui les ont générés sont compressés.

```
Épisodes répétés (N occurrences d'une entité/relation)
    ↓
Extracteur de faits (REBEL, ~400MB, inference CPU ~200ms)
    ↓
Knowledge Graph personnel (NetworkX + RDF-lite)
    ↓
Recall hybride : épisodique + sémantique
```

**H4.c — L'agent explique son raisonnement**

Un agent qui dit "j'ai trouvé ça" sans expliquer pourquoi n'est pas fiable. La trace de raisonnement est obligatoire :

```
[RECALL] Requête : "la réunion budget GEDDVIT"
  → Épisode 12 (score 0.87) : "Budget GEDDVIT Q3" — 14 mars
     Raison : entités [GEDDVIT, budget], similarité 0.91
  → Épisode 7 (score 0.61) : "Réunion équipe" — 2 mars
     Raison : entité [GEDDVIT], temporalité 0.72
```

`recall_engine.explain()` est déjà là. Le pousser jusqu'à l'API et l'UI.

---

### H5 — Index et pipeline incrémental

**H5.a — FAISS pour 1M artefacts**

sklearn NearestNeighbors tient jusqu'à ~100k épisodes. Au-delà : FAISS IVF-PQ.

*Migration :* quand `n_episodes > 5000`. FAISS tourne sur CPU, GPU accélère seulement le build (10x, optionnel).

**H5.b — Pipeline fully incremental**

Aujourd'hui `build_pipeline()` repart de zéro. La cible :

```
Nouveau message
  → EpisodeSegmenter.update(artifact)    # pas de rebuild
  → MemoryIndex.update(episode_centroid) # ajout partiel
  → MemoryGraph.update(entities)         # diff graph
  → MemoryStore.append(artifact)         # append
```

*Zéro GPU. Nécessite de rendre toutes les structures incrementales.*

---

### H6 — Évaluation comme fondation scientifique

**H6.a — Annoter 200 épisodes manuellement**

C'est le vrai déblocage. 4-6h de travail humain → ground truth réelle → fine-tuning possible → benchmark publiable. Outil : script notebook qui affiche les messages et pose "même épisode ?" → JSON.

**H6.b — Métriques produit, pas juste ARI**

L'ARI mesure la segmentation, pas l'utilité réelle.

| Métrique | Ce qu'elle mesure |
|----------|------------------|
| ARI | Qualité de la segmentation |
| Recall@k | La bonne réponse est-elle dans les k résultats ? |
| MRR | À quel rang apparaît la bonne réponse ? |
| Latence P95 | 95% des recalls en < X ms |
| Satisfaction humaine | Label pertinent ? Rappel utile ? |

---

### H7 — GPU comme accélérateur ponctuel, pas dépendance

**Règle d'or : tout ce qui est en production tourne sur CPU. Le GPU n'intervient que pour l'entraînement.**

| Tâche | Solution | Coût estimé |
|-------|----------|-------------|
| Fine-tuning bi-encoder | Google Colab Pro (A100) | ~10€ / session |
| Fine-tuning DistilBERT | Lambda Labs (RTX 3090) | ~10€ (5h × 2€/h) |
| Inference quotidienne | CPU local | 0€ |
| FAISS index build >100k | Lambda Labs spot (A10) | ~0.5€ |

Budget GPU total estimé : **< 100€/an** pour un système de qualité publiable.

---

### H8 — Memory Engine comme infrastructure cognitive personnelle

**H8.a — Multi-source**

WhatsApp est un point d'entrée. La mémoire personnelle vit aussi dans les emails, notes Obsidian, fichiers, calendriers. L'architecture `parsers/` est déjà modulaire — chaque source est un parser, le reste du pipeline est commun.

Sources à brancher : Gmail (MBOX), Obsidian (Markdown), Notion (JSON), calendrier (ICS), terminal history.

**H8.b — Privacy-first : tout en local**

Aucune donnée ne sort de la machine. Les LLMs locaux (Ollama + Mistral 7B ou Phi-3 mini) remplacent l'API pour le summarization et l'agent en mode offline. L'API Claude reste optionnelle pour la qualité maximale.

*Benchmark :* Phi-3 mini sur CPU = ~4 tokens/s. Suffisant pour du summarization batch.

**H8.c — Vision terminale : l'OS de la connaissance personnelle**

Le Memory Agent n'est plus un outil qu'on lance manuellement. C'est un processus background qui :

1. Ingère les nouveaux artefacts en temps réel (inotify / webhook)
2. Met à jour le graphe incrémentalement
3. Propose des rappels proactifs en contexte (Decision Engine actif)
4. Répond aux questions sur le passé personnel
5. Identifie les patterns à long terme ("tu travailles toujours sur AWS le dimanche")

**Ce n'est pas un chatbot. C'est une couche cognitive augmentée sur l'expérience personnelle.**

---

## Roadmap vers la vision

```
Court terme (< 1 mois)
  ├── LaBSE + GLiNER                     plan V4 — 2 jours
  ├── SuperDialseg evaluation             plan V4 — 1 jour
  ├── 5-fold CV + variance report         protocole — 1 jour
  └── Decision Engine v0 (règles)         H0 — 1 semaine

Moyen terme (1-3 mois)
  ├── Annotation manuelle 200 épisodes    H6.a
  ├── Embedding contextuel 5-window       H2.b — CPU
  ├── MLP AttachScore appris              H3.a — CPU
  ├── Recall proactif background          H4.a
  └── Pipeline incremental               H5.b

Long terme (3-12 mois)
  ├── Bi-encoder fine-tuné               H2.a — 1 session Colab (~10€)
  ├── Soft assignment probabiliste        H3.b
  ├── Knowledge Graph sémantique          H4.b
  ├── FAISS à 1M artefacts               H5.a
  └── Multi-source (email, notes, cal)   H8.a

Vision (12+ mois)
  └── OS de la connaissance personnelle  H8.c
```

---

> **L'insight fondateur :**
>
> La mémoire fournit le contexte structuré.
> L'agent transforme ce contexte en action.
>
> Sans contexte → agent stupide.
> Sans action → mémoire inutile.
>
> C'est ça qui fait passer de "AI qui répond" à "AI qui comprend et agit".
