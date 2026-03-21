# 10 — Ivias × OpenClaw : Vision Produit & Architecture

> Rédigé le 2026-03-21
> Statut : Vision validée — à implémenter Phase B

---

## 1. Le problème fondamental

Le cerveau humain oublie. Pas par inattention — par design.
Des milliers d'échanges textuels quotidiens (WhatsApp, Signal, mail, SMS)
contiennent des décisions, des engagements, des idées fortes.
**Tout ça s'évapore.**

Aucun outil existant ne résout ça :

| Outil | Pourquoi ça ne marche pas |
|-------|--------------------------|
| Notion / Obsidian | Effort actif requis — 99% abandonnent |
| ChatGPT memory | Faits plats, pas d'épisodes, pas de temporalité |
| Rewind.ai | Screenshots sans structure ni sens |
| Apple Intelligence | Résumés de notifications, pas de mémoire inter-temporelle |
| OpenClaw (vanilla) | RAG sur Markdown — document retrieval, pas mémoire épisodique |

---

## 2. La vision Ivias

> *"Une extension de ta mémoire personnelle qui capture ce que ton cerveau
> laisse tomber, et qui parle quand c'est pertinent."*

Trois couches progressives :

```
Couche 1 — Mémoire vivante
  Tes conversations → épisodes structurés → cycle de vie → decay

Couche 2 — Jumeau numérique
  Mémoire + LLM = agent qui te représente et parle pour toi

Couche 3 — Réseau de jumeaux
  Ton jumeau ↔ jumeau de Marc ↔ jumeau de l'entreprise X
  → connexions pertinentes sans effort humain
```

---

## 3. OpenClaw comme couche de capture

**OpenClaw** (MIT License — Peter Steinberger) est un assistant IA personnel
open source qui tourne en local et connecte 21+ plateformes de messagerie
à un seul agent.

### Ce qu'OpenClaw apporte gratuitement

| Composant | Détail |
|-----------|--------|
| Canaux | WhatsApp, Signal, iMessage, Telegram, Slack, Discord, Matrix, Teams... |
| Apps mobiles | iOS (SwiftUI) + Android (Compose) |
| Gateway daemon | systemd/launchd, WebSocket local port 18789 |
| Auth / pairing | Sécurité, approbation des devices |
| Session + historique | JSONL par session |
| Plugin SDK | Hooks extensibles dans le loop agent |
| BYOK | OAuth Anthropic + OpenAI, API keys |

### Ce qu'Ivias apporte par-dessus

| Composant | Détail |
|-----------|--------|
| Segmentation épisodique | mE5 + AttachScore composite V4 B+C (ARI 0.90) |
| Structure épisodique | DeBERTa axis_scores (résolution / saillance / temporalité) |
| Cycle de vie | BORN → ACTIVE → DORMANT → COMPRESSED → ARCHIVED → REVIVED |
| Decay Ebbinghaus | retention(e,t) = exp(-λ(e) × t / (1 + log1p(access_count))) |
| Decision Engine | Surfacing proactif — le jumeau parle avant qu'on lui pose une question |
| Réseau de jumeaux | Protocole agent-to-agent inter-mémoires |

**OpenClaw = les yeux et les oreilles. Ivias = le cerveau et la mémoire.**

---

## 4. Architecture technique

### 4.1 Comment le téléphone se connecte au laptop (mode vanilla)

Il n'y a pas de sync — le téléphone ne stocke rien.

```
📱 Téléphone (app OpenClaw)
        │
        │ WebSocket
        │ ┌─ même WiFi → Bonjour/mDNS (auto-découverte)
        │ └─ hors WiFi → Tailscale (VPN personnel gratuit)
        │
💻 Laptop (Gateway OpenClaw + plugin Ivias)
  ├── Mémoire locale (memory.db)
  ├── Pipeline épisodique
  ├── Tous les canaux connectés
  └── LLM appelé via token BYOK user
```

### 4.2 Flux de traitement complet (production)

```
📱 Message entrant (WhatsApp / Signal / iMessage / email)
        ↓
OpenClaw hook: message_received
        ↓
Plugin Ivias — Preprocessor
  {message_id, channel, author_id, timestamp,
   content, conversation_id, direction,
   langue_estimée, marqueurs_discursifs, longueur}
        ↓
━━━━━━━━━━━━━━ FAST PATH (<50ms) ━━━━━━━━━━━━━━
  1. Charger épisode courant + son centroïd
  2. Extraire features (Feature Extractor) :
       A. Sémantiques :
          - embedding message (mE5-small ONNX en vanilla)
          - sim(msg, centroïd épisode)      ← clé
          - sim(msg, dernier message)
          - delta entre les deux
       B. Temporelles :
          - gap depuis dernier message
          - durée de l'épisode courant
       C. Discursives :
          - marqueurs de rupture ("au fait", "sinon", "bon")
          - changement d'auteur
          - changement de canal
       D. Structurelles :
          - réponse à un autre message
          - présence média
  3. Calculer boundary_score composite :
       boundary_score =
           w1 * (1 - sim(msg, centroïd))   ← semantic_break
         + w2 * temporal_break
         + w3 * intent_shift
         + w4 * entity_shift
         + w5 * discourse_break
       threshold_eff = attach_threshold
                     + boundary_k × (1 - P_boundary)
  4. Décision à double seuil :
       score < seuil_bas   → APPEND direct     (certain)
       seuil_bas ≤ score ≤ seuil_haut → SLOW PATH (ambigu)
       score > seuil_haut  → NEW direct         (certain)
       + cas spécial : REACTIVATE si épisode DORMANT
                       correspondant trouvé (différenciateur Ivias)
  5. Écriture atomique SQLite (transaction + lock conversation_id)
       - message sauvegardé
       - épisode créé / mis à jour
       - centroïd mis à jour : centroid = α*centroid + (1-α)*msg_emb
       - status = RAW
        ↓
━━━━━━━━━━━━━━ JOB QUEUE (async immédiat) ━━━━━━━━━━━━━━
  status = PENDING_ENRICHMENT
        ↓
━━━━━━━━━━━━━━ SLOW PATH (3-10s, background) ━━━━━━━━━━━━━━
  - LLM valide / corrige la frontière
  - Génère summary, goal, decisions
  - GLiNER extrait les entités
  - Recalcule axis_scores si besoin
  - Versionne chaque champ (voir section 4.4)
  status = ENRICHED  (ou FAILED_ENRICHMENT)
        ↓
━━━━━━━━━━━━━━ CRON PÉRIODIQUE (toutes les heures) ━━━━━━━━
  - Recalcule decay Ebbinghaus sur tous les ENRICHED
  - Transitions lifecycle (DORMANT, ARCHIVED)
  - Consolidation nocturne : liens causaux entre épisodes
        ↓
━━━━━━━━━━━━━━ AVANT RÉPONSE AGENT ━━━━━━━━━━━━━━━━━━━━━━━
OpenClaw hook: before_agent_start
  - Recherche hybride (vecteur + BM25 + axis_scores)
  - Ranking composite (voir section 4.5)
  - Top 3-7 épisodes max
  - Compression en Memory Context Pack
  - Injection dans le prompt agent
        ↓
Réponse enrichie → canal d'origine
```

### 4.3 Schéma de la base mémoire (SQLite)

```sql
episodes
  id            TEXT PRIMARY KEY,
  start_time    INTEGER,
  end_time      INTEGER,
  status        TEXT,   -- RAW/PENDING_ENRICHMENT/ENRICHED/
                        -- FAILED_ENRICHMENT/STALE/DORMANT/ARCHIVED
  message_count INTEGER,
  channels      TEXT,   -- JSON array ["whatsapp", "email"]
  participants  TEXT,   -- JSON array ["user", "marc"]

  -- Enrichissement versionné (slow path)
  title_v1           TEXT,   -- titre court généré par LLM
  summary_v1         TEXT,
  summary_model      TEXT,   -- "claude-opus-4-5"
  summary_updated_at INTEGER,
  goal_v1            TEXT,
  decisions_v1       TEXT,   -- JSON array
  main_intent        TEXT,   -- housing_dispute / travel_planning / ...

  -- Scores épisodiques (DeBERTa)
  axis_resolution  REAL,    -- [0,1]
  axis_salience    REAL,    -- [0,1]
  axis_temporality REAL,    -- [0,1]
  importance_score REAL,    -- score composite pour ranking

  -- Lifecycle Ebbinghaus
  decay_factor   REAL,
  access_count   INTEGER DEFAULT 0,
  last_accessed  INTEGER,

  -- Représentation vectorielle
  centroid_emb   BLOB,   -- EMA des embeddings messages (mE5 768d)
                         -- mise à jour à chaque APPEND :
                         -- centroid = α*centroid + (1-α)*msg_emb  (α≈0.85)
  embedding      BLOB    -- embedding du résumé (pour search sémantique)

messages
  id             TEXT PRIMARY KEY,
  episode_id     TEXT,
  content        TEXT,
  author_id      TEXT,
  timestamp      INTEGER,
  channel        TEXT,    -- whatsapp/signal/email/telegram
  direction      TEXT,    -- inbound/outbound
  embedding      BLOB

entities
  episode_id     TEXT,
  entity         TEXT,
  type           TEXT,    -- PERSON/ORG/DATE/DECISION...
  source         TEXT     -- "gliner_v1"

episode_links
  from_id        TEXT,
  to_id          TEXT,
  link_type      TEXT,    -- caused_by/follows/reactivates
  strength       REAL,
  created_at     INTEGER
```

Vector search via **sqlite-vec** intégré — pas de FAISS séparé.

### 4.4 États d'un épisode

```python
class EpisodeStatus(Enum):
    RAW                 = "raw"               # message reçu, pas encore enrichi
    PENDING_ENRICHMENT  = "pending_enrichment" # en queue pour LLM
    ENRICHED            = "enriched"           # complet
    FAILED_ENRICHMENT   = "failed_enrichment"  # LLM indispo ou erreur
    STALE               = "stale"              # enrichi mais LLM a changé → à retraiter
    DORMANT             = "dormant"            # decay Ebbinghaus sous seuil
    ARCHIVED            = "archived"           # inactif depuis N jours
```

**Règle** : `FAILED_ENRICHMENT` → retraitable. L'épisode existe avec son contenu brut,
même si le LLM était indisponible. La mémoire ne dépend pas du LLM pour survivre.

### 4.5 Idempotence du slow path

Chaque champ enrichi est versionné par provenance :

```python
# Pas ça — écrasement aveugle :
episode.summary = llm_response

# Ça — versionné et traçable :
episode.summary_v1         = llm_response
episode.summary_model      = "claude-opus-4-5"
episode.summary_updated_at = now()
```

Si le LLM change ou si le job tourne deux fois, l'historique reste cohérent.
Les champs `_v1` deviennent `_v2` lors d'un réenrichissement explicite.

### 4.6 Ranking mémoire avant injection agent

```python
score_final = (
    0.30 * similarité_sémantique(query, episode.embedding)  # pertinence
  + 0.25 * récence_normalisée(episode.end_time)             # fraîcheur
  + 0.25 * episode.axis_salience                            # mémorabilité DeBERTa
  + 0.15 * episode.axis_resolution                          # décision fermée = utile
  + 0.05 * log1p(episode.access_count)                      # fréquence d'usage
)

# Filtre :
# - top 5 épisodes max
# - status in [ENRICHED, DORMANT]  (pas RAW ni FAILED)
# - score_final > seuil_min
```

### 4.7 Memory Context Pack (format d'injection)

Ce que le LLM agent reçoit — jamais les épisodes bruts :

```
Relevant memory:
1. Budget Q2 avec Marc — il y a 2 jours
   Décision : revue budget reportée au lundi
   En attente : envoyer tableur révisé
   [résolution: 0.87 | saillance: 0.72]

2. Voyage avec Sophie — il y a 1 semaine
   Contrainte : renouvellement passeport en cours
   [résolution: 0.34 | saillance: 0.65]
```

Format stable, compact, structuré — le LLM ne se noie pas dans le contexte.

### 4.8 Plugin OpenClaw

```typescript
// extensions/ivias/index.ts

const IVIAS_URL = process.env.IVIAS_URL ?? "http://localhost:8765" // vanilla
                                        // "https://api.ivias.ai"  // cloud

// Capture chaque message → fast path Ivias
on('message_received', async (msg) => {
  await fetch(`${IVIAS_URL}/ingest`, {
    method: 'POST',
    body: JSON.stringify({
      message_id:      msg.id,
      user_id:         ctx.userId,
      content:         msg.text,
      channel:         msg.channel,
      author_id:       msg.author,
      timestamp:       msg.timestamp,
      conversation_id: msg.threadId,
      direction:       'inbound',
    })
  })
})

// Avant chaque réponse → injecte les épisodes pertinents
on('before_agent_start', async (ctx) => {
  const pack = await fetch(`${IVIAS_URL}/memory/search`, {
    method: 'POST',
    body: JSON.stringify({ query: ctx.message, user_id: ctx.userId, top_k: 5 })
  })
  if (pack.episodes.length > 0) {
    ctx.injectContext(pack.formatted)  // Memory Context Pack
  }
})
```

### 4.9 API Ivias (endpoints)

```
POST /ingest                  ← reçoit les messages normalisés d'OpenClaw
POST /memory/search           ← retourne le Memory Context Pack (top K épisodes)
GET  /episodes                ← liste avec filtres (status, date, channel)
GET  /episodes/{id}           ← détail complet d'un épisode
POST /episodes/{id}/enrich    ← force réenrichissement (STALE → PENDING)
POST /memory/export           ← export complet memory.db (portabilité user)
GET  /health                  ← statut pipeline + jobs en attente
GET  /twin/status             ← état du jumeau numérique (Phase D)
POST /twin/connect            ← connexion réseau de jumeaux (Phase D)
```

### 4.10 Gestion des messages rapprochés (conflits)

Si 3 messages arrivent en quelques centaines de ms :

```python
# Transaction SQLite atomique
with db.transaction():
    # Lock logique par conversation_id
    episode = get_or_lock_current_episode(conversation_id)
    message = insert_message(content, timestamp, ...)
    episode = update_episode_append(episode, message)
    # Commit atomique — pas de race condition
```

Le job queue est séquentiel par `conversation_id` — pas de slow path concurrent
sur le même épisode.

---

## 5. Les deux modes produit

### Mode Vanilla — "Mémoire personnelle locale"

```
📱 Téléphone
        ↓ WiFi / Tailscale
💻 Laptop user
  OpenClaw Gateway + Plugin Ivias
  Ivias Local Engine (FastAPI Python)
  memory.db LOCAL
  100% privé — Ivias ne voit rien
  0€ cloud
```

**Promesse utilisateur** :
> *"Ta mémoire reste chez toi. Personne n'y touche. Même pas nous."*

**Modèle économique** : plugin open source gratuit.
Sert de laboratoire, d'argument de confiance, et d'acquisition.

**Stack utilisateur** :
- OpenClaw installé (Node 22+)
- Plugin Ivias installé
- Python 3.11+ + modèles téléchargés une fois (~600MB : mE5 + TCN + DeBERTa)

---

### Mode Cloud — "Réseau de jumeaux"

```
📱 Téléphone
        ↓ Internet
☁️  Serveurs Ivias
  Pipeline managed (mE5 + TCN + DeBERTa)
  memory_{user}.db hébergée
  Decision Engine proactif
  Jumeau A ←→ Jumeau B ←→ Jumeau C
```

**Promesse utilisateur** :
> *"Ton jumeau travaille pour toi même quand tu dors."*

**Modèle économique** :

| Tier | Prix | Ce que tu as |
|------|------|--------------|
| Vanilla | Gratuit | Local, 100% privé |
| Cloud Personnel | ~9€/mois | Mémoire hébergée, multi-device, toujours en ligne |
| Réseau Jumeaux | ~19€/mois | Jumeau connecté, connexions proactives, Decision Engine |

**Coût infra** : ~1-2€/user/mois → **7-17€ de marge nette**.

**Passage vanilla → cloud** : changer une seule variable d'environnement dans le plugin.

---

## 6. Roadmap

### Phase A — Pipeline (en cours)
- [x] mE5 embeddings (11 740 messages)
- [x] AttachScore composite V4 B+C (ARI 0.90, gap 0.035)
- [x] DeBERTa axis_scores (résolution / saillance / temporalité)
- [x] GLiNER entités
- [ ] Schéma SQLite + sqlite-vec (section 4.3)
- [ ] États EpisodeStatus (section 4.4)
- [ ] Cron decay Ebbinghaus

### Phase B — Vanilla
- [ ] Ivias Local Engine (FastAPI Python)
- [ ] Plugin OpenClaw v0 (hooks message_received + before_agent_start)
- [ ] Fast path : AttachScore → décision NEW/APPEND/REACTIVATE
- [ ] Slow path : job queue LLM enrichissement
- [ ] Memory Context Pack + ranking (section 4.6-4.7)
- [ ] 5-10 beta users en local

### Phase C — Cloud Personnel
- [ ] API Ivias cloud (Fly.io ou Render)
- [ ] Auth multi-tenant
- [ ] memory_{user}.db isolée par user
- [ ] Multi-device (laptop éteint → mémoire toujours en ligne)
- [ ] Dashboard web minimal

### Phase D — Réseau de jumeaux
- [ ] Jumeau v1 : LLM + mémoire = représentation personnelle
- [ ] Protocole agent-to-agent (A2A / MCP)
- [ ] Réseau fermé (10-50 jumeaux) pour valider les connexions
- [ ] Decision Engine proactif complet
- [ ] Ivias = OS cognitif personnel

---

## 7. Décisions architecturales clés

### Pourquoi OpenClaw et pas from scratch
- Licence MIT — 0 contrainte commerciale
- 21+ canaux déjà intégrés — des mois de travail évités
- Apps mobiles iOS + Android déjà codées
- Système de plugins clair — 2 hooks suffisent pour démarrer
- Focus Ivias : la mémoire épisodique, pas l'infra canal

### Pourquoi AttachScore composite et pas TCN seul
- TCN = signal principal, pas autorité unique
- `threshold_eff = attach_threshold + boundary_k × (1 - P_boundary)` calibre dynamiquement
- Sem + Ent + Temp + Goal dans l'AttachScore couvrent les cas que TCN manque
- V4 B+C validé : ARI 0.90, gap tune-test 0.035 (propre)
- **Ne pas remplacer par des heuristiques simples en refactorant**

### Pourquoi la réactivation est le différenciateur
- Aucun système existant (DialSeg, BERTSeg, LLM sliding window) ne distingue
  "nouvel épisode" de "réactivation d'épisode dormant"
- C'est ce qui rend la mémoire vivante — pas une liste plate
- En prod : `REACTIVATE` incrémente `access_count`, réinitialise le decay,
  crée un `episode_link` de type `reactivates`

### Pourquoi SQLite et pas Postgres
- Local-first : un fichier = toute la mémoire
- Portable : l'utilisateur peut exporter et posséder sa mémoire
- sqlite-vec : vector search sans infrastructure externe
- Compatible cloud : Turso (SQLite distribué) si besoin multi-device avancé

### Pourquoi BYOK pour le LLM
- 0€ coût LLM pour Ivias
- L'utilisateur choisit son provider (Claude, GPT, Gemini, local)
- Privacy : les données vont chez le provider choisi par l'user, pas chez Ivias
- La mémoire brute survit même si le LLM est indisponible (status FAILED_ENRICHMENT)

### Pourquoi vanilla d'abord
- Argument de confiance fort : "ta mémoire reste chez toi"
- Laboratoire de validation sans infrastructure cloud
- Acquisition : open source → communauté → conversion cloud
- Modèle Obsidian prouvé : gratuit local → payant sync cloud

---

## 8. Risques techniques à anticiper

| Risque | Mitigation |
|--------|-----------|
| LLM lent / indisponible | Fast path indépendant, status FAILED_ENRICHMENT retraitable |
| Messages rapprochés (race condition) | Transaction SQLite + lock par conversation_id |
| Sur-interprétation LLM | Champs probabilistes, versionnés, corrigeables |
| Explosion du contexte agent | Top 5 max, Memory Context Pack compact, seuil score_min |
| Slow path qui tourne deux fois | Idempotence : champs versionnés, merge contrôlé |
| Modèles trop lourds en vanilla | mE5-small ONNX (~117MB) en fallback si laptop limité |

---

## 9. Ce que personne d'autre ne fait

```
Outil A : capture tout → pas de structure
Outil B : structure bien → effort manuel
Outil C : agent conversationnel → pas de mémoire persistante
Outil D : RAG sur notes → tu dois écrire les notes

Ivias : capture passive + AttachScore composite + lifecycle
      + decay Ebbinghaus + réactivation + Decision Engine proactif
```

> *"Tu arrêtes de perdre les choses qui comptaient."*
