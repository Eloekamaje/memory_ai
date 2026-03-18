"""
dataset_generator.py — Générateur de dataset synthétique pour expérimentation.

Produit un CSV de ~200 artefacts répartis en épisodes vrais, avec :
  - 5 thèmes (GEDDVIT, SCHOOL, HEALTH, HOUSING, TRAVEL)
  - 10+ entités récurrentes
  - Types variés (message, email, note, document, reunion, decision)
  - Épisodes entrelacés temporellement
  - Réactivations (retour sur un thème après pause)
  - Bruit contrôlé (messages ambigus)
"""

import csv
import os
import random
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Configuration des thèmes et templates
# ---------------------------------------------------------------------------

THEMES = {
    "GEDDVIT": {
        "entities_pool": ["Jean", "Geddvit", "budget", "donor", "finance"],
        "templates": [
            ("message", "Meeting with {e1} about {e2} budget"),
            ("email", "Email to {e1} regarding {e2} financial report"),
            ("document", "Prepare budget document for {e2}"),
            ("email", "Donor {e1} requested budget corrections for {e2}"),
            ("reunion", "Weekly meeting on {e2} project progress"),
            ("decision", "Decision to allocate funds for {e2}"),
            ("note", "Note about {e2} budget adjustments discussed with {e1}"),
            ("message", "{e1} confirmed budget approval for {e2}"),
            ("email", "Send financial summary to {e1} about {e2}"),
            ("document", "Final budget report for {e2} project"),
            ("message", "Follow up with {e1} on {e2} expenses"),
            ("email", "{e1} asked about {e2} remaining funds"),
        ],
    },
    "SCHOOL": {
        "entities_pool": ["Tatiana", "school", "certificate", "inscription", "director"],
        "templates": [
            ("message", "School requested documents for inscription"),
            ("email", "Send birth certificate to {e1}"),
            ("document", "Prepare inscription file for {e1}"),
            ("message", "{e1} confirmed reception of documents"),
            ("reunion", "Meeting with {e1} director about enrollment"),
            ("email", "Request for additional {e1} documents"),
            ("note", "Note about {e1} enrollment deadline"),
            ("decision", "Decision to enroll at {e1}"),
            ("message", "{e1} admission confirmed"),
            ("email", "Payment of {e1} tuition fees"),
            ("message", "{e1} schedule received for next semester"),
            ("document", "Student registration form for {e1}"),
        ],
    },
    "HEALTH": {
        "entities_pool": ["Dr Martin", "clinic", "prescription", "appointment", "insurance"],
        "templates": [
            ("message", "Book appointment with {e1}"),
            ("email", "Confirmation of appointment at {e1}"),
            ("note", "Prescription notes from {e1}"),
            ("document", "Insurance claim for {e1} visit"),
            ("message", "{e1} called about test results"),
            ("reunion", "Follow-up consultation with {e1}"),
            ("email", "Send medical records to {e1}"),
            ("decision", "Decision to change treatment per {e1} recommendation"),
            ("message", "Pharmacy confirmed {e1} prescription ready"),
            ("email", "Insurance approved coverage for {e1} procedure"),
            ("note", "Health notes after visit to {e1}"),
            ("document", "Medical report from {e1}"),
        ],
    },
    "HOUSING": {
        "entities_pool": ["landlord", "apartment", "lease", "moving", "agent"],
        "templates": [
            ("message", "Contact {e1} about new apartment"),
            ("email", "Lease agreement sent by {e1}"),
            ("document", "Sign lease for new {e1}"),
            ("reunion", "Visit {e1} with rental agent"),
            ("decision", "Decision to move to new {e1}"),
            ("note", "Moving checklist for {e1} relocation"),
            ("message", "{e1} confirmed move-in date"),
            ("email", "Request deposit refund from {e1}"),
            ("document", "Utility transfer forms for {e1}"),
            ("message", "Keys received from {e1}"),
            ("email", "Address change notification to {e1}"),
            ("note", "Furniture delivery schedule for new {e1}"),
        ],
    },
    "TRAVEL": {
        "entities_pool": ["airline", "hotel", "passport", "visa", "Cameroun"],
        "templates": [
            ("message", "Book flight with {e1} to {e2}"),
            ("email", "Flight confirmation from {e1}"),
            ("document", "Visa application for {e1}"),
            ("reunion", "Planning trip to {e1} with family"),
            ("note", "Travel itinerary for {e1} trip"),
            ("decision", "Decision to travel to {e1} in summer"),
            ("email", "Hotel reservation confirmation in {e1}"),
            ("message", "Passport renewal for {e1} travel"),
            ("document", "Travel insurance for {e1} trip"),
            ("email", "Cancel hotel in {e1} due to schedule change"),
            ("message", "Pack list for {e1} trip"),
            ("note", "Budget estimate for {e1} vacation"),
        ],
    },
}

# Entités partagées entre thèmes (source de bruit)
SHARED_ENTITIES = ["document", "budget", "email"]

# Messages ambigus (bruit contrôlé)
NOISE_TEMPLATES = [
    ("message", "Quick question about the document"),
    ("message", "Thanks for the update"),
    ("email", "Please review the attached file"),
    ("note", "Reminder to follow up this week"),
    ("message", "OK sounds good"),
    ("message", "Let me check and get back to you"),
    ("email", "Forwarding the information you requested"),
    ("message", "Can we discuss this tomorrow?"),
]


# ---------------------------------------------------------------------------
# Générateur
# ---------------------------------------------------------------------------

def generate_dataset(
    n_artifacts: int = 200,
    n_days: int = 30,
    start_date: datetime = datetime(2026, 2, 1, 8, 0),
    noise_ratio: float = 0.08,
    seed: int = 42,
) -> list:
    """
    Génère une liste de dicts représentant des artefacts synthétiques.
    """

    random.seed(seed)

    records = []
    artifact_id = 0
    episode_counter = 0

    theme_names = list(THEMES.keys())

    # planifier les épisodes : chaque thème a 3-5 épisodes étalés
    episode_plan = []

    for theme_name in theme_names:
        n_episodes = random.randint(3, 5)
        for ep_idx in range(n_episodes):
            episode_counter += 1
            ep_id = f"E{episode_counter}"

            # jour de début de l'épisode
            day_start = random.randint(0, n_days - 3)
            # durée de l'épisode (1-4 jours)
            ep_duration = random.randint(1, 4)

            # nombre d'artefacts dans cet épisode
            n_ep_artifacts = random.randint(3, 8)

            episode_plan.append({
                "episode_id": ep_id,
                "theme": theme_name,
                "day_start": day_start,
                "duration": ep_duration,
                "n_artifacts": n_ep_artifacts,
            })

    # générer les artefacts pour chaque épisode
    for ep in episode_plan:
        theme = THEMES[ep["theme"]]
        templates = theme["templates"]
        entities_pool = theme["entities_pool"]

        for _ in range(ep["n_artifacts"]):
            artifact_id += 1
            template = random.choice(templates)
            art_type, content_tpl = template

            # remplir les entités
            e1 = random.choice(entities_pool[:3])
            e2 = random.choice(entities_pool[1:])
            content = content_tpl.format(e1=e1, e2=e2)

            # timestamp aléatoire dans la fenêtre de l'épisode
            day_offset = random.randint(0, ep["duration"])
            hour = random.randint(7, 21)
            minute = random.randint(0, 59)

            ts = start_date + timedelta(
                days=ep["day_start"] + day_offset,
                hours=hour - 8,
                minutes=minute
            )

            # entités de cet artefact (2-4 tirées du pool)
            n_ent = random.randint(2, min(4, len(entities_pool)))
            entities = random.sample(entities_pool, n_ent)

            # importance
            importance = round(random.uniform(0.3, 0.9), 2)
            if art_type == "decision":
                importance = round(random.uniform(0.7, 1.0), 2)

            records.append({
                "artifact_id": str(artifact_id),
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "content": content,
                "entities": ",".join(entities),
                "true_episode_id": ep["episode_id"],
                "theme": ep["theme"],
                "artifact_type": art_type,
                "source": random.choice(["whatsapp", "email", "notes", "drive"]),
                "importance": importance,
            })

    # ajouter du bruit
    n_noise = int(n_artifacts * noise_ratio)
    for _ in range(n_noise):
        artifact_id += 1
        art_type, content = random.choice(NOISE_TEMPLATES)

        day = random.randint(0, n_days - 1)
        hour = random.randint(7, 21)
        minute = random.randint(0, 59)

        ts = start_date + timedelta(days=day, hours=hour - 8, minutes=minute)

        # assigner un épisode aléatoire (le bruit est rattaché à un vrai épisode
        # pour simuler un message ambigu, pas un épisode isolé)
        noise_ep = random.choice(episode_plan)

        records.append({
            "artifact_id": str(artifact_id),
            "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "content": content,
            "entities": random.choice(SHARED_ENTITIES),
            "true_episode_id": noise_ep["episode_id"],
            "theme": noise_ep["theme"],
            "artifact_type": art_type,
            "source": "unknown",
            "importance": round(random.uniform(0.1, 0.4), 2),
        })

    # trier par timestamp
    records.sort(key=lambda r: r["timestamp"])

    return records


def save_dataset(records: list, path: str):
    """Sauvegarde les records en CSV."""

    fieldnames = [
        "artifact_id", "timestamp", "content", "entities",
        "true_episode_id", "theme", "artifact_type", "source", "importance",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Dataset saved: {path}")
    print(f"  Artifacts: {len(records)}")
    print(f"  Episodes:  {len(set(r['true_episode_id'] for r in records))}")
    print(f"  Themes:    {len(set(r['theme'] for r in records))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # dataset 200
    records_200 = generate_dataset(n_artifacts=200, seed=42)
    save_dataset(records_200, os.path.join(data_dir, "synthetic_200.csv"))

    # dataset 500
    records_500 = generate_dataset(n_artifacts=500, n_days=60, seed=123)
    save_dataset(records_500, os.path.join(data_dir, "synthetic_500.csv"))
