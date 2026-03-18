from typing import List
import numpy as np

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from models import Artifact, Episode


# ---------------------------------------------------------------------------
# Vecteurs de prédiction / vérité terrain
# ---------------------------------------------------------------------------

def build_prediction_vector(artifacts: List[Artifact], episodes: List[Episode]):
    """
    Convertit les épisodes prédits en vecteur aligné avec les artefacts.
    """

    y_pred = [None] * len(artifacts)

    for episode in episodes:
        for idx in episode.artifact_indices:
            y_pred[idx] = episode.id

    return y_pred


def build_true_vector(artifacts: List[Artifact]):
    """
    Extrait les labels vérité terrain.
    """

    y_true = []

    for artifact in artifacts:
        y_true.append(artifact.true_episode_id)

    return y_true


# ---------------------------------------------------------------------------
# Métriques de clustering
# ---------------------------------------------------------------------------

def compute_clustering_scores(y_true, y_pred):

    ari = adjusted_rand_score(y_true, y_pred)

    nmi = normalized_mutual_info_score(y_true, y_pred)

    return {
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
    }


# ---------------------------------------------------------------------------
# Fragmentation
# ---------------------------------------------------------------------------

def compute_fragmentation(y_true, y_pred):

    true_episodes = len(set(y_true))
    predicted_episodes = len(set(y_pred))

    if true_episodes == 0:
        return {}

    fragmentation_ratio = predicted_episodes / true_episodes

    return {
        "true_episode_count": true_episodes,
        "predicted_episode_count": predicted_episodes,
        "fragmentation_ratio": fragmentation_ratio,
    }


# ---------------------------------------------------------------------------
# Cohérence intra-épisode
# ---------------------------------------------------------------------------

def compute_episode_coherence(episodes: List[Episode], embeddings: np.ndarray):
    """
    Cohérence moyenne des épisodes :
        1/|A_e| * Σ sim(a_i, centroïde_e)

    Retourne la cohérence par épisode et la moyenne globale.
    """

    coherences = []

    for episode in episodes:
        if len(episode.artifact_indices) < 2:
            coherences.append(1.0)
            continue

        ep_embeddings = embeddings[episode.artifact_indices]
        centroid = np.mean(ep_embeddings, axis=0).reshape(1, -1)
        sims = cosine_similarity(ep_embeddings, centroid).flatten()
        coherences.append(float(np.mean(sims)))

    return {
        "episode_coherences": coherences,
        "mean_coherence": float(np.mean(coherences)) if coherences else 0.0,
    }


# ---------------------------------------------------------------------------
# Pipeline d'évaluation complet
# ---------------------------------------------------------------------------

def evaluate_segmentation(artifacts: List[Artifact],
                          episodes: List[Episode],
                          embeddings: np.ndarray):
    """
    Évaluation complète : clustering + fragmentation + cohérence.
    """

    results = {}

    y_true = build_true_vector(artifacts)
    y_pred = build_prediction_vector(artifacts, episodes)

    # métriques de clustering (seulement si vérité terrain disponible)
    has_ground_truth = any(label is not None for label in y_true)

    if has_ground_truth:
        results.update(compute_clustering_scores(y_true, y_pred))
        results.update(compute_fragmentation(y_true, y_pred))

    # cohérence (toujours calculable)
    coherence = compute_episode_coherence(episodes, embeddings)
    results["mean_coherence"] = coherence["mean_coherence"]
    results["episode_count"] = len(episodes)
    results["artifact_count"] = len(artifacts)

    return results