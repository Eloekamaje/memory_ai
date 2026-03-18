import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Artifact

def parse_experiment_csv(path):

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    artifacts = []

    for _, row in df.iterrows():

        entities = []
        if isinstance(row.get("entities"), str):
            entities = [e.strip() for e in row["entities"].split(",")]

        artifacts.append(
            Artifact(
                id=str(row["artifact_id"]),
                timestamp=row["timestamp"],
                author=row.get("author", "unknown"),
                content=row["content"],
                entities=entities,
                source=row.get("source", "csv"),
                artifact_type=row.get("artifact_type", "message"),
                importance=float(row.get("importance", 0.5)),
                true_episode_id=row.get("true_episode_id"),
            )
        )

    return artifacts
