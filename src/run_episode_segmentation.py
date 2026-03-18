import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from parsers.whatsapp_parser import parse_whatsapp_chat
from embedding_engine import embed_texts
from episode_algorithm import EpisodeSegmenter

# charger données
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chat.txt")
artifacts = parse_whatsapp_chat(DATA_PATH)

# embeddings
embeddings = embed_texts([a.content for a in artifacts])

# segmentation
segmenter = EpisodeSegmenter(
    time_threshold_minutes=120,
    semantic_threshold=0.10
)

episodes = segmenter.segment(artifacts, embeddings)

# affichage
for i, episode in enumerate(episodes):

    print("\n====================")
    print("Episode", i + 1)

    for index in episode:
        a = artifacts[index]
        print(a.timestamp, "-", a.author, ":", a.content)
