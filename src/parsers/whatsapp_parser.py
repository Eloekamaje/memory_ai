import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from models import Artifact

pattern = r"(\d{4}-\d{2}-\d{2}), (\d{2}) h (\d{2}) - ([^:]+): (.*)"

def parse_whatsapp_chat(file_path):
    artifacts = []

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()

            match = re.match(pattern, line)
            if match:
                date = match.group(1)
                hour = match.group(2)
                minute = match.group(3)
                author = match.group(4)
                message = match.group(5)

                timestamp = datetime.strptime(
                    f"{date} {hour}:{minute}", "%Y-%m-%d %H:%M"
                )

                artifacts.append(
                    Artifact(
                        id=str(idx),
                        timestamp=timestamp,
                        author=author,
                        content=message,
                    )
                )

    artifacts.sort(key=lambda a: a.timestamp)

    return artifacts
