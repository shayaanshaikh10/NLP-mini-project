"""
utils.py - small helpers for logging and timing.
"""

import json
import time
from datetime import datetime

from src import config


def log_step(message):
    """Print a message and append it, with a timestamp, to BUILD_LOG.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"- **{timestamp}**: {message}\n"
    print(message)
    with open(config.BUILD_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(line)


class Timer:
    """Context manager that measures how many seconds a block of code takes.

    Example:
        with Timer() as t:
            train_model()
        print(t.seconds)
    """

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.seconds = time.perf_counter() - self.start


def save_json(data, path):
    """Save a Python dictionary as a pretty JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def load_json(path):
    """Load a JSON file into a dictionary (returns {} if the file is missing)."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)
