import json
import os

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "default.json"))

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)
