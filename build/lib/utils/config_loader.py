import json
import os

def load_config(filename="config.json"):
    # Get absolute path to the project root, two levels up from this file
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", filename))

    with open(config_path, "r") as f:
        return json.load(f)
