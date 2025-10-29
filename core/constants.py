# constants.py
# Loads config constants from constants.yaml and exposes them for app-wide usage

import yaml
import os

# Get absolute path to constants.yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "constants.yaml")

# Load YAML safely
def load_constants():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Load once at module level
CONSTANTS = load_constants()

# Expose specific constants
DEFAULT_FUNCTIONAL_CATEGORIES = CONSTANTS.get("DEFAULT_FUNCTIONAL_CATEGORIES", {})
TYPE_KEYWORDS = CONSTANTS.get("TYPE_KEYWORDS", {})
COLOR_MAP = CONSTANTS.get("COLOR_MAP", {})
MANA_SYMBOLS = CONSTANTS.get("MANA_SYMBOLS", [])
SYNERGY_THRESHOLD = CONSTANTS.get("SYNERGY_THRESHOLD", 0.0)

# Helper to expose all constants
__all__ = [
    "DEFAULT_FUNCTIONAL_CATEGORIES",
    "TYPE_KEYWORDS",
    "COLOR_MAP",
    "MANA_SYMBOLS",
    "SYNERGY_THRESHOLD",
    "load_constants",
]
