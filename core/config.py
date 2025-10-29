from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "assets" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
DEFAULT_FUNCTIONAL_CATEGORIES = CONFIG["default_functional_categories"]
TYPE_KEYWORDS = CONFIG["type_keywords"]
