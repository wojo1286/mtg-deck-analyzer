from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.stats import augment_with_basic_lands


def test_augment_with_basic_lands_adds_missing_cards_even_split():
    df = pd.DataFrame(
        {
            "deck_id": ["d1"] * 90,
            "name": [f"Card {i}" for i in range(90)],
            "type": ["Creature"] * 90,
            "cmc": [3] * 90,
            "price_clean": [1.0] * 90,
            "category": ["Spell"] * 90,
        }
    )

    augmented = augment_with_basic_lands(df, commander_colors=["R", "G"], target_size=100)

    assert len(augmented[augmented["deck_id"] == "d1"]) == 100
    basics = augmented[augmented["type"] == "Land"]
    assert basics.shape[0] == 10
    counts = basics.groupby("name").size().to_dict()
    # Two colors should be split as evenly as possible
    assert counts.get("Mountain", 0) in {5, 6}
    assert counts.get("Forest", 0) in {4, 5}
    assert sum(counts.values()) == 10


def test_augment_with_basic_lands_handles_colorless():
    df = pd.DataFrame(
        {
            "deck_id": ["d2"] * 100,
            "name": [f"Card {i}" for i in range(100)],
            "type": ["Creature"] * 100,
        }
    )

    augmented = augment_with_basic_lands(df, commander_colors=[], target_size=100)
    # Already at target, no new rows should be added
    assert len(augmented[augmented["deck_id"] == "d2"]) == 100
