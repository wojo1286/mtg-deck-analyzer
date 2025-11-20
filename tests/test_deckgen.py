import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.deckgen import generate_average_deck, summarize_deck


def _sample_df():
    return pd.DataFrame(
        {
            "deck_id": ["d1", "d1", "d1", "d2", "d2", "d2", "d2"],
            "name": [
                "Sol Ring",
                "Lightning Bolt",
                "Command Tower",
                "Rhystic Study",
                "Arcane Signet",
                "Island",
                "Command Tower",
            ],
            "type": [
                "Artifact",
                "Instant",
                "Land",
                "Enchantment",
                "Artifact",
                "Basic Land — Island",
                "Land",
            ],
            "cmc": [1, 1, 0, 3, 2, 0, 0],
            "price_clean": [1.0, 0.5, 0.2, 5.0, 1.2, 0.05, 0.2],
            "category": [
                "Ramp|Mana Rock",
                "Removal",
                "Fixing",
                "Card Advantage",
                "Ramp",
                "",
                "Fixing",
            ],
        }
    )


def test_generate_average_deck_respects_land_targets_and_size():
    df = _sample_df()
    deck = generate_average_deck(df, total_size=10, commander_colors=["U", "R"], land_target=4)

    assert len(deck) == 10

    summary = summarize_deck(deck, df)
    land_total = summary["counts_by_type"]["count"].sum()
    assert land_total == len(deck)

    assert summary["basics"] == 3  # Islands + Mountains split
    assert summary["non_basics"] == 1


def test_summarize_deck_counts_match_deck_list():
    df = _sample_df()
    deck = [
        "Sol Ring",
        "Lightning Bolt",
        "Island",
        "Command Tower",
        "Island",
    ]
    summary = summarize_deck(deck, df)

    assert summary["counts_by_type"]["count"].sum() == len(deck)

    land_rows = summary["counts_by_type"][summary["counts_by_type"]["primary_type"] == "Land"]
    assert land_rows["count"].sum() == summary["basics"] + summary["non_basics"]

    categories = set(summary["functions_covered"]["category"].tolist())
    assert {"Ramp", "Mana Rock", "Removal"}.issubset(categories)
