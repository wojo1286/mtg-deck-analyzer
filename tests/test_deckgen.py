from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.deckgen import generate_average_deck, summarize_deck


def _candidate_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "deck_id": ["d1"] * 8 + ["d2"] * 4,
            "name": [
                "Sol Ring",
                "Arcane Signet",
                "Cultivate",
                "Kodama's Reach",
                "Command Tower",
                "Breeding Pool",
                "Forest",
                "Island",
                "Beast Within",
                "Growth Spiral",
                "Reliquary Tower",
                "Mystic Remora",
            ],
            "type": [
                "Artifact",
                "Artifact",
                "Sorcery",
                "Sorcery",
                "Land",
                "Land",
                "Basic Land — Forest",
                "Basic Land — Island",
                "Instant",
                "Instant",
                "Land",
                "Enchantment",
            ],
            "cmc": [1, 2, 3, 3, 0, 0, 0, 0, 3, 2, 0, 1],
            "price_clean": [1.0, 1.2, 0.4, 0.35, 0.25, 6.5, 0.05, 0.05, 0.15, 0.25, 0.5, 2.0],
            "category": [
                "Ramp|Mana Rock",
                "Ramp|Mana Rock",
                "Ramp|Fixing",
                "Ramp|Fixing",
                "Fixing",
                "Fixing",
                "",
                "",
                "Removal",
                "Card Advantage|Ramp",
                "Card Advantage",
                "Card Advantage",
            ],
        }
    )


def test_generate_average_deck_targets_land_count_and_size():
    df = _candidate_pool()
    deck = generate_average_deck(
        df, total_size=12, commander_colors=["U", "G"], target_land_count=5
    )

    assert len(deck) == 12

    summary = summarize_deck(deck, df)
    land_count = (
        summary["counts_by_type"].set_index("type").loc["Land", "count"]
    )
    assert land_count == 5
    assert summary["basics"] + summary["non_basics"] == land_count


def test_summarize_deck_land_breakdown_matches_cards():
    df = pd.DataFrame(
        {
            "deck_id": ["d1"] * 5,
            "name": ["Plains", "Island", "Command Tower", "Sol Ring", "Arcane Signet"],
            "type": [
                "Basic Land — Plains",
                "Basic Land — Island",
                "Land",
                "Artifact",
                "Artifact",
            ],
            "cmc": [0, 0, 0, 1, 2],
            "price_clean": [0.05, 0.05, 0.5, 1.0, 1.2],
            "category": ["", "", "Fixing", "Ramp|Mana Rock", "Ramp"],
        }
    )

    deck = ["Plains", "Island", "Command Tower", "Sol Ring", "Arcane Signet"]
    summary = summarize_deck(deck, df)

    assert summary["counts_by_type"]["count"].sum() == len(deck)
    land_count = summary["counts_by_type"].set_index("type").loc["Land", "count"]
    assert land_count == summary["basics"] + summary["non_basics"]
    assert summary["basics"] == 2
    assert summary["non_basics"] == 1


def test_functions_covered_counts_all_tags():
    df = pd.DataFrame(
        {
            "deck_id": ["d3"] * 3,
            "name": ["Cultivate", "Growth Spiral", "Beast Within"],
            "type": ["Sorcery", "Instant", "Instant"],
            "cmc": [3, 2, 3],
            "price_clean": [0.4, 0.2, 0.15],
            "category": ["Ramp|Card Advantage", "Ramp|Card Advantage", "Removal"],
        }
    )

    deck = ["Cultivate", "Growth Spiral", "Beast Within"]
    summary = summarize_deck(deck, df)

    func_counts = summary["functions_covered"].set_index("category")["count"].to_dict()
    assert func_counts["Ramp"] == 2
    assert func_counts["Card Advantage"] == 2
    assert func_counts["Removal"] == 1
    assert summary["counts_by_type"]["count"].sum() == len(deck)


def test_summarize_deck_deduplicates_metadata_rows():
    df = pd.DataFrame(
        {
            "deck_id": ["d1"] * 4,
            "name": ["Command Tower", "Command Tower", "Swamp", "Swamp"],
            "type": ["Land", "Land", "Basic Land — Swamp", "Basic Land — Swamp"],
            "cmc": [0, 0, 0, 0],
            "price_clean": [0.5, 1.0, 0.05, 0.05],
            "category": ["Fixing", "", "", ""],
        }
    )

    deck = ["Command Tower", "Swamp", "Swamp"]
    summary = summarize_deck(deck, df)

    assert summary["counts_by_type"]["count"].sum() == len(deck)
    land_count = summary["counts_by_type"].set_index("type").loc["Land", "count"]
    assert land_count == summary["basics"] + summary["non_basics"]
    assert summary["basics"] == 2
    assert summary["non_basics"] == 1
