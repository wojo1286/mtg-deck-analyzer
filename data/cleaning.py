# data/cleaning.py
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def clean_and_prepare_data(df: pd.DataFrame, categories_df: pd.DataFrame | None = None):
    if df is None or df.empty:
        return (pd.DataFrame(), False, 0, pd.DataFrame())

    work = df.copy()

    # normalize price -> price_clean
    if "price_clean" not in work.columns:
        if "price" in work.columns:
            work["price_clean"] = pd.to_numeric(
                work["price"].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )
        else:
            work["price_clean"] = np.nan

    # cmc numeric
    work["cmc"] = pd.to_numeric(work.get("cmc", np.nan), errors="coerce")

    # type fallback
    work["type"] = work.get("type", "Unknown").fillna("Unknown")

    functional = False
    if categories_df is not None and not categories_df.empty and "name" in categories_df.columns:
        merged = pd.merge(work, categories_df, on="name", how="left")
        merged["category"] = merged["category"].fillna("Uncategorized")
        work = merged
        functional = True

    num_decks = work["deck_id"].nunique() if "deck_id" in work.columns else 0

    pop_all = (
        work.groupby("name", as_index=False)
        .agg(count=("deck_id", "nunique"))
        .sort_values("count", ascending=False)
    )
    if num_decks:
        pop_all["inclusion_rate"] = pop_all["count"] / num_decks * 100
    else:
        pop_all["inclusion_rate"] = 0.0

    return work, functional, num_decks, pop_all


@st.cache_data(show_spinner=False)
def calculate_average_stats(
    df: pd.DataFrame, num_decks: int, active_func_categories: list[str] | None = None
):
    if df is None or df.empty or not num_decks:
        return {}

    stats = {}
    d = df.copy()

    # Primary type mapping
    primary_types = [
        "Creature",
        "Instant",
        "Sorcery",
        "Artifact",
        "Enchantment",
        "Land",
        "Planeswalker",
        "Battle",
    ]
    d["primary_type"] = (
        d["type"].astype(str).apply(lambda x: next((t for t in primary_types if t in x), "Other"))
    )

    # CMC for spells
    non_land = d[d["primary_type"] != "Land"]
    cmc_valid = pd.to_numeric(non_land["cmc"], errors="coerce").dropna()
    stats["avg_cmc_non_land"] = cmc_valid.mean() if not cmc_valid.empty else 0
    stats["median_cmc_non_land"] = cmc_valid.median() if not cmc_valid.empty else 0

    # Type distribution per deck
    type_counts = d.groupby("deck_id")["primary_type"].value_counts().unstack(fill_value=0)
    avg_type_counts = type_counts.mean()
    stats["avg_type_counts"] = avg_type_counts.to_dict()

    # Land breakdown (infer basics roughly)
    deck_known = d.groupby("deck_id").size().reset_index(name="known_cards")
    deck_known["inferred_basics"] = (100 - deck_known["known_cards"]).clip(lower=0)
    stats["avg_basic_lands"] = float(deck_known["inferred_basics"].mean())

    land_df = d[d["primary_type"] == "Land"]
    basic_names = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
    non_basic_land_df = land_df[~land_df["name"].isin(basic_names)]
    stats["avg_non_basic_lands"] = float(
        non_basic_land_df.groupby("deck_id").size().mean() if not non_basic_land_df.empty else 0
    )
    stats["avg_total_lands"] = stats["avg_basic_lands"] + stats["avg_non_basic_lands"]

    # Functional category averages (if present)
    stats["avg_functional_counts"] = {}
    if "category" in d.columns and active_func_categories:
        d["category_list"] = d["category"].fillna("").astype(str).str.split("|")
        cats = d.explode("category_list")
        cats = cats[(cats["category_list"] != "") & (cats["category_list"] != "Uncategorized")]
        per_deck = cats.groupby("deck_id")["category_list"].value_counts().unstack(fill_value=0)
        avg_cat = per_deck.mean()
        stats["avg_functional_counts"] = {
            c: float(avg_cat.get(c, 0)) for c in active_func_categories
        }

    # Deck prices
    if "price_clean" in d.columns:
        deck_total_prices = d.groupby("deck_id")["price_clean"].sum()
        stats["avg_deck_price"] = float(
            deck_total_prices.mean() if not deck_total_prices.empty else 0
        )
        stats["median_deck_price"] = float(
            deck_total_prices.median() if not deck_total_prices.empty else 0
        )
        stats["min_deck_price"] = float(
            deck_total_prices.min() if not deck_total_prices.empty else 0
        )
        stats["max_deck_price"] = float(
            deck_total_prices.max() if not deck_total_prices.empty else 0
        )

    # CMC distribution
    cmc_dist = cmc_valid.value_counts().sort_index()
    stats["cmc_distribution"] = {int(k): int(v) for k, v in cmc_dist.items()}
    return stats
