"""
Cleaning and aggregation utilities for parsed deck data.
Handles normalization, deduplication, and basic statistics.
"""

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def clean_and_prepare_data(cards: list[dict]) -> pd.DataFrame:
    """Normalize and clean parsed card data before analysis."""
    if not cards:
        return pd.DataFrame(columns=["deck_id", "deck_source", "cmc", "name", "type", "price"])

    df = pd.DataFrame(cards)

    # Normalize columns
    df["name"] = df["name"].str.strip().str.title()
    df["type"] = df["type"].fillna("Unknown")
    df["cmc"] = pd.to_numeric(df["cmc"], errors="coerce").fillna(0)
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .astype(float)
        .fillna(0.0)
    )

    # Drop duplicates within deck_id scope
    df = df.drop_duplicates(subset=["deck_id", "name"])

    # Reorder columns
    return df[["deck_id", "deck_source", "name", "type", "cmc", "price"]]


@st.cache_data(show_spinner=False)
def calculate_average_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for a cleaned deck DataFrame."""
    if df.empty:
        return {
            "total_cards": 0,
            "avg_cmc": 0,
            "avg_price": 0,
            "unique_types": [],
        }

    summary = {
        "total_cards": len(df),
        "avg_cmc": round(df["cmc"].mean(), 2),
        "avg_price": round(df["price"].mean(), 2),
        "unique_types": sorted(df["type"].unique().tolist()),
    }
    return summary
