# data/tags.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_card_tags() -> pd.DataFrame:
    """Load card tags exported from Scryfall Tagger."""
    path = Path(__file__).with_name("card_tags.csv")
    if not path.exists():
        return pd.DataFrame(columns=["name", "category"])

    df = pd.read_csv(path)
    if "name" not in df.columns:
        df["name"] = ""
    if "category" not in df.columns:
        df["category"] = ""

    df["name"] = df["name"].astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str)
    return df
