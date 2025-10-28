# analysis/stats.py
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Utilities ----------

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    if df is None or df.empty:
        return False
    missing = [c for c in cols if c not in df.columns]
    return len(missing) == 0

# ---------- Budget filter ----------

@st.cache_data(show_spinner=False)
def budget_filtered(df: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    """Keep rows with price <= cap OR unknown price. None means no filter."""
    if df is None or df.empty or cap is None:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if "price" not in df.columns:
        return df.copy()
    out = df.copy()
    out["price_clean"] = (
        pd.to_numeric(
            out["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
    )
    return out[(out["price_clean"].isna()) | (out["price_clean"] <= cap)]

# ---------- Popularity / inclusion ----------

@st.cache_data(show_spinner=False)
def inclusion_table(
    df: pd.DataFrame,
    *,
    exclude_types: list[str] | None = None,
    exclude_top_n: int | None = None,
) -> pd.DataFrame:
    """
    Returns one row per card:
      name, count (# of distinct decks), inclusion_rate (% of decks),
      avg_price, avg_cmc, type
    """
    if not _ensure_cols(df, ["deck_id", "name"]):
        return pd.DataFrame(columns=["name","count","inclusion_rate","avg_price","avg_cmc","type"])

    work = df.copy()

    # Optional type filtering
    if exclude_types and "type" in work.columns:
        work = work[~work["type"].isin(exclude_types)]

    # --- Ensure numeric columns for safe aggregation ---
    # price_num: prefer price_clean if present; otherwise coerce from price
    if "price_clean" in work.columns:
        work["price_num"] = pd.to_numeric(work["price_clean"], errors="coerce")
    elif "price" in work.columns:
        work["price_num"] = pd.to_numeric(
            work["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
    else:
        work["price_num"] = np.nan

    # cmc_num: coerce to numeric
    if "cmc" in work.columns:
        work["cmc_num"] = pd.to_numeric(work["cmc"], errors="coerce")
    else:
        work["cmc_num"] = np.nan

    # --- Aggregate popularity & averages ---
    pop = (
        work.groupby(["name"], as_index=False)
            .agg(
                count=("deck_id", "nunique"),
                avg_price=("price_num", "mean"),
                avg_cmc=("cmc_num", "mean"),
                type=("type", "first"),
            )
            .sort_values("count", ascending=False)
    )

    # Drop top N staples if requested
    if exclude_top_n and exclude_top_n > 0 and len(pop) > exclude_top_n:
        pop = pop.iloc[exclude_top_n:, :]

    # Inclusion rate
    num_decks = work["deck_id"].nunique()
    pop["inclusion_rate"] = (pop["count"] / num_decks * 100).round(2) if num_decks else 0.0

    # Nice rounding for display
    pop["avg_price"] = pop["avg_price"].round(2)
    pop["avg_cmc"] = pop["avg_cmc"].round(2)

    return pop.reset_index(drop=True)

# ---------- Mana curve (spells only) ----------

@st.cache_data(show_spinner=False)
def mana_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "cmc" not in df.columns or "type" not in df.columns:
        return pd.DataFrame(columns=["cmc","count"])
    spells = df[~df["type"].astype(str).str.contains("Land", na=False)].copy()
    spells["cmc"] = pd.to_numeric(spells["cmc"], errors="coerce")
    spells = spells.dropna(subset=["cmc"])
    return spells.groupby("cmc").size().reset_index(name="count").sort_values("cmc")

# ---------- Type breakdown (average per deck) ----------

@st.cache_data(show_spinner=False)
def type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if not _ensure_cols(df, ["deck_id", "type"]):
        return pd.DataFrame(columns=["type","avg_count_per_deck"])
    per_deck = df.groupby(["deck_id","type"]).size().reset_index(name="n")
    avg = per_deck.groupby("type")["n"].mean().reset_index(name="avg_count_per_deck")
    return avg.sort_values("avg_count_per_deck", ascending=False)

# ---------- Co-occurrence (top N by inclusion) ----------

@st.cache_data(show_spinner=False)
def cooccurrence_matrix(
    df: pd.DataFrame,
    *,
    top_n: int = 40,
    min_decks: int = 2,
    normalize: bool = False,      # False = raw counts; True = Jaccard similarity
    zero_diagonal: bool = True,
) -> pd.DataFrame:
    """
    Build a card × card co-occurrence matrix across decks.

    Parameters
    ----------
    df : DataFrame with at least ['deck_id', 'name'] and ideally ['type','cmc','price_clean'].
    top_n : number of most frequent cards to include (after min_decks filter).
    min_decks : keep cards that appear in at least this many unique decks.
    normalize : if True, return Jaccard similarities instead of raw co-occurrence counts.
    zero_diagonal : if True, set the diagonal to 0.

    Returns
    -------
    DataFrame (index='Card B', columns='Card A') with counts or Jaccard values.
    """
    if df is None or df.empty or "deck_id" not in df.columns or "name" not in df.columns:
        return pd.DataFrame()

    # Use each card at most once per deck
    dfu = df[["deck_id", "name"]].dropna().drop_duplicates()

    # Filter to cards that appear in >= min_decks unique decks
    deck_counts = dfu.groupby("name")["deck_id"].nunique()
    keep_cards = deck_counts[deck_counts >= min_decks].sort_values(ascending=False)
    if keep_cards.empty:
        return pd.DataFrame()

    # Restrict to top_n cards
    top_cards = keep_cards.head(top_n).index
    dfu = dfu[dfu["name"].isin(top_cards)]

    # Build deck → set(cards) mapping
    cards_by_deck = dfu.groupby("deck_id")["name"].apply(set)

    # Initialize order & matrix
    order = list(top_cards)
    idx = {c: i for i, c in enumerate(order)}
    mat = np.zeros((len(order), len(order)), dtype=float)

    # Fill co-occurrence counts
    for deck_cards in cards_by_deck:
        deck_list = [c for c in deck_cards if c in idx]
        for i in range(len(deck_list)):
            ci = deck_list[i]
            ii = idx[ci]
            for j in range(i, len(deck_list)):
                cj = deck_list[j]
                jj = idx[cj]
                mat[ii, jj] += 1.0
                if ii != jj:
                    mat[jj, ii] += 1.0

    M = pd.DataFrame(mat, index=order, columns=order)

    if normalize:
        # Jaccard similarity: |A∩B| / |A∪B| with |A| = diag
        diag = np.diag(M.values).astype(float)
        denom = (diag[:, None] + diag[None, :] - M.values)
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.where(denom > 0, M.values / denom, 0.0)
        M = pd.DataFrame(J, index=order, columns=order)

    if zero_diagonal:
        np.fill_diagonal(M.values, 0.0)

    # Label axes for your heatmap code
    M.index.name = "Card B"
    M.columns.name = "Card A"
    return M