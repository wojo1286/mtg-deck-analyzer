# analysis/stats.py
from __future__ import annotations
import itertools
import pandas as pd

def _deck_key(df: pd.DataFrame) -> str:
    """Choose the most reliable column to identify a unique deck."""
    for col in ("deck_id", "deck_url", "deck_name"):
        if col in df.columns:
            return col
    # absolute fallback (shouldn't happen with our scraper)
    return "deck_name"

def _clean_price(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    s = (series.astype(str)
               .str.replace(r"[^\d.\-]", "", regex=True)
               .replace({"": None}))
    return pd.to_numeric(s, errors="coerce")

def _as_numeric(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")

def inclusion_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["name","count","inclusion_rate","avg_price","avg_cmc","type"])

    key = _deck_key(df)
    tmp = df.copy()
    tmp["price_num"] = _clean_price(tmp.get("price"))
    tmp["cmc_num"]   = _as_numeric(tmp.get("cmc"))

    num_decks = max(1, tmp[key].nunique())

    grp = (tmp.groupby("name")
             .agg(count=(key, "nunique"),
                  avg_price=("price_num", "mean"),
                  avg_cmc=("cmc_num", "mean"),
                  type=("type", "first"))
             .reset_index())
    grp["inclusion_rate"] = (grp["count"] / float(num_decks)) * 100.0
    grp = grp[["name","count","inclusion_rate","avg_price","avg_cmc","type"]]
    grp = grp.sort_values(["count","inclusion_rate","name"], ascending=[False, False, True], kind="stable")
    return grp

def mana_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["cmc","count"])
    spells = df[~df["type"].str.contains("Land", na=False)].copy()
    spells["cmc_num"] = _as_numeric(spells.get("cmc"))
    curve = (spells.dropna(subset=["cmc_num"])
                   .groupby("cmc_num")
                   .size()
                   .reset_index(name="count")
                   .rename(columns={"cmc_num":"cmc"}))
    try:
        curve["cmc"] = curve["cmc"].astype(int)
    except Exception:
        pass
    return curve.sort_values("cmc")

def type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["type","avg_count_per_deck"])
    key = _deck_key(df)
    num_decks = max(1, df[key].nunique())
    counts = df.groupby([key,"type"]).size().reset_index(name="n")
    avg = (counts.groupby("type")["n"].mean().reset_index(name="avg_count_per_deck")
                 .sort_values("avg_count_per_deck", ascending=False))
    return avg

def cooccurrence_matrix(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    key = _deck_key(df)

    inc = inclusion_table(df)
    keep = set(inc.head(top_n)["name"].tolist())

    per_deck = (df[df["name"].isin(keep)]
                  .groupby(key)["name"]
                  .apply(lambda s: sorted(set(s))))
    if per_deck.empty:
        return pd.DataFrame()

    names = sorted(keep)
    idx = {n:i for i,n in enumerate(names)}
    n = len(names)

    import numpy as np
    mat = np.zeros((n, n), dtype=int)
    for cards in per_deck:
        # increment pairwise co-occurrence for each deck
        for i in range(len(cards)):
            for j in range(i+1, len(cards)):
                a, b = cards[i], cards[j]
                ia, ib = idx[a], idx[b]
                mat[ia, ib] += 1
                mat[ib, ia] += 1
        # diagonal = deck frequency (optional: comment this out to zero diagonal)
        for a in cards:
            mat[idx[a], idx[a]] += 1

    return pd.DataFrame(mat, index=names, columns=names)

def budget_filtered(df: pd.DataFrame, max_price: float | None) -> pd.DataFrame:
    if df is None or df.empty or max_price is None:
        return df if df is not None else pd.DataFrame()
    tmp = df.copy()
    tmp["price_num"] = _clean_price(tmp.get("price"))
    return tmp[(tmp["price_num"].isna()) | (tmp["price_num"] <= float(max_price))].drop(columns=["price_num"], errors="ignore")
