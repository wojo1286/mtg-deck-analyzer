# analysis/synergy.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_jaccard_matrix(
    df: pd.DataFrame,
    min_decks: int = 2,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Compute a card × card Jaccard similarity matrix."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "deck_id" not in df.columns or "name" not in df.columns:
        raise ValueError("DataFrame must contain 'deck_id' and 'name' columns")

    work = df.copy()
    if exclude:
        work = work[~work["name"].isin(exclude)]

    work = work[["deck_id", "name"]].dropna().drop_duplicates()

    deck_counts = work.groupby("name")["deck_id"].nunique()
    keep = deck_counts[deck_counts >= min_decks]
    if keep.empty:
        return pd.DataFrame()

    cards = keep.index.tolist()
    work = work[work["name"].isin(cards)]

    matrix = (
        work.assign(present=1)
        .pivot_table(
            index="deck_id",
            columns="name",
            values="present",
            fill_value=0,
            aggfunc="max",
        )
        .astype(float)
    )
    if matrix.empty:
        return pd.DataFrame()

    co_counts = matrix.T.dot(matrix)
    counts = np.diag(co_counts.values)
    union = counts[:, None] + counts[None, :] - co_counts.values

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, co_counts.values / union, 0.0)

    M = pd.DataFrame(jaccard, index=co_counts.index, columns=co_counts.columns)
    np.fill_diagonal(M.values, 0.0)
    return M


def card_synergy_density(
    df: pd.DataFrame,
    min_decks: int = 2,
    exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Compute average positive Jaccard similarity per card."""
    M = _compute_jaccard_matrix(df, min_decks=min_decks, exclude=exclude)
    if M.empty:
        return pd.DataFrame(columns=["name", "synergy_density", "top_partners"])

    vals = M.replace(0, np.nan)
    density = vals.mean(axis=1).fillna(0.0)

    partners = {}
    for name in M.index:
        row_sorted = M.loc[name].sort_values(ascending=False)
        top = row_sorted[row_sorted > 0].head(10)
        partners[name] = ", ".join(top.index)

    out = pd.DataFrame(
        {
            "name": density.index,
            "synergy_density": density.values,
            "top_partners": [partners[n] for n in density.index],
        }
    )
    return out.sort_values("synergy_density", ascending=False).reset_index(drop=True)


def card_tag_synergy(
    df: pd.DataFrame,
    min_decks: int = 2,
    category_col: str = "category",
) -> pd.DataFrame:
    """Compute simple tag-level synergy metrics for each card."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["name", "tag", "delta", "p_tag_given", "p_tag"])
    if "deck_id" not in df.columns or "name" not in df.columns:
        raise ValueError("DataFrame must contain 'deck_id' and 'name' columns")
    if category_col not in df.columns:
        return pd.DataFrame(columns=["name", "tag", "delta", "p_tag_given", "p_tag"])

    work = df.copy()
    work[category_col] = work[category_col].fillna("").astype(str)
    work["category_list"] = work[category_col].str.split("|")

    deck_tag = (
        work.explode("category_list")
        .query("category_list != '' and category_list != 'Uncategorized'")
        .drop_duplicates(subset=["deck_id", "category_list"])
    )
    if deck_tag.empty:
        return pd.DataFrame(columns=["name", "tag", "delta", "p_tag_given", "p_tag"])

    num_decks = df["deck_id"].nunique()
    if num_decks == 0:
        return pd.DataFrame(columns=["name", "tag", "delta", "p_tag_given", "p_tag"])

    tag_decks = deck_tag.groupby("category_list")["deck_id"].nunique()
    p_tag = (tag_decks / num_decks).to_dict()

    card_decks = (
        work[["name", "deck_id"]]
        .drop_duplicates()
        .groupby("name")["deck_id"]
        .apply(set)
        .to_dict()
    )

    rows: list[dict[str, float | str]] = []
    for card, dset in card_decks.items():
        if len(dset) < min_decks:
            continue
        sub_tags = deck_tag[deck_tag["deck_id"].isin(dset)]
        if sub_tags.empty:
            continue
        tag_counts = sub_tags.groupby("category_list")["deck_id"].nunique()
        for tag, decks_with_tag in tag_counts.items():
            p_tag_given = decks_with_tag / len(dset)
            base = p_tag.get(tag, 0.0)
            rows.append(
                {
                    "name": card,
                    "tag": tag,
                    "delta": p_tag_given - base,
                    "p_tag_given": p_tag_given,
                    "p_tag": base,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return (
        out.sort_values(["name", "delta"], ascending=[True, False])
        .groupby("name")
        .head(5)
        .reset_index(drop=True)
    )
