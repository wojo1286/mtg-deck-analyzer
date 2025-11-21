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


def card_tag_synergy(df: pd.DataFrame, deck_col: str | None = None) -> pd.DataFrame:
    """
    Compute tag-level synergy for tagged cards.

    Expects a row-per-card-per-deck DataFrame `df` with at least:
      - 'name'      : card name
      - 'category'  : pipe-delimited tags, e.g. 'Ramp|Mana Rock'
      - a deck id column (see `deck_col`)

    Returns a DataFrame with columns:
      ['name', 'tag', 'delta', 'p_tag_given', 'p_tag']

    Where:
      - p_tag       = P(tag appears in a deck)
      - p_tag_given = P(tag appears in a deck | card is in that deck)
      - delta       = p_tag_given - p_tag (synergy vs baseline)
    """

    if df is None or df.empty:
        return pd.DataFrame(columns=["name", "tag", "delta", "p_tag_given", "p_tag"])

    # Work out which column identifies decks
    if deck_col is None:
        for candidate in ("deck_id", "deck_idx", "deck_index", "deck"):
            if candidate in df.columns:
                deck_col = candidate
                break
        else:
            raise KeyError(
                "card_tag_synergy() could not find a deck id column. "
                "Expected one of: 'deck_id', 'deck_idx', 'deck_index', 'deck'."
            )

    work = df.copy()

    # Normalize category strings
    cat = work.get("category")
    if cat is None:
        # No tags at all
        return pd.DataFrame(columns=["name", "tag", "delta", "p_tag_given", "p_tag"])

    work["category"] = (
        cat.fillna("")
        .astype(str)
        .str.strip()
    )

    # Expand tags: one row per (name, deck, tag)
    tag_rows: list[tuple[str, object, str]] = []

    for _, row in work.iterrows():
        raw = row["category"]
        if not raw:
            continue
        if isinstance(raw, str) and raw.lower() == "uncategorized":
            continue

        tags = [t.strip() for t in raw.split("|") if t.strip()]
        if not tags:
            continue

        for tag in tags:
            tag_rows.append((row["name"], row[deck_col], tag))

    if not tag_rows:
        # No actual tags -> nothing to compute
        return pd.DataFrame(columns=["name", "tag", "delta", "p_tag_given", "p_tag"])

    tags_df = pd.DataFrame(tag_rows, columns=["name", deck_col, "tag"])

    # Universe of decks
    n_decks = work[deck_col].nunique()
    if n_decks == 0:
        return pd.DataFrame(columns=["name", "tag", "delta", "p_tag_given", "p_tag"])

    # How often each tag appears in decks
    decks_with_tag = tags_df.groupby("tag")[deck_col].nunique().rename("decks_with_tag")
    p_tag = decks_with_tag / float(n_decks)

    # How often each card appears in decks
    decks_with_card = work.groupby("name")[deck_col].nunique().rename("decks_with_card")

    # How often each (card, tag) pair appears together in decks
    card_tag_decks = (
        tags_df.groupby(["name", "tag"])[deck_col]
        .nunique()
        .rename("decks_with_both")
    )

    result = card_tag_decks.to_frame().reset_index()

    result = result.merge(
        decks_with_card.reset_index(),
        on="name",
        how="left",
    )

    result = result.merge(
        p_tag.rename("p_tag").reset_index(),
        on="tag",
        how="left",
    )

    # Probabilities
    result["p_tag_given"] = (
        result["decks_with_both"] / result["decks_with_card"].replace(0, np.nan)
    )
    result["delta"] = result["p_tag_given"] - result["p_tag"]

    # Clean up and sort – cards with the highest positive delta first
    result = result[["name", "tag", "delta", "p_tag_given", "p_tag"]]
    result = result.sort_values(["delta", "name", "tag"], ascending=[False, True, True])

    return result.reset_index(drop=True)
