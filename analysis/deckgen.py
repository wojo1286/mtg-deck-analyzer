# analysis/deckgen.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

try:
    from analysis.stats import inclusion_table

    _HAS_INC_TABLE = True
except Exception:
    _HAS_INC_TABLE = False


@dataclass
class Range:
    min: int
    max: int
    cur: int = 0

    def need(self) -> int:
        return max(0, self.min - self.cur)

    def room(self) -> int:
        return max(0, self.max - self.cur)

    def can_add(self) -> bool:
        return self.cur < self.max

    def bump(self, k: int = 1) -> None:
        self.cur += k


@dataclass
class ConstraintState:
    types: Dict[str, Range]
    funcs: Dict[str, Range]

    def clone(self) -> "ConstraintState":
        return ConstraintState(
            types={k: Range(v.min, v.max, v.cur) for k, v in self.types.items()},
            funcs={k: Range(v.min, v.max, v.cur) for k, v in self.funcs.items()},
        )


def _normalize_constraints(
    type_constraints: Dict[str, Tuple[int, int]] | None,
    func_constraints: Dict[str, Tuple[int, int]] | None,
) -> ConstraintState:
    def _norm(d: Dict[str, Tuple[int, int]] | None) -> Dict[str, Range]:
        out: Dict[str, Range] = {}
        if d:
            for k, v in d.items():
                low, high = int(v[0]), int(v[1])
                if high < low:
                    high = low
                out[k] = Range(low, high, 0)
        return out

    return ConstraintState(types=_norm(type_constraints), funcs=_norm(func_constraints))


def _split_categories(series: pd.Series) -> List[List[str]]:
    vals = series.fillna("").astype(str).tolist()
    return [v.split("|") if v else [] for v in vals]


# ---------------------------
# Candidate preparation
# ---------------------------


def prepare_candidates(
    df: pd.DataFrame,
    must_exclude: Iterable[str] | None = None,
    must_include: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Deduplicate by name, attach helper columns: category_list, cmc_filled,
    and a composite score (popularity/efficiency).
    NOTE: 'category' is optional; if missing we treat it as empty.
    """
    must_exclude = set(must_exclude or [])
    must_include = set(must_include or [])

    cols_needed = {"name", "type", "cmc", "price_clean"}  # 'category' optional now
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"prepare_candidates: missing columns: {sorted(missing)}")

    work = df.copy()
    if "category" not in work.columns:
        work["category"] = ""  # ensure downstream code works

    base = (
        work[["name", "type", "cmc", "price_clean", "category", "deck_id"]]
        .copy()
        .dropna(subset=["name"])
    )

    # Popularity proxy = unique decks count per card.
    if _HAS_INC_TABLE:
        pop = inclusion_table(base)[["name", "count"]].rename(columns={"count": "deck_count"})
    else:
        pop = (
            base.groupby("name")["deck_id"]
            .nunique()
            .reset_index(name="deck_count")
            .sort_values("deck_count", ascending=False)
        )

    one_row = base.sort_values(["name", "price_clean"], ascending=[True, True]).drop_duplicates(
        subset=["name"], keep="first"
    )

    cand = one_row.merge(pop, on="name", how="left")
    cand["deck_count"] = cand["deck_count"].fillna(0).astype(float)

    # Categories
    cand["category_list"] = _split_categories(cand["category"])

    # CMC filled for efficiency
    med_cmc = pd.to_numeric(cand["cmc"], errors="coerce").median()
    cand["cmc_filled"] = pd.to_numeric(cand["cmc"], errors="coerce").fillna(
        med_cmc if pd.notna(med_cmc) else 3
    )

    # Composite score: popularity vs efficiency with a soft price tilt
    price = pd.to_numeric(cand["price_clean"], errors="coerce").fillna(0.0)
    cand["efficiency"] = cand["deck_count"] / (cand["cmc_filled"] + 1.0)
    cand["score"] = cand["efficiency"] / (1.0 + (price / 50.0))

    # Filter excludes; keep must_include (caller seeds deck before selection)
    cand = cand[~cand["name"].isin(must_exclude)].reset_index(drop=True)
    cand = cand.sort_values(["score", "name"], ascending=[False, True]).reset_index(drop=True)
    return cand


# ---------------------------
# Constraint-aware selection
# ---------------------------


def _apply_initials(state: ConstraintState, df: pd.DataFrame, initial_cards: Iterable[str]) -> None:
    initial = set(initial_cards or [])
    if not initial:
        return
    sub = df[df["name"].isin(initial)]
    for _, row in sub.iterrows():
        t = str(row.get("type") or "")
        if t in state.types:
            state.types[t].bump()
        for cat in row.get("category_list", []):
            if cat in state.funcs:
                state.funcs[cat].bump()


def _card_fits(state: ConstraintState, row: pd.Series) -> bool:
    """Hard max checks: reject if picking this would exceed any max already at limit."""
    t = str(row.get("type") or "")
    if t in state.types and not state.types[t].can_add():
        return False
    for cat in row.get("category_list", []):
        if cat in state.funcs and not state.funcs[cat].can_add():
            return False
    return True


def _need_score(state: ConstraintState, row: pd.Series) -> float:
    """
    How much does this card help satisfy unmet mins?
    We sum (remaining need) across all matching constraints, plus a tiny tie-break on type coverage.
    """
    s = 0.0
    t = str(row.get("type") or "")
    if t in state.types:
        s += 2.0 * state.types[t].need()  # type mins weigh a bit more

    cats = row.get("category_list", []) or []
    for c in cats:
        if c in state.funcs:
            s += 1.0 * state.funcs[c].need()

    # small nudge for covering more unique functions (helps breadth early)
    s += 0.01 * len(cats)
    return s


def _accept(state: ConstraintState, row: pd.Series) -> None:
    t = str(row.get("type") or "")
    if t in state.types:
        state.types[t].bump()
    for c in row.get("category_list", []) or []:
        if c in state.funcs:
            state.funcs[c].bump()


def fill_deck_slots(
    candidates: pd.DataFrame,
    *,
    type_constraints: Dict[str, Tuple[int, int]] | None = None,
    func_constraints: Dict[str, Tuple[int, int]] | None = None,
    initial: Iterable[str] | None = None,
    total_size: int = 100,
    prefer_nonlands_until: int | None = None,
) -> List[str]:
    """ "
    Args:
        candidates: output of prepare_candidates()
        type_constraints / func_constraints: {key: (min, max)}
        initial: cards seeded up-front (must-haves)
        total_size: final deck length target (usually 100)
        prefer_nonlands_until: if set, avoid 'Land' type until we have this many non-lands.
    """
    state = _normalize_constraints(type_constraints, func_constraints)
    deck: List[str] = []

    initial = list(initial or [])
    used = set()
    # Seed initial cards (present in candidates)
    _apply_initials(state, candidates, initial)
    for n in initial:
        if n in set(candidates["name"]):
            deck.append(n)
            used.add(n)

    while len(deck) < total_size:
        # apply optional early non-land preference
        if (
            prefer_nonlands_until is not None
            and sum(
                (
                    1
                    for n in deck
                    if candidates.loc[candidates["name"] == n, "type"]
                    .astype(str)
                    .str.contains("Land")
                    .any()
                )
            )
            < 0
        ):  # computed inline later; keep logic simple by soft-blocking in scoring
            pass

        # Evaluate need scores filtered to available candidates
        pool = candidates[~candidates["name"].isin(used)]

        if pool.empty:
            break

        # Filter “hard” maxes & optional early non-land bias
        pool = pool[pool.apply(lambda r: _card_fits(state, r), axis=1)]
        if prefer_nonlands_until is not None:
            nonlands_taken = sum(
                (
                    1
                    for n in deck
                    if not str(
                        candidates.loc[candidates["name"] == n, "type"].values[0]
                    ).startswith("Land")
                )
            )
            if nonlands_taken < prefer_nonlands_until:
                pool = pool[~pool["type"].astype(str).str.startswith("Land")]

        if pool.empty:
            break

        # Score: unmet-need first, then composite score as tiebreaker
        need = pool.apply(lambda r: _need_score(state, r), axis=1)
        # choose the best by (need_score, score) with deterministic name tiebreak
        ranked = pool.assign(_need=need).sort_values(
            by=["_need", "score", "name"], ascending=[False, False, True]
        )

        pick = ranked.iloc[0]
        deck.append(pick["name"])
        used.add(pick["name"])
        _accept(state, pick)

    # Trim in case of overshoot (shouldn't happen, but safe)
    return deck[:total_size]


# ---------------------------
# Average deck generation
# ---------------------------


def generate_average_deck(
    df: pd.DataFrame,
    *,
    total_size: int = 100,
    commander_colors: List[str] | None = None,
) -> List[str]:
    """
    Builds an “average” shell:
    - mean counts per type across decks
    - average non-basic lands
    - fills basics by commander color identity
    - fills remaining with most popular spells
    """
    if df is None or df.empty:
        return []

    basic_land_names = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}

    per_deck_counts = df.groupby("deck_id").size()
    basics_inferred = (total_size - per_deck_counts).clip(lower=0)

    types_per_deck = (
        df.assign(primary_type=df["type"].astype(str))
        .groupby(["deck_id", "primary_type"])
        .size()
        .unstack(fill_value=0)
    )
    avg_types = types_per_deck.mean().sort_values(ascending=False)

    land_df = df[df["type"].astype(str).str.contains("Land", na=False)]
    non_basic_land_df = land_df[~land_df["name"].isin(basic_land_names)]
    avg_non_basics = (
        non_basic_land_df.groupby("deck_id").size().mean() if not non_basic_land_df.empty else 0.0
    )

    avg_basics = basics_inferred.mean() if not basics_inferred.empty else 0.0

    template = avg_types.copy()
    template["Non-Basic Land"] = round(avg_non_basics)
    template["Basic Land"] = round(avg_basics)
    if "Land" in template:
        template = template.drop(labels=["Land"])

    tgt = total_size
    if int(round(template.sum())) <= 0:
        return []

    scaled = (template / template.sum() * tgt).round().astype(int)
    drift = tgt - int(scaled.sum())
    if drift != 0 and not scaled.empty:
        scaled.iloc[0] += drift

    names_used: set[str] = set()
    deck: List[str] = []

    spells = df[~df["type"].astype(str).str.contains("Land", na=False)]
    if _HAS_INC_TABLE:
        pop = inclusion_table(spells)[["name", "count"]].rename(columns={"count": "deck_count"})
    else:
        pop = spells.groupby("name")["deck_id"].nunique().reset_index(name="deck_count")
    pop = pop.sort_values(["deck_count", "name"], ascending=[False, True])

    def _pull(predicate, n: int):
        nonlocal deck, names_used
        if n <= 0:
            return
        pool = (
            spells[predicate(spells)]
            .drop_duplicates(subset=["name"])
            .merge(pop, on="name", how="left")
            .sort_values(["deck_count", "name"], ascending=[False, True])
        )
        for _, r in pool.iterrows():
            if len(deck) >= tgt:
                break
            nm = r["name"]
            if nm in names_used:
                continue
            deck.append(nm)
            names_used.add(nm)
            if len([x for x in deck if predicate(spells[spells["name"] == x])]) >= n:
                break

    for t, n in scaled.items():
        if n <= 0 or "Land" in t:
            continue
        _pull(lambda df_: df_["type"].astype(str).eq(t), int(n))

    num_nb = int(scaled.get("Non-Basic Land", 0))
    if num_nb > 0 and not non_basic_land_df.empty:
        nb_pop = (
            non_basic_land_df.groupby("name")["deck_id"]
            .nunique()
            .reset_index(name="deck_count")
            .sort_values(["deck_count", "name"], ascending=[False, True])
        )
        for _, r in nb_pop.iterrows():
            if len(deck) >= tgt or num_nb <= 0:
                break
            nm = r["name"]
            if nm in names_used:
                continue
            deck.append(nm)
            names_used.add(nm)
            num_nb -= 1

    num_basic = int(scaled.get("Basic Land", 0))
    if num_basic > 0:
        color_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        basics = [color_map[c] for c in (commander_colors or []) if c in color_map] or ["Wastes"]
        q, r = divmod(num_basic, len(basics))
        for i, b in enumerate(basics):
            deck.extend([b] * (q + (1 if i < r else 0)))

    if len(deck) < tgt:
        for _, r in pop.iterrows():
            if len(deck) >= tgt:
                break
            nm = r["name"]
            if nm in names_used:
                continue
            deck.append(nm)
            names_used.add(nm)

    return deck[:tgt]

def summarize_deck(
    deck: List[str],
    df_cards: pd.DataFrame,
    total_size: int | None = None,
) -> dict:
    """
    Build a simple summary of a generated deck for the dashboard.

    Parameters
    ----------
    deck : ordered list of card names selected for the deck (may contain duplicates).
    df_cards : card-level data frame used when generating the deck. Must contain
        at least `name`, `type`, `cmc`, `price_clean`, and optionally `category`.
    total_size : nominal deck size target (unused except for sanity checks).

    Returns
    -------
    Dict with the following keys:
        counts_by_type : DataFrame[type, count]
        cmc_curve      : DataFrame[cmc, count] (spells only)
        functions_covered : DataFrame[category, count]
        price_total    : float
        basics         : int
        non_basics     : int
    """
    if not deck or df_cards is None or df_cards.empty:
        return {}

    # Ensure the columns we need exist
    base_cols = ["name", "type", "cmc", "price_clean", "category"]
    work = df_cards.copy()
    for c in base_cols:
        if c not in work.columns:
            if c == "category":
                work[c] = ""
            else:
                work[c] = pd.NA

    # One row per card name with a "count" of copies in the generated deck
    deck_df = (
        pd.DataFrame({"name": deck})
        .groupby("name")
        .size()
        .reset_index(name="count")
        .merge(work[base_cols], on="name", how="left")
    )

    # --- Counts by primary type ---
    counts_by_type = (
        deck_df.groupby("type")["count"]
        .sum()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # --- Price total (copies × price_clean, unknown price = 0) ---
    price_num = pd.to_numeric(deck_df["price_clean"], errors="coerce").fillna(0.0)
    price_total = float((price_num * deck_df["count"]).sum())

    # --- Land breakdown: basics vs non-basics ---
    basic_names = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}

    type_str = deck_df["type"].astype(str)
    is_land = type_str.str.contains("Land", na=False)
    is_basic = (
        deck_df["name"].isin(basic_names)
        | type_str.str.contains("Basic Land", na=False)
    )

    basics = int(deck_df.loc[is_basic, "count"].sum())
    non_basics = int(deck_df.loc[is_land & ~is_basic, "count"].sum())

    # --- CMC curve (spells only, weighted by copies) ---
    spells = deck_df[~type_str.str.contains("Land", na=False)].copy()
    spells["cmc_num"] = pd.to_numeric(spells["cmc"], errors="coerce")

    cmc_curve = (
        spells.loc[spells["cmc_num"].notna()]
        .groupby("cmc_num")["count"]
        .sum()
        .reset_index()
        .rename(columns={"cmc_num": "cmc"})
        .sort_values("cmc")
    )

    # --- Functional coverage from Tagger categories (if present) ---
    work_cat = deck_df.copy()
    work_cat["category"] = work_cat["category"].fillna("").astype(str)
    work_cat["category_list"] = work_cat["category"].str.split("|")

    rows = []
    for _, row in work_cat.iterrows():
        for cat in row["category_list"]:
            if not cat or cat == "Uncategorized":
                continue
            rows.append({"category": cat, "count": row["count"]})

    if rows:
        functions_covered = (
            pd.DataFrame(rows)
            .groupby("category")["count"]
            .sum()
            .reset_index()
            .sort_values("count", ascending=False)
        )
    else:
        functions_covered = pd.DataFrame(columns=["category", "count"])

    return {
        "counts_by_type": counts_by_type,
        "cmc_curve": cmc_curve,
        "functions_covered": functions_covered,
        "price_total": price_total,
        "basics": basics,
        "non_basics": non_basics,
    }