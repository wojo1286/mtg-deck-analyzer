from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# We’ll reuse your popularity computation from analysis.stats if available.
# If you prefer not to import, you can inline a simple popularity table function.
try:
    from analysis.stats import inclusion_table
    _HAS_INC_TABLE = True
except Exception:
    _HAS_INC_TABLE = False


# ---------------------------
# Helpers / data structures
# ---------------------------

@dataclass
class Range:
    min: int
    max: int
    cur: int = 0

    def need(self) -> int:
        """How many to reach min (>=0)."""
        return max(0, self.min - self.cur)

    def room(self) -> int:
        """How many until max (>=0)."""
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
    """Split pipe-separated categories into lists; empty -> []."""
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
    and a composite score (popularity/efficiency). Safe if 'category' or
    'price_clean' are missing.
    """
    must_exclude = set(must_exclude or [])
    must_include = set(must_include or [])

    # Minimal hard requirements
    cols_needed = {"name", "type", "cmc", "deck_id"}
    missing_hard = cols_needed - set(df.columns)
    if missing_hard:
        raise ValueError(f"prepare_candidates: missing columns: {sorted(missing_hard)}")

    base = (
        df[["name", "type", "cmc", "deck_id"] + ([c for c in ["price_clean", "price", "category"] if c in df.columns])]
        .copy()
        .dropna(subset=["name"])
    )

    # Ensure price_clean
    if "price_clean" not in base.columns:
        if "price" in base.columns:
            base["price_clean"] = pd.to_numeric(
                base["price"].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )
        else:
            base["price_clean"] = np.nan

    # Ensure category (string, possibly empty)
    if "category" not in base.columns:
        base["category"] = ""

    # Popularity proxy = unique decks per card
    if _HAS_INC_TABLE:
        pop = inclusion_table(base)[["name", "count"]].rename(columns={"count": "deck_count"})
    else:
        pop = (
            base.groupby("name")["deck_id"]
            .nunique()
            .reset_index(name="deck_count")
            .sort_values("deck_count", ascending=False)
        )

    # Keep the lowest-price row per card to give cheap options a nudge
    one_row = (
        base.sort_values(["name", "price_clean"], ascending=[True, True])
        .drop_duplicates(subset=["name"], keep="first")
    )

    cand = one_row.merge(pop, on="name", how="left")
    cand["deck_count"] = cand["deck_count"].fillna(0).astype(float)

    # Split categories into list
    cand["category_list"] = cand["category"].fillna("").astype(str).apply(lambda s: [t for t in s.split("|") if t])

    # CMC filled for efficiency
    med_cmc = pd.to_numeric(cand["cmc"], errors="coerce").median()
    cand["cmc_filled"] = pd.to_numeric(cand["cmc"], errors="coerce").fillna(med_cmc if pd.notna(med_cmc) else 3)

    # Composite score: popularity up, cmc down, slight price penalty
    price = pd.to_numeric(cand["price_clean"], errors="coerce").fillna(0.0)
    cand["efficiency"] = cand["deck_count"] / (cand["cmc_filled"] + 1.0)
    cand["score"] = cand["efficiency"] / (1.0 + (price / 50.0))

    # Excludes (must_include handled by the caller seeding initial list)
    cand = cand[~cand["name"].isin(must_exclude)].reset_index(drop=True)

    # Deterministic order
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
    List[str]:
    """
    Greedy selector that prioritizes meeting mins, respects maxes, and chooses
    highest-scoring candidates that *also* help unmet needs.

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
        if prefer_nonlands_until is not None and sum(
            (1 for n in deck if candidates.loc[candidates["name"] == n, "type"].astype(str).str.contains("Land").any())
        ) < 0:  # computed inline later; keep logic simple by soft-blocking in scoring
            pass

        # Evaluate need scores filtered to available candidates
        pool = candidates[~candidates["name"].isin(used)]

        if pool.empty:
            break

        # Filter “hard” maxes & optional early non-land bias
        pool = pool[pool.apply(lambda r: _card_fits(state, r), axis=1)]
        if prefer_nonlands_until is not None:
            nonlands_taken = sum(
                (1 for n in deck if not str(candidates.loc[candidates["name"] == n, "type"].values[0]).startswith("Land"))
            )
            if nonlands_taken < prefer_nonlands_until:
                pool = pool[~pool["type"].astype(str).str.startswith("Land")]

        if pool.empty:
            break

        # Score: unmet-need first, then composite score as tiebreaker
        need = pool.apply(lambda r: _need_score(state, r), axis=1)
        # choose the best by (need_score, score) with deterministic name tiebreak
        ranked = (
            pool.assign(_need=need)
            .sort_values(by=["_need", "score", "name"], ascending=[False, False, True])
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
    List[str]:
    """
    Builds an “average” shell:
    - mean counts per type across decks
    - average non-basic lands
    - fills basics by commander color identity
    - fills remaining with most popular spells
    """
    if df is None or df.empty:
        return []

    # Basic setup
    basic_land_names = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}

    # Known cards per deck
    per_deck_counts = df.groupby("deck_id").size()
    # Estimate basics as the remainder (won’t be perfect, but gives useful averages)
    basics_inferred = (total_size - per_deck_counts).clip(lower=0)

    # Types per deck
    types_per_deck = (
        df.assign(primary_type=df["type"].astype(str))
          .groupby(["deck_id", "primary_type"])
          .size()
          .unstack(fill_value=0)
    )
    avg_types = types_per_deck.mean().sort_values(ascending=False)

    # Non-basic lands average
    land_df = df[df["type"].astype(str).str.contains("Land", na=False)]
    non_basic_land_df = land_df[~land_df["name"].isin(basic_land_names)]
    avg_non_basics = non_basic_land_df.groupby("deck_id").size().mean() if not non_basic_land_df.empty else 0.0

    # Average basics (inferred)
    avg_basics = basics_inferred.mean() if not basics_inferred.empty else 0.0

    # Turn into integer template summing to (total_size - commander slot if you want; here we just do total_size)
    template = avg_types.copy()

    # Move Land to (non-basic + basics)
    template["Non-Basic Land"] = round(avg_non_basics)
    template["Basic Land"] = round(avg_basics)
    if "Land" in template:
        template = template.drop(labels=["Land"])

    # Normalize to (total_size - 0). We target 99 spells + 1 commander in EDH usually; keep 100 here, caller can adjust.
    tgt = total_size
    s = int(round(template.sum()))
    if s <= 0:
        return []

    scaled = (template / template.sum() * (tgt)).round().astype(int)
    # Fix rounding drift
    drift = tgt - int(scaled.sum())
    if drift != 0 and not scaled.empty:
        scaled.iloc[0] += drift

    # Build deck by picking top-N popular for each type bucket
    names_used: set[str] = set()
    deck: List[str] = []

    # Popularity table on spells only
    spells = df[~df["type"].astype(str).str.contains("Land", na=False)]
    if _HAS_INC_TABLE:
        pop = inclusion_table(spells)[["name", "count"]].rename(columns={"count": "deck_count"})
    else:
        pop = spells.groupby("name")["deck_id"].nunique().reset_index(name="deck_count")
    pop = pop.sort_values(["deck_count", "name"], ascending=[False, True])

    # Helper to pull top cards for a predicate until quota reached
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
            if len(deck) >= len(names_used) and len(deck) % max(n, 1) == 0:
                # not a precise stop, but we exit after filling each bucket outer scope
                pass
            if len([x for x in deck if predicate(spells[spells["name"] == x])]) >= n:
                break

    # Fill by primary types (excluding lands)
    for t, n in scaled.items():
        if n <= 0:
            continue
        if "Land" in t:
            continue
        _pull(lambda df_: df_["type"].astype(str).eq(t), int(n))

    # Non-basics
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

    # Basics by color identity
    num_basic = int(scaled.get("Basic Land", 0))
    if num_basic > 0:
        color_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        basics = [color_map[c] for c in (commander_colors or []) if c in color_map]
        if not basics:
            basics = ["Wastes"]
        # even split w/ remainder
        q, r = divmod(num_basic, len(basics))
        for i, b in enumerate(basics):
            deck.extend([b] * (q + (1 if i < r else 0)))

    # Backfill if short
    if len(deck) < tgt:
        # any remaining popular spells
        for _, r in pop.iterrows():
            if len(deck) >= tgt:
                break
            nm = r["name"]
            if nm in names_used:
                continue
            deck.append(nm)
            names_used.add(nm)

    return deck[:tgt]


# ---------------------------
# Deck summary
# ---------------------------

def summarize_deck(
    decklist: List[str],
    ref_df: pd.DataFrame,
    total_size: int = 100,
) -> Dict[str, pd.DataFrame | float | int | Dict]:
    """
    Summarize a generated deck using card info from ref_df.

    Returns dict with:
      - counts_by_type (DataFrame)
      - cmc_curve (DataFrame)
      - price_total (float)
      - basics / non_basics (ints)
      - functions_covered (DataFrame)
    """
    if not decklist or ref_df is None or ref_df.empty:
        return {}

    cards = pd.DataFrame({"name": decklist})
    info = (
        cards.merge(
            ref_df[["name", "type", "cmc", "price_clean", "category"]].drop_duplicates("name"),
            on="name",
            how="left",
        )
    )

    # Types
    counts_by_type = (
        info["type"].fillna("Unknown").astype(str).value_counts().reset_index()
        .rename(columns={"index": "type", "type": "count"})
    )

    # CMC curve (non-lands)
    curve = (
        info[~info["type"].astype(str).str.contains("Land", na=False)]
        .assign(cmc_num=pd.to_numeric(info["cmc"], errors="coerce").fillna(0).astype(int))
        .groupby("cmc_num")
        .size()
        .reset_index(name="count")
        .rename(columns={"cmc_num": "cmc"})
        .sort_values("cmc")
    )

    # Price total
    price_total = pd.to_numeric(info["price_clean"], errors="coerce").fillna(0).sum()

    # Basics vs non-basics
    basic_land_names = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
    land_mask = info["type"].astype(str).str.contains("Land", na=False)
    basics = info[land_mask & info["name"].isin(basic_land_names)]
    non_basics = info[land_mask & ~info["name"].isin(basic_land_names)]

    # Functions covered
    funcs = []
    for _, r in info.iterrows():
        cats = (str(r.get("category") or "")).split("|") if pd.notna(r.get("category")) else []
        for c in cats:
            c = c.strip()
            if c:
                funcs.append(c)
    fc = (
        pd.Series(funcs).value_counts().reset_index()
        .rename(columns={"index": "function", 0: "count"})
        if funcs else pd.DataFrame(columns=["function", "count"])
    )

    return {
        "counts_by_type": counts_by_type,
        "cmc_curve": curve,
        "price_total": float(price_total),
        "basics": int(len(basics)),
        "non_basics": int(len(non_basics)),
        "functions_covered": fc,
        "cards_enriched": info,  # handy table to display/export
    }
