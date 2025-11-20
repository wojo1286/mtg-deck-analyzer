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


# Preferred order for primary type reporting
_PRIMARY_TYPE_ORDER = [
    "Creature",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
    "Planeswalker",
    "Land",
]


def _extract_primary_type_local(type_value: str | None) -> str:
    """Simple primary type extractor that avoids external dependencies."""
    text = str(type_value or "")
    for candidate in _PRIMARY_TYPE_ORDER:
        if candidate in text:
            return candidate
    return "Unknown"


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
    target_land_count: int | None = None,
    land_target: int | None = None,
) -> List[str]:
    """
    Builds an “average” shell that explicitly budgets land slots:
    - Targets a land count (default ~37% of deck size, clamped to [30, 40] for EDH).
    - Uses non-basic lands from the scraped pool when available.
    - Fills remaining land slots with basics based on commander color identity.
    - Fills the rest with the most popular spells (non-lands).

    `target_land_count` is the preferred knob; `land_target` is kept for
    backward compatibility and will override `target_land_count` when set.
    """
    if df is None or df.empty:
        return []

    basic_land_names = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
    commander_colors = commander_colors or []

    try:
        tgt = max(1, int(total_size))
    except Exception:
        tgt = 100

    # Backwards compatibility: `land_target` overrides `target_land_count` if provided
    land_target = target_land_count if target_land_count is not None else land_target
    if land_target is None:
        # EDH heuristic: ~37% lands for a 100-card deck, clamped to [30, 40]
        land_target = max(30, min(40, round(tgt * 0.37)))
    land_target = min(tgt, max(0, int(land_target)))

    spells = df[~df["type"].astype(str).str.contains("Land", na=False)].copy()
    land_df = df[df["type"].astype(str).str.contains("Land", na=False)].copy()
    non_basic_land_df = land_df[~land_df["name"].isin(basic_land_names)]

    def _popularity(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["name", "deck_count"])
        if _HAS_INC_TABLE:
            pop_df = inclusion_table(frame)[["name", "count"]].rename(
                columns={"count": "deck_count"}
            )
        else:
            pop_df = (
                frame.groupby("name")["deck_id"]
                .nunique()
                .reset_index(name="deck_count")
            )
        return pop_df.sort_values(["deck_count", "name"], ascending=[False, True]).reset_index(
            drop=True
        )

    spell_pop = _popularity(spells)
    land_pop = _popularity(non_basic_land_df)

    deck: list[str] = []

    # 1) Pick non-basic lands up to the target (unique by name for variety)
    num_nonbasic_needed = min(land_target, len(land_pop))
    deck.extend(land_pop.head(num_nonbasic_needed)["name"].tolist())

    # 2) Fill remaining land slots with basics informed by commander colors
    basics_needed = land_target - num_nonbasic_needed
    if basics_needed > 0:
        color_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        basics_pool = [color_map[c] for c in commander_colors if c in color_map] or ["Wastes"]
        q, r = divmod(basics_needed, len(basics_pool))
        for idx, bname in enumerate(basics_pool):
            copies = q + (1 if idx < r else 0)
            deck.extend([bname] * copies)

    # 3) Fill the rest with the most popular spells (avoid duplicates until exhausted)
    spells_needed = tgt - len(deck)
    if spells_needed > 0 and not spell_pop.empty:
        used_names: set[str] = set(deck)
        for _, row in spell_pop.iterrows():
            if len(deck) >= tgt:
                break
            nm = row["name"]
            if nm in used_names:
                continue
            deck.append(nm)
            used_names.add(nm)

    # 4) Backfill with any remaining unique card names if still short
    if len(deck) < tgt:
        used_names = set(deck)
        remaining = (
            df[~df["name"].isin(used_names)]["name"].dropna().drop_duplicates().tolist()
        )
        for nm in remaining:
            if len(deck) >= tgt:
                break
            deck.append(nm)
            used_names.add(nm)

    # 5) Allow repeats of popular spells if the pool is very small
    if len(deck) < tgt and not spell_pop.empty:
        idx = 0
        while len(deck) < tgt:
            deck.append(spell_pop.iloc[idx % len(spell_pop)]["name"])
            idx += 1

    # 6) Final safety: pad with basics if the dataset is extremely sparse
    if len(deck) < tgt:
        color_map = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
        basics_pool = [color_map[c] for c in commander_colors if c in color_map] or ["Wastes"]
        idx = 0
        while len(deck) < tgt:
            deck.append(basics_pool[idx % len(basics_pool)])
            idx += 1

    # Final safety trim/pad
    return deck[:tgt]

def summarize_deck(
    deck: List[str],
    df_cards: pd.DataFrame,
    total_size: int | None = None,
) -> dict:
    """
    Build a summary of a generated deck for the dashboard, using the deck list
    as the single source of truth (one row per card instance).

    Parameters
    ----------
    deck : ordered list of card names selected for the deck (may contain duplicates).
    df_cards : card-level data frame used when generating the deck. Must contain
        at least `name`, `type`, `cmc`, `price_clean`, and optionally `category`.
    total_size : nominal deck size target (unused except for sanity checks).

    Returns
    -------
    Dict with the following keys:
        counts_by_type : DataFrame[type, count] derived strictly from deck_df rows
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
            work[c] = "" if c == "category" else pd.NA

    # Deduplicate metadata per card name and join against the full deck list (one row per card slot)
    meta = work[base_cols].dropna(subset=["name"]).drop_duplicates(subset=["name"], keep="first")
    deck_df = pd.DataFrame({"slot": range(1, len(deck) + 1), "name": deck}).merge(
        meta, on="name", how="left"
    )

    # Primary type extraction for consistent grouping
    deck_df["primary_type"] = deck_df["type"].apply(_extract_primary_type_local).fillna(
        "Unknown"
    )

    # --- Counts by primary type ---
    deck_df["summary_type"] = deck_df["primary_type"].where(
        deck_df["primary_type"].isin(_PRIMARY_TYPE_ORDER), "Other"
    )
    type_order_with_other = _PRIMARY_TYPE_ORDER + ["Other"]
    counts_by_type = (
        deck_df["summary_type"].value_counts().reindex(type_order_with_other, fill_value=0).reset_index()
    )
    counts_by_type.columns = ["type", "count"]

    # --- Price total (copies × price_clean, unknown price = 0) ---
    price_num = pd.to_numeric(deck_df["price_clean"], errors="coerce").fillna(0.0)
    price_total = float(price_num.sum())

    # --- Land breakdown: basics vs non-basics ---
    basic_names = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
    type_str = deck_df["type"].astype(str)
    is_land = deck_df["summary_type"].eq("Land") | type_str.str.contains("Land", na=False)
    is_basic = deck_df["name"].isin(basic_names) | type_str.str.contains("Basic Land", na=False)
    basics = int((is_land & is_basic).sum())
    non_basics = int((is_land & ~is_basic).sum())

    # --- CMC curve (spells only, weighted by copies) ---
    spells = deck_df[~is_land].copy()
    spells["cmc_num"] = pd.to_numeric(spells["cmc"], errors="coerce")
    spells = spells[spells["cmc_num"].notna()]
    spells["cmc_int"] = spells["cmc_num"].round().astype(int)
    cmc_curve = (
        spells.groupby("cmc_int")
        .size()
        .reset_index(name="count")
        .rename(columns={"cmc_int": "cmc"})
        .sort_values("cmc")
    )

    # --- Functional coverage from Tagger categories (if present) ---
    work_cat = deck_df.copy()
    work_cat["category"] = work_cat["category"].fillna("").astype(str)
    work_cat["category_list"] = work_cat["category"].str.split("|")

    rows: list[dict] = []
    for _, row in work_cat.iterrows():
        for cat in row["category_list"]:
            cat_clean = cat.strip()
            if not cat_clean or cat_clean == "Uncategorized":
                continue
            rows.append({"category": cat_clean})

    if rows:
        functions_covered = (
            pd.DataFrame(rows)
            .groupby("category")
            .size()
            .reset_index(name="count")
            .sort_values(["count", "category"], ascending=[False, True])
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