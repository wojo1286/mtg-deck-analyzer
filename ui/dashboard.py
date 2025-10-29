# ui/dashboard.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.stats import inclusion_table, mana_curve, type_breakdown, cooccurrence_matrix
from analysis.deckgen import (
    prepare_candidates,
    fill_deck_slots,
    generate_average_deck,
    summarize_deck,
)


def render_parsed(df: pd.DataFrame):
    st.subheader("Parsed Cards")
    if df is None or df.empty:
        st.info("No cards parsed yet.")
        return
    st.dataframe(df, use_container_width=True, height=300)


def render_popularity(
    df: pd.DataFrame,
    *,
    top_n: int = 25,
    price_cap: float | None = None,  # (cap already applied upstream, but kept for API compat)
    exclude_types: list[str] | None = None,
    exclude_top_n: int | None = None,
):
    st.subheader("Card Popularity & Inclusion")
    inc = inclusion_table(df, exclude_types=exclude_types, exclude_top_n=exclude_top_n)
    if inc.empty:
        st.info("No popularity data (check filters).")
        return
    st.dataframe(inc.head(50), use_container_width=True, height=420)
    fig = px.bar(
        inc.head(top_n),
        x="count",
        y="name",
        orientation="h",
        hover_data=["inclusion_rate", "avg_price", "avg_cmc", "type"],
        title=f"Top {min(top_n,len(inc))} by # of Decks",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


def render_curve(df: pd.DataFrame):
    st.subheader("Mana Curve (Spells Only)")
    curve = mana_curve(df)
    if curve.empty:
        st.info("No spell CMC data available.")
        return
    st.plotly_chart(
        px.bar(curve, x="cmc", y="count", title="Spell CMC Distribution"),
        use_container_width=True,
    )


def render_types(df: pd.DataFrame):
    st.subheader("Average Card Type Counts per Deck")
    types_avg = type_breakdown(df)
    if types_avg.empty:
        st.info("No type data available.")
        return
    st.plotly_chart(
        px.bar(types_avg, x="type", y="avg_count_per_deck", title="Average per Deck"),
        use_container_width=True,
    )


def render_cooccurrence(df, top_n=40):
    st.subheader("Card Co-occurrence (Top N)")

    colA, colB, colC = st.columns(3)
    min_decks = colA.slider(
        "Min # decks per card",
        1,
        10,
        2,
        help="Cards must appear in at least this many decks to be included.",
    )
    zero_diag = colB.checkbox("Hide diagonal", True, help="Hide a card co-occurring with itself.")
    use_norm = colC.checkbox(
        "Normalize (Jaccard similarity)",
        False,
        help="Shows association strength instead of raw counts.",
    )

    cooc = cooccurrence_matrix(
        df,
        top_n=top_n,
        min_decks=min_decks,
        normalize=use_norm,
        zero_diagonal=zero_diag,
    )

    if cooc.empty:
        st.info("Co-occurrence matrix is empty (increase Top N or lower the min-decks threshold).")
        return

    # Build long-form for the heatmap
    matrix = cooc  # already has index/columns named "Card A"/"Card B"
    cooc_long = (
        matrix.stack(dropna=False).rename("Co-occurs" if not use_norm else "Jaccard").reset_index()
    )

    title = (
        "Co-occurrence Heatmap (Deck Count)"
        if not use_norm
        else "Co-occurrence Heatmap (Jaccard Similarity)"
    )
    fig_heat = px.density_heatmap(
        cooc_long,
        x="Card A",
        y="Card B",
        z="Co-occurs" if not use_norm else "Jaccard",
        nbinsx=len(matrix.columns),
        nbinsy=len(matrix.index),
        histfunc="sum",
        title=title,
    )
    fig_heat.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig_heat.update_yaxes(tickfont=dict(size=10))
    fig_heat.update_traces(hovertemplate="A: %{x}<br>B: %{y}<br>Value: %{z}<extra></extra>")
    fig_heat.update_layout(
        xaxis_nticks=min(50, len(matrix.columns)),
        yaxis_nticks=min(50, len(matrix.index)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "Diagonal shows how many decks each card appears in (self-count). "
        "If hidden, diagonal is zeroed."
    )


def _default_type_targets():
    # sensible EDH defaults; adjust to taste
    # (min, max) per primary type
    return {
        "Creature": (20, 32),
        "Instant": (5, 12),
        "Sorcery": (4, 10),
        "Artifact": (6, 12),
        "Enchantment": (4, 10),
        "Planeswalker": (0, 4),
        # Land handled separately in the generator; we bias nonlands first
    }


def _default_function_targets():
    # Keep short & general. User can tune.
    return {
        "Ramp": (8, 12),
        "Card Advantage": (6, 10),
        "Removal": (6, 10),
        "Protection": (0, 6),
        "Recursion": (0, 5),
        "Counterspell": (0, 6),
        "Sweeper": (2, 6),
        "Tutor": (0, 4),
    }


def _two_int_sliders(col, label, low, high, default_min, default_max):
    m = col.number_input(f"{label} min", min_value=0, max_value=99, value=int(default_min), step=1)
    M = col.number_input(f"{label} max", min_value=m, max_value=99, value=int(default_max), step=1)
    return int(m), int(M)


def render_deck_generator(df: pd.DataFrame, *, commander_colors: list[str] | None = None):
    st.header("Deck Template Generator")

    if df is None or df.empty:
        st.info("Scrape some decks first to enable generation.")
        return

    with st.expander("Constraints & Options", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 1])

        total_size = c1.number_input("Deck size", 60, 200, 100, step=1)
        prefer_nonlands = c2.number_input("Prefer non-lands until (count)", 0, 99, 60, step=1)
        max_price_cap = c3.number_input(
            "Optional per-card price cap (0 = none)", 0.0, 9999.0, 0.0, step=0.5
        )

        # TYPE sliders
        st.markdown("**Type targets (min / max)**")
        tcols = st.columns(3)
        type_defaults = _default_type_targets()
        type_constraints = {}
        for i, (t, (mn, mx)) in enumerate(type_defaults.items()):
            col = tcols[i % 3]
            a, b = _two_int_sliders(col, t, 0, 99, mn, mx)
            type_constraints[t] = (a, b)

        # FUNCTION sliders (from defaults; you can also infer dynamically from df["category"])
        st.markdown("**Functional targets (min / max)**")
        fcols = st.columns(3)
        func_defaults = _default_function_targets()
        func_constraints = {}
        for i, (cat, (mn, mx)) in enumerate(func_defaults.items()):
            col = fcols[i % 3]
            a, b = _two_int_sliders(col, cat, 0, 99, mn, mx)
            func_constraints[cat] = (a, b)

        # Must-have / must-not-have quick inputs
        st.markdown("**Includes / Excludes (comma-separated card names)**")
        inc_text = st.text_input("Must-include", "")
        exc_text = st.text_input("Must-exclude", "")
        must_include = [s.strip() for s in inc_text.split(",") if s.strip()]
        must_exclude = [s.strip() for s in exc_text.split(",") if s.strip()]

    # Apply per-card price cap (optional) before candidate prep
    work = df.copy()
    if max_price_cap and max_price_cap > 0:
        mask_unknown = work["price_clean"].isna() | (work["price_clean"] == "")
        price_num = pd.to_numeric(work["price_clean"], errors="coerce")
        work = work[mask_unknown | (price_num <= max_price_cap)].copy()

    # Prepare candidate pool
    candidates = prepare_candidates(work, must_exclude=must_exclude, must_include=must_include)

    cA, cB = st.columns(2)
    go_avg = cA.button("✨ Generate Average Deck")
    go_constrained = cB.button("🧩 Generate Constrained Deck")

    deck: list[str] | None = None

    if go_avg:
        deck = generate_average_deck(work, total_size=total_size, commander_colors=commander_colors)

    if go_constrained:
        deck = fill_deck_slots(
            candidates,
            type_constraints=type_constraints,
            func_constraints=func_constraints,
            initial=must_include,
            total_size=total_size,
            prefer_nonlands_until=int(prefer_nonlands),
        )

    if deck is None:
        return

    st.success(f"Generated {len(deck)} cards.")
    # Show list & export
    out_df = pd.DataFrame({"#": np.arange(1, len(deck) + 1), "Card": deck})
    st.dataframe(out_df, use_container_width=True, height=420)

    # Download as text / csv
    st.download_button("Download .txt", "\n".join(deck), file_name="generated_deck.txt")
    st.download_button("Download .csv", out_df.to_csv(index=False), file_name="generated_deck.csv")

    # Summary
    st.subheader("Deck Summary")
    summary = summarize_deck(deck, work, total_size=total_size)
    if summary:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Counts by Type**")
            st.dataframe(summary["counts_by_type"], use_container_width=True, height=260)
            st.markdown(f"**Estimated Price Total:** ${summary['price_total']:.2f}")
            st.markdown(f"**Lands:** {summary['basics']} basic / {summary['non_basics']} non-basic")
        with cc2:
            st.markdown("**CMC Curve (Spells Only)**")
            curve = summary["cmc_curve"]
            if not curve.empty:
                fig = px.bar(curve, x="cmc", y="count", title="Spell CMC Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No spell CMC data available.")
        st.markdown("**Functions Covered**")
        st.dataframe(summary["functions_covered"], use_container_width=True, height=260)
