import streamlit as st
from core.cache import ensure_playwright
from core.config import DEFAULT_FUNCTIONAL_CATEGORIES
from data.scraping import scrape_deck_metadata
from data.decklists import fetch_decklists
from analysis.stats import inclusion_table, mana_curve, type_breakdown, cooccurrence_matrix, budget_filtered
import plotly.express as px

commander = "atraxa-praetors-voice"
df_decks = scrape_deck_metadata(commander)
df_cards = fetch_decklists(df_decks, max_decks=3)

st.subheader("Parsed Cards")
st.dataframe(df_cards, use_container_width=True, height=300)

# --- optional budget filter (per-card price cap) ---
with st.expander("Budget Filter", expanded=False):
    cap = st.number_input("Exclude cards above this price (keeps unknown prices)", min_value=0.0, value=0.0, step=0.5)
filtered = budget_filtered(df_cards, cap if cap and cap > 0 else None)

# --- Popularity / Inclusion ---
st.subheader("Card Popularity & Inclusion")
inc = inclusion_table(filtered)
st.dataframe(inc.head(50), use_container_width=True, height=420)
if not inc.empty:
    fig_pop = px.bar(inc.head(25), x="count", y="name", orientation="h",
                     hover_data=["inclusion_rate","avg_price","avg_cmc","type"],
                     title="Top 25 by # of Decks")
    fig_pop.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_pop, use_container_width=True)

# --- Mana Curve ---
st.subheader("Mana Curve (Spells Only)")
curve = mana_curve(filtered)
if not curve.empty:
    st.plotly_chart(px.bar(curve, x="cmc", y="count", title="Spell CMC Distribution"),
                    use_container_width=True)
else:
    st.info("No spell CMC data available.")

# --- Type Breakdown ---
st.subheader("Average Card Type Counts per Deck")
types_avg = type_breakdown(filtered)
if not types_avg.empty:
    st.plotly_chart(px.bar(types_avg, x="type", y="avg_count_per_deck", title="Average per Deck"),
                    use_container_width=True)
else:
    st.info("No type data available.")

# --- Co-occurrence Matrix ---
st.subheader("Card Co-occurrence (Top N)")
top_n = st.slider("Top N cards to include in co-occurrence matrix:", 10, 100, 40, step=5)
cooc = cooccurrence_matrix(filtered, top_n=top_n)
if not cooc.empty:
    # Heatmap-friendly long form
    cooc_long = cooc.stack().reset_index()
    cooc_long.columns = ["Card A", "Card B", "Co-occurs in # Decks"]
    fig_heat = px.density_heatmap(
        cooc_long, x="Card A", y="Card B", z="Co-occurs in # Decks",
        nbinsx=len(cooc.index), nbinsy=len(cooc.columns),
        histfunc="sum", title="Co-occurrence Heatmap (Deck Count)"
    )
    # keep squares readable
    fig_heat.update_layout(xaxis_nticks=min(50, len(cooc.columns)),
                           yaxis_nticks=min(50, len(cooc.index)))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Diagonal shows how many decks each card appears in (self-count).")
else:
    st.info("Co-occurrence matrix is empty (try increasing Top N).")