# app.py
import streamlit as st

from data.scraping import scrape_deck_metadata
from data.decklists import fetch_decklists_shared

from ui.dashboard import (
    render_parsed,
    render_popularity,
    render_curve,
    render_types,
    render_cooccurrence,
)

from analysis.stats import budget_filtered

from ui.dashboard import render_deck_generator

st.set_page_config(page_title="MTG Deckbuilding Analysis Tool - Modular Version", layout="wide")
st.title("MTG Deckbuilding Analysis Tool - Modular Version")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Data Source")
    commander = st.text_input("Commander slug (EDHREC)", value="atraxa-praetors-voice",
                              help="Example: 'ojer-axonil-deepest-might'")
    deck_limit = st.slider("How many decks to scrape", min_value=1, max_value=50, value=3, step=1)

    st.divider()
    st.header("Budget Preset")
    preset = st.radio(
        "Price cap per card",
        ["No cap", "$1", "$5", "$10", "$25", "Custom…"],
        index=0,
    )
    custom_cap = None
    if preset == "No cap":
        cap = None
    elif preset == "$1":
        cap = 1.0
    elif preset == "$5":
        cap = 5.0
        # fallthrough handled
    elif preset == "$10":
        cap = 10.0
    elif preset == "$25":
        cap = 25.0
    else:
        custom_cap = st.number_input("Custom cap ($)", min_value=0.0, value=5.0, step=0.5)
        cap = custom_cap

# --- Fetch deck metadata ---
with st.spinner(f"Fetching EDHREC deck metadata for commander: {commander}"):
    df_decks = scrape_deck_metadata(commander)

if df_decks is None or df_decks.empty:
    st.warning("No deck metadata returned. Try a different commander slug or check connectivity.")
    st.stop()

# --- Scrape deckpreview pages and parse cards (shared browser) ---
with st.spinner(f"Scraping up to {deck_limit} decks and parsing card tables..."):
    df_cards = fetch_decklists_shared(df_decks, max_decks=deck_limit)

if df_cards is None or df_cards.empty:
    st.warning("No cards were parsed from any deck. Try a different commander or increase the number of decks.")
    st.stop()


# --- Apply budget filter once; all charts use this 'filtered' ---
filtered = budget_filtered(df_cards, cap)

# --- Quick “nothing to show” nudge if filters removed everything ---
if filtered.empty:
    st.info("Your filters removed all cards. Try lowering the budget cap or scraping more decks.")
    st.stop()

# --- Popularity/type filters in one place ---
with st.expander("Advanced Filters", expanded=False):
    unique_types = sorted([t for t in filtered["type"].dropna().unique().tolist() if t])
    exclude_types = st.multiselect("Exclude types", options=unique_types, default=[])
    exclude_top_n = st.slider("Exclude top N staples (by # of decks)", 0, 50, 0, step=5)

# --- Render sections ---
render_parsed(filtered)
render_popularity(filtered, top_n=25, price_cap=cap,
                  exclude_types=exclude_types, exclude_top_n=exclude_top_n)
render_curve(filtered)
render_types(filtered)

top_n_cooc = st.slider("Top N cards to include in co-occurrence matrix:", 10, 100, 40, step=5)
render_cooccurrence(filtered, top_n=top_n_cooc)

from analysis.deckgen import (
    prepare_candidates,
    fill_deck_slots,
    generate_average_deck,
    summarize_deck,
)

# Example: build candidates from your scraped df (after any filters)
cands = prepare_candidates(df_cards, must_exclude=[], must_include=[])

# Example constraints
type_constraints = {"Creature": (20, 32), "Instant": (5, 12), "Sorcery": (4, 10)}
func_constraints = {"Ramp": (8, 12), "Card Advantage": (6, 10), "Removal": (6, 10)}

generated = fill_deck_slots(
    cands,
    type_constraints=type_constraints,
    func_constraints=func_constraints,
    initial=[],               # your must-haves list
    total_size=100,
    prefer_nonlands_until=60  # optional early spell bias
)

summary = summarize_deck(generated, df_cards)

# --- Deck Generator Panel ---
# If you know the commander color identity, pass it here; otherwise leave None
commander_colors = None  # e.g., ['W','U','B','G'] for Atraxa
render_deck_generator(filtered, commander_colors=commander_colors)