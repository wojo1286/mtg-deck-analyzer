"""Main Streamlit application entrypoint."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.stats import augment_with_basic_lands, budget_filtered, inclusion_table
from data.cleaning import clean_and_prepare_data
from data.decklists import fetch_decklists_shared
from data.scraping import scrape_deck_metadata
from data.staples import load_ci_staples
from data.tags import (
    has_gsheet_categories,
    load_card_tags,
    load_tags_from_gsheet,
    save_tags_to_gsheet,
    scrape_scryfall_tagger,
)
from ui.dashboard import (
    render_cooccurrence,
    render_curve,
    render_deck_generator,
    render_parsed,
    render_popularity,
    render_synergy_explorer,
    render_types,
)


st.set_page_config(
    page_title="MTG Deckbuilding Analysis Tool - Modular Version",
    layout="wide",
)
st.title("MTG Deckbuilding Analysis Tool - Modular Version")


def _parse_colors(key: str | None) -> list[str]:
    valid = {"W", "U", "B", "R", "G"}
    if not key:
        return []
    return [c for c in key.upper() if c in valid]


# --- Sidebar controls ----------------------------------------------------
commander_default = st.session_state.get("commander_slug", "")
num_decks_default = int(st.session_state.get("num_decks", 30))
ci_default = st.session_state.get("ci_key", "")
target_size_default = int(st.session_state.get("target_deck_size", 100))

with st.sidebar:
    st.header("Data Source")
    commander_slug = st.text_input(
        "Commander slug (EDHREC)",
        value=commander_default,
        help="Example: 'ojer-axonil-deepest-might'",
    )
    num_decks = st.slider(
        "How many decks to scrape",
        10,
        100,
        min(max(10, num_decks_default), 100),
        step=10,
    )
    ci_key = st.text_input(
        "Color identity key",
        value=ci_default,
        help="Used to look up EDHREC staples, e.g. 'B', 'UB', 'WUBRG'.",
    )
    target_deck_size = st.slider(
        "Target deck size",
        60,
        120,
        min(max(60, target_size_default), 120),
        step=5,
    )

    fetch_clicked = st.button("Fetch decks")
    clear_clicked = st.button("Clear data")

    st.divider()
    st.header("Budget Preset")
    preset = st.radio(
        "Price cap per card",
        ["No cap", "$1", "$5", "$10", "$25", "Custom…"],
        index=0,
    )
    if preset == "No cap":
        cap = None
    elif preset == "$1":
        cap = 1.0
    elif preset == "$5":
        cap = 5.0
    elif preset == "$10":
        cap = 10.0
    elif preset == "$25":
        cap = 25.0
    else:
        cap = st.number_input("Custom cap ($)", min_value=0.0, value=5.0, step=0.5)


if clear_clicked:
    for key in ("deck_data", "df_cards_raw", "filtered", "commander_slug", "num_decks"):
        st.session_state.pop(key, None)
    st.session_state.pop("ci_key", None)
    st.session_state.pop("target_deck_size", None)
    st.cache_data.clear()
    st.experimental_rerun()


if fetch_clicked:
    if not commander_slug:
        st.warning("Enter a commander slug before fetching decks.")
    else:
        st.session_state["commander_slug"] = commander_slug
        st.session_state["num_decks"] = int(num_decks)
        st.session_state["ci_key"] = ci_key
        st.session_state["target_deck_size"] = int(target_deck_size)

        with st.spinner(f"Fetching EDHREC deck metadata for commander: {commander_slug}"):
            deck_meta = scrape_deck_metadata(commander_slug, max_decks=int(num_decks))
        st.session_state["deck_data"] = deck_meta

        if deck_meta is None or deck_meta.empty:
            st.warning("No deck metadata returned. Try a different commander slug or check connectivity.")
        else:
            with st.spinner(f"Scraping up to {num_decks} decks and parsing card tables..."):
                cards_raw = fetch_decklists_shared(deck_meta, max_decks=int(num_decks))
            st.session_state["df_cards_raw"] = cards_raw
            if cards_raw is None or cards_raw.empty:
                st.warning(
                    "No cards were parsed from any deck. Try a different commander or increase the number of decks."
                )


deck_meta = st.session_state.get("deck_data")
df_cards_raw = st.session_state.get("df_cards_raw")

if df_cards_raw is None or df_cards_raw.empty:
    st.info("Click **Fetch decks** to scrape EDHREC data.")
    st.stop()

st.session_state["target_deck_size"] = int(target_deck_size)

# --- Tag loading ---------------------------------------------------------
gsheet_tags = load_tags_from_gsheet()
fallback_tags = load_card_tags()
if "tags_editor_df" not in st.session_state:
    base_tags = gsheet_tags if not gsheet_tags.empty else fallback_tags
    st.session_state["tags_editor_df"] = base_tags.copy()

active_tags = st.session_state.get("tags_editor_df")
if active_tags is None or active_tags.empty:
    active_tags = gsheet_tags if not gsheet_tags.empty else fallback_tags
    st.session_state["tags_editor_df"] = active_tags.copy()

# --- Cleaning and enrichment --------------------------------------------
df_cards, has_functional, num_decks_scraped, pop_all = clean_and_prepare_data(
    df_cards_raw, categories_df=st.session_state["tags_editor_df"]
)

if df_cards.empty:
    st.warning("No card rows were parsed after cleaning.")
    st.stop()

ci_staples = load_ci_staples(ci_key)
df_cards["is_ci_staple"] = df_cards["name"].isin(ci_staples)

popularity = inclusion_table(df_cards)
if popularity.empty:
    local_staples: set[str] = set()
else:
    local_staples = set(popularity.loc[popularity["inclusion_rate"] >= 60.0, "name"])

df_cards["is_local_staple"] = df_cards["name"].isin(local_staples)
df_cards["is_staple"] = df_cards["is_ci_staple"] | df_cards["is_local_staple"]

filtered = budget_filtered(df_cards, cap)
if filtered is None or filtered.empty:
    st.info("Your filters removed all cards. Try lowering the budget cap or scrape more decks.")
    st.stop()

st.session_state["filtered"] = filtered

commander_colors = _parse_colors(ci_key)

target_deck_size = int(st.session_state.get("target_deck_size", target_deck_size))
filtered_with_basics = augment_with_basic_lands(
    filtered, commander_colors=commander_colors, target_size=target_deck_size
)

# --- Advanced filters ----------------------------------------------------
with st.expander("Advanced Filters", expanded=False):
    unique_types = sorted([t for t in filtered["type"].dropna().unique().tolist() if t])
    exclude_types = st.multiselect("Exclude types", options=unique_types, default=[])
    exclude_top_n = st.slider("Exclude top N staples (by # of decks)", 0, 50, 0, step=5)

# --- Render sections -----------------------------------------------------
render_parsed(filtered)
render_popularity(
    filtered,
    top_n=25,
    price_cap=cap,
    exclude_types=exclude_types,
    exclude_top_n=exclude_top_n,
)
render_curve(filtered)
render_types(filtered_with_basics)

top_n_cooc = st.slider("Top N cards to include in co-occurrence matrix:", 10, 100, 40, step=5)
render_cooccurrence(filtered, top_n=top_n_cooc)

with st.expander("Card roles (raw tags)", expanded=False):
    tagged = filtered[["name", "category"]].drop_duplicates().sort_values("name")
    if tagged.empty:
        st.info("No tag data available. Add categories via the Tag Editor below.")
    else:
        st.dataframe(tagged, use_container_width=True)

sheet_available = has_gsheet_categories()
with st.expander("Tag Editor", expanded=False):
    st.markdown(
        "Edit card categories. Google Sheets is the primary source and CSV is used as a fallback."
    )
    editor_df = st.session_state["tags_editor_df"].copy()
    if "name" not in editor_df.columns:
        editor_df["name"] = ""
    if "category" not in editor_df.columns:
        editor_df["category"] = ""
    editor_df["name"] = editor_df["name"].astype(str)
    editor_df["category"] = editor_df["category"].fillna("").astype(str)

    edited = st.data_editor(
        editor_df.sort_values("name").reset_index(drop=True),
        num_rows="dynamic",
        use_container_width=True,
        key="tag_editor_table",
    )
    st.session_state["tags_editor_df"] = edited

    col_fetch, col_save = st.columns(2)

    fetch_disabled = not sheet_available or df_cards.empty
    if col_fetch.button("Fetch tags from Scryfall Tagger to Google Sheet", disabled=fetch_disabled):
        base_lookup = (
            edited.assign(category=edited["category"].astype(str).str.strip())
            .dropna(subset=["name"])
            .set_index("name")["category"]
        )
        missing_cards = [
            name
            for name in df_cards["name"].dropna().unique()
            if base_lookup.get(name, "").strip() == ""
        ]
        if not missing_cards:
            st.info("All scraped cards already have categories.")
        else:
            with st.spinner(f"Scraping Scryfall Tagger for {len(missing_cards)} cards..."):
                scraped = scrape_scryfall_tagger(missing_cards, junk_tags=None)
            if scraped.empty:
                st.warning("Scraping finished, but no new tags were found.")
            else:
                merged = pd.concat([edited, scraped], ignore_index=True)
                merged = (
                    merged.dropna(subset=["name"])
                    .drop_duplicates(subset=["name"], keep="last")
                    .sort_values("name")
                    .reset_index(drop=True)
                )
                try:
                    save_tags_to_gsheet(merged)
                    st.success("Google Sheet updated with scraped tags.")
                except Exception as exc:  # pragma: no cover - GSheets runtime
                    st.error(f"Failed to update Google Sheet: {exc}")
                st.session_state["tags_editor_df"] = merged
                st.experimental_rerun()

    if col_save.button("Save edited tags back to sheet", disabled=not sheet_available):
        try:
            save_tags_to_gsheet(edited.sort_values("name").reset_index(drop=True))
            st.success("Google Sheet updated successfully!")
        except Exception as exc:  # pragma: no cover - GSheets runtime
            st.error(f"Failed to update Google Sheet: {exc}")

    if not sheet_available:
        st.info("Google Sheets credentials not configured. Using local CSV fallback.")

staple_names = set(filtered.loc[filtered["is_staple"], "name"])
render_synergy_explorer(filtered, staples=staple_names)

render_deck_generator(
    filtered,
    commander_colors=commander_colors,
    df_for_average=filtered_with_basics,
    default_total_size=target_deck_size,
)
