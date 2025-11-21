# ===================================================================
# 1. SETUP & IMPORTS
# ===================================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import random
from copy import deepcopy
import time
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import subprocess
import sys
import urllib.parse


# --- Visualization ---
import plotly.express as px

# --- Advanced Analytics ---
# Removed mlxtend and sklearn imports

# --- Google Sheets Connection (THE CORRECT IMPORT) ---
from streamlit_gsheets import GSheetsConnection

# --- Page Config ---
st.set_page_config(layout="wide", page_title="MTG Deckbuilding Analysis Tool")


# ===================================================================
# 2. PLAYWRIGHT INSTALLATION (RUNS ONCE PER CONTAINER START)
# ===================================================================
@st.cache_resource
def setup_playwright():
    """
    Installs Playwright's browser binary. This is cached to run only once.
    """
    st.write("Verifying and (if needed) installing Playwright dependencies...")
    try:
        command = [sys.executable, "-m", "playwright", "install", "chromium"]
        with st.spinner("Installing Chromium browser (this may take a moment on the first run)..."):
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,  # Set a 10-minute timeout for the installation
            )
        st.success("Playwright environment is ready!")
        with st.expander("Show installation logs"):
            st.code(process.stdout)
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        st.error("Failed to install Playwright dependencies. The application cannot continue.")
        st.error(f"Error: {e}")
        st.code(e.stderr if hasattr(e, "stderr") else "No stderr output.")
        st.stop()
    return True


# This line executes the setup function.
setup_complete = setup_playwright()

# ===================================================================
# 3. DATA SCRAPING & PROCESSING FUNCTIONS
# ===================================================================

TYPE_KEYWORDS = [
    "Creature",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
    "Planeswalker",
    "Land",
    "Battle",
    "Tribal",
    "Conspiracy",
    "Phenomenon",
    "Plane",
    "Scheme",
    "Vanguard",
    "Dungeon" "Planeswalker",
    "Land",
    "Battle",
]


def _extract_primary_type(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return None

    lowered = cleaned.lower()
    for keyword in TYPE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword.lower())}\b", lowered):
            return keyword

    # Handle common separators like em dashes or slashes (e.g. "Creature — Elf")
    for separator in ("—", "-", "/"):
        if separator in cleaned:
            prefix = cleaned.split(separator, 1)[0].strip()
            if prefix:
                return _extract_primary_type(prefix)

    return None


def _extract_type_from_row(tr, tds, type_idx):
    attribute_keys = {
        "data-type-line",
        "data-typeline",
        "data-type",
        "data-card-type",
        "data-cardtype",
        "data-card-types",
        "data-cardtypes",
        "data-type_line",
    }
    type_hint_attrs = {
        "data-title",
        "title",
        "aria-label",
        "data-tooltip",
        "data-tooltip-content",
        "data-tooltip-title",
        "data-label",
        "data-th",
        "headers",
    }

    candidate_texts: list[str] = []

    if type_idx is not None and len(tds) > type_idx:
        candidate_texts.append(tds[type_idx].get_text(" ", strip=True))

    def _record_value(value):
        if not value:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _record_value(item)
        else:
            candidate_texts.append(str(value))

    for attr in attribute_keys:
        _record_value(tr.get(attr))

    def collect_from_tag(tag):
        if tag is None:
            return
        for attr, value in tag.attrs.items():
            attr_lower = attr.lower()
            if attr_lower in attribute_keys:
                _record_value(value)
            elif attr_lower in type_hint_attrs:
                joined = " ".join(value) if isinstance(value, (list, tuple, set)) else str(value)
                if "type" in joined.lower():
                    _record_value(joined)
                    text_value = tag.get_text(" ", strip=True)
                    if text_value:
                        candidate_texts.append(text_value)
        class_list = tag.get("class", [])
        if isinstance(class_list, str):
            class_list = [class_list]
        if any("type" in cls for cls in class_list):
            candidate_texts.append(tag.get_text(" ", strip=True))
        data_title = tag.get("data-title")
        if data_title and "type" in str(data_title).lower():
            candidate_texts.append(tag.get_text(" ", strip=True))

    for td in tds:
        collect_from_tag(td)
        for child in td.find_all(True):
            collect_from_tag(child)

    if tr:
        collect_from_tag(tr)

    for text in candidate_texts:
        ctype = _extract_primary_type(text)
        if ctype:
            return ctype

    row_text = tr.get_text(" ", strip=True) if tr else ""
    return _extract_primary_type(row_text)


def parse_table(html, deck_id, deck_source):
    soup = BeautifulSoup(html, "html.parser")
    cards = []
    for table in soup.find_all("table"):
        header_row = table.find("tr")
        header_cells = header_row.find_all(["th", "td"]) if header_row else []
        has_header = bool(header_row and header_row.find_all("th"))

        type_idx = price_idx = cmc_idx = None
        for idx, cell in enumerate(header_cells):
            header_text = cell.get_text(strip=True).lower()
            if "type" in header_text:
                type_idx = idx
            elif "price" in header_text or "card kingdom" in header_text:
                price_idx = idx
            elif "cmc" in header_text or "cost" in header_text:
                cmc_idx = idx

        rows = table.find_all("tr")
        data_rows = rows[1:] if has_header else rows

        if type_idx is None:
            st.warning(f"Could not find 'Type' column header when parsing deck {deck_id}.")

        for tr in data_rows:
            tds = tr.find_all("td")
            if not tds:
                continue

            name_el = tr.find("a")
            name = name_el.get_text(strip=True) if name_el else None

            cmc = None
            if cmc_idx is not None and len(tds) > cmc_idx:
                cmc = tds[cmc_idx].get_text(strip=True)
            else:
                cmc_el = tr.find("span", class_="float-right")
                cmc = cmc_el.get_text(strip=True) if cmc_el else None

            raw_type = None
            if type_idx is not None and len(tds) > type_idx:
                raw_type = tds[type_idx].get_text(strip=True)
            else:
                raw_type = next(
                    (
                        td.get_text(strip=True)
                        for td in tds
                        if _extract_primary_type(td.get_text(strip=True)) is not None
                    ),
                    None,
                )
            ctype = _extract_primary_type(raw_type)

            price = None
            if price_idx is not None and len(tds) > price_idx:
                price = tds[price_idx].get_text(strip=True)
            else:
                price = next(
                    (
                        td.get_text(strip=True)
                        for td in reversed(tds)
                        if td.get_text(strip=True).startswith("$")
                    ),
                    None,
                )

            if name:
                cards.append(
                    {
                        "deck_id": deck_id,
                        "deck_source": deck_source,
                        "cmc": cmc,
                        "name": name,
                        "type": ctype,
                        "price": price,
                    }
                )
    return cards


@st.cache_data
def get_commander_color_identity(commander_slug):
    """Fetches the commander's color identity from EDHREC's JSON endpoint."""
    try:
        url = f"https://json.edhrec.com/pages/commanders/{commander_slug}.json"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        data = response.json()
        return (
            data.get("container", {}).get("json_dict", {}).get("card", {}).get("color_identity", [])
        )
    except (requests.RequestException, KeyError, IndexError):
        return []


def run_scraper(
    commander_slug,
    deck_limit,
    bracket_slug="",
    budget_slug="",
    bracket_name="All Decks",
):
    st.info(f"🔍 Fetching deck metadata for '{commander_slug}' (Bracket: {bracket_name})...")
    base_url = f"https://json.edhrec.com/pages/decks/{commander_slug}"
    if bracket_slug:
        base_url += f"/{bracket_slug}"
    if budget_slug:
        base_url += f"/{budget_slug}"
    json_url = base_url + ".json"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(json_url, headers=headers)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch metadata. Error: {e}")
        return None, []

    color_identity = (
        data.get("container", {}).get("json_dict", {}).get("card", {}).get("color_identity", [])
    )
    if not color_identity:
        color_identity = get_commander_color_identity(commander_slug)
    st.session_state.commander_colors = color_identity

    decks = data.get("table", [])
    if not decks:
        st.error(f"No decks found for '{commander_slug}' in '{bracket_name}'.")
        return None, []

    df_meta = pd.json_normalize(decks).head(deck_limit)
    df_meta["deckpreview_url"] = df_meta["urlhash"].apply(
        lambda x: f"https://edhrec.com/deckpreview/{x}"
    )
    st.success(f"Found {len(decks)} decks. Scraping the first {len(df_meta)}.")

    all_cards = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page()
        for i, row in df_meta.iterrows():
            deck_id, deck_url = row["urlhash"], row["deckpreview_url"]
            status_text.text(f"[{i+1}/{len(df_meta)}] Fetching {deck_url}")
            try:
                page.goto(deck_url, timeout=90000)
                page.click('button[data-rr-ui-event-key="table"]')
                page.wait_for_selector("div[class*='TableView_table']", timeout=10000)
                type_header_selector = 'th:has-text("Type")'
                is_type_visible = page.is_visible(type_header_selector)
                if not is_type_visible:
                    try:
                        page.click('button:has-text("Edit Columns")', timeout=5000)
                        page.wait_for_selector('div[class*="dropdown-menu show"]', timeout=5000)
                        type_button_selector = (
                            'div[class*="dropdown-menu show"] button:has-text("Type")'
                        )
                        page.click(type_button_selector, timeout=5000)
                        page.wait_for_selector(type_header_selector, timeout=10000)
                    except Exception as e:
                        st.warning(
                            f"Could not enable 'Type' column for {deck_url}. Skipping. Error: {e}"
                        )
                        continue
                try:
                    page.wait_for_selector("table tbody tr td", timeout=15000)
                except Exception as e:
                    st.warning(
                        f"Data rows (<td> elements) did not appear for deck {deck_url}. Skipping. Error: {e}"
                    )
                    continue

                html = page.content()
                src_el = BeautifulSoup(html, "html.parser").find(
                    "a",
                    href=lambda x: x and any(d in x for d in ["moxfield", "archidekt"]),
                )
                deck_source = src_el["href"] if src_el else "Unknown"
                cards = parse_table(html, deck_id, deck_source)
                if cards:
                    all_cards.extend(cards)
                else:
                    st.warning(
                        f"No cards parsed for {deck_url}, though page loaded and filters applied."
                    )
            except Exception as e:
                status_text.text(f"⚠️ Skipping deck {deck_id} due to error: {e}")
            finally:
                time.sleep(random.uniform(0.5, 1.5))
            progress_bar.progress((i + 1) / len(df_meta))
        browser.close()

    if not all_cards:
        st.error("Scraping complete, but no cards were parsed.")
        return None, []
    st.success("✅ Scraping complete!")
    return pd.DataFrame(all_cards), color_identity


@st.cache_data
def clean_and_prepare_data(_df, _categories_df=None):
    dfc = _df.copy()
    dfc["price_clean"] = pd.to_numeric(
        dfc.get("price", "").astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )
    dfc["cmc"] = pd.to_numeric(dfc.get("cmc"), errors="coerce")
    dfc["type"] = dfc.get("type", "Unknown").fillna("Unknown")
    functional_analysis_enabled = False
    if _categories_df is not None and not _categories_df.empty:
        dfc = pd.merge(dfc, _categories_df, on="name", how="left")
        dfc["category"] = dfc["category"].fillna("Uncategorized")
        functional_analysis_enabled = True
    num_decks = dfc["deck_id"].nunique()
    pop_all = (
        dfc.groupby("name")
        .agg(count=("deck_id", "nunique"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    pop_all["inclusion_rate"] = (pop_all["count"] / num_decks) * 100
    return dfc, functional_analysis_enabled, num_decks, pop_all


@st.cache_data
def calculate_average_stats(_df, num_decks, active_func_categories=[]):
    if _df.empty or num_decks == 0:
        return {}
    stats = {}
    df_copy = _df.copy()
    df_copy["cmc_filled"] = pd.to_numeric(df_copy["cmc"], errors="coerce")
    primary_types = [
        "Creature",
        "Instant",
        "Sorcery",
        "Artifact",
        "Enchantment",
        "Land",
        "Planeswalker",
        "Battle",
    ]
    df_copy["primary_type"] = df_copy["type"].apply(
        lambda x: next((ptype for ptype in primary_types if ptype in str(x)), "Other")
    )
    deck_known_counts = df_copy.groupby("deck_id").size().reset_index(name="known_cards")
    deck_known_counts["inferred_basics"] = (100 - deck_known_counts["known_cards"]).clip(lower=0)
    avg_basic_count = deck_known_counts["inferred_basics"].mean()
    non_land_df = df_copy[df_copy["primary_type"] != "Land"].copy()
    cmc_valid_non_land_df = non_land_df.dropna(subset=["cmc_filled"])
    stats["avg_cmc_non_land"] = (
        cmc_valid_non_land_df["cmc_filled"].mean() if not cmc_valid_non_land_df.empty else 0
    )
    stats["median_cmc_non_land"] = (
        cmc_valid_non_land_df["cmc_filled"].median() if not cmc_valid_non_land_df.empty else 0
    )
    type_counts = df_copy.groupby("deck_id")["primary_type"].value_counts().unstack(fill_value=0)
    avg_type_counts = type_counts.mean()
    basic_land_names = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
    land_df = df_copy[df_copy["primary_type"] == "Land"]
    non_basic_land_df = land_df[~land_df["name"].isin(basic_land_names)]
    avg_non_basic_count = (
        non_basic_land_df.groupby("deck_id").size().mean() if not non_basic_land_df.empty else 0
    )
    avg_total_land_count = avg_non_basic_count + avg_basic_count
    stats["avg_total_lands"] = avg_total_land_count
    stats["avg_non_basic_lands"] = avg_non_basic_count
    stats["avg_basic_lands"] = max(0, avg_basic_count)
    avg_type_counts["Land"] = avg_total_land_count
    for basic in basic_land_names:
        if basic in avg_type_counts:
            del avg_type_counts[basic]
    stats["avg_type_counts"] = avg_type_counts.to_dict()
    total_cards_assumed = 100
    valid_counts = {t: max(0, c) for t, c in avg_type_counts.items()}
    total_avg_cards = sum(valid_counts.values())
    if total_avg_cards > 0:
        stats["avg_type_percentages"] = {
            t: (c / total_avg_cards) * 100 for t, c in valid_counts.items()
        }
    else:
        stats["avg_type_percentages"] = {t: 0 for t in valid_counts.keys()}
    stats["avg_functional_counts"] = {}
    if "category" in df_copy.columns and not df_copy["category"].isnull().all():
        df_copy["category_list"] = df_copy["category"].fillna("").astype(str).str.split("|")
        exploded_cats = df_copy.explode("category_list")
        exploded_cats = exploded_cats[
            (exploded_cats["category_list"] != "")
            & (exploded_cats["category_list"] != "Uncategorized")
            & pd.notna(exploded_cats["category_list"])
        ]
        category_deck_counts = (
            exploded_cats.groupby("deck_id")["category_list"].value_counts().unstack(fill_value=0)
        )
        avg_category_counts_all = category_deck_counts.mean()
        stats["avg_functional_counts"] = {
            cat: avg_category_counts_all.get(cat, 0) for cat in active_func_categories
        }
    if "price_clean" in df_copy.columns:
        df_copy["price_filled"] = pd.to_numeric(df_copy["price_clean"], errors="coerce").fillna(0)
        deck_total_prices = df_copy.groupby("deck_id")["price_filled"].sum()
        stats["avg_deck_price"] = deck_total_prices.mean() if not deck_total_prices.empty else 0
        stats["median_deck_price"] = (
            deck_total_prices.median() if not deck_total_prices.empty else 0
        )
        stats["min_deck_price"] = deck_total_prices.min() if not deck_total_prices.empty else 0
        stats["max_deck_price"] = deck_total_prices.max() if not deck_total_prices.empty else 0
    cmc_dist = cmc_valid_non_land_df["cmc_filled"].value_counts().sort_index()
    stats["cmc_distribution"] = cmc_dist.to_dict()
    return stats


def popularity_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["name", "count", "avg_price", "avg_cmc", "type"])
    return (
        frame.groupby("name")
        .agg(
            count=("deck_id", "nunique"),
            avg_price=("price_clean", "mean"),
            avg_cmc=("cmc", "mean"),
            type=("type", "first"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )


def parse_decklist(text: str) -> list:
    lines = text.strip().split("\n")
    return [re.sub(r"^\d+\s*x?\s*", "", line).strip() for line in lines if line.strip()]


# --- REMOVED build_cococcurrence FUNCTION ---


def _fill_deck_slots(
    candidates_df,
    constraints,
    initial_decklist=[],
    lands_df=pd.DataFrame(),
    color_identity=[],
):
    decklist, used_cards = list(initial_decklist), set(initial_decklist)
    basic_land_names = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
    initial_cards_info = candidates_df[
        candidates_df["name"].isin(initial_decklist)
    ].drop_duplicates(subset=["name"])
    if not initial_cards_info.empty:
        if (
            "category_list" not in initial_cards_info.columns
            and "category" in initial_cards_info.columns
        ):
            initial_cards_info["category_list"] = (
                initial_cards_info["category"].astype(str).str.split("|")
            )
        for _, card in initial_cards_info.iterrows():
            if "types" in constraints and card.get("type") in constraints["types"]:
                constraints["types"][card["type"]]["current"] += 1
            if "functions" in constraints and isinstance(card.get("category_list"), list):
                for func in card.get("category_list", []):
                    if func in constraints["functions"]:
                        constraints["functions"][func]["current"] += 1
    target_non_land_count = 63
    for _, card in candidates_df.iterrows():
        if len(decklist) >= target_non_land_count or card["name"] in used_cards:
            continue
        if card["name"] in basic_land_names:
            continue
        can_add = True
        card_type = card.get("type")
        card_cats = (
            card.get("category_list", []) if isinstance(card.get("category_list"), list) else []
        )
        if "types" in constraints and card_type in constraints["types"]:
            if (
                constraints["types"][card_type]["current"]
                >= constraints["types"][card_type]["target"][1]
            ):
                can_add = False
        if can_add and "functions" in constraints:
            violates_func = False
            for func in card_cats:
                if func in constraints["functions"]:
                    if (
                        constraints["functions"][func]["current"]
                        >= constraints["functions"][func]["target"][1]
                    ):
                        violates_func = True
                        break
            if violates_func:
                needed_for_min = False
                for func in card_cats:
                    if func in constraints["functions"]:
                        if (
                            constraints["functions"][func]["current"]
                            < constraints["functions"][func]["target"][0]
                        ):
                            needed_for_min = True
                            break
                if not needed_for_min:
                    can_add = False
        if can_add:
            decklist.append(card["name"])
            used_cards.add(card["name"])
            if "types" in constraints and card_type in constraints["types"]:
                constraints["types"][card_type]["current"] += 1
            if "functions" in constraints:
                best_func_to_increment = None
                max_need = -1
                for func in card_cats:
                    if func in constraints["functions"]:
                        current = constraints["functions"][func]["current"]
                        target_min = constraints["functions"][func]["target"][0]
                        target_max = constraints["functions"][func]["target"][1]
                        need_score = 0
                        if current < target_min:
                            need_score = (target_min - current) + 100
                        elif current < target_max:
                            need_score = target_max - current
                        else:
                            need_score = -1
                        if need_score > max_need:
                            max_need = need_score
                            best_func_to_increment = func
                if best_func_to_increment:
                    constraints["functions"][best_func_to_increment]["current"] += 1
    remaining_non_land = target_non_land_count - len(decklist)
    if remaining_non_land > 0:
        fillers = candidates_df[~candidates_df["name"].isin(used_cards)].head(remaining_non_land)
        decklist.extend(fillers["name"].tolist())
        used_cards.update(fillers["name"].tolist())
    needed_lands = 100 - len(decklist)
    num_non_basics_to_add = 0
    if not lands_df.empty and "name" in lands_df.columns:
        avg_non_basic_target = min(needed_lands, 28)
        pop_non_basics = popularity_table(lands_df)
        top_lands = pop_non_basics[~pop_non_basics["name"].isin(used_cards)].head(
            avg_non_basic_target
        )
        decklist.extend(top_lands["name"].tolist())
        used_cards.update(top_lands["name"].tolist())
        num_non_basics_to_add = len(top_lands)
    num_basics_to_add = needed_lands - num_non_basics_to_add
    if num_basics_to_add > 0:
        basic_land_map = {
            "W": "Plains",
            "U": "Island",
            "B": "Swamp",
            "R": "Mountain",
            "G": "Forest",
        }
        commander_basics = [basic_land_map[c] for c in color_identity if c in basic_land_map]
        if not commander_basics:
            commander_basics = ["Wastes"]
        num_colors = len(commander_basics)
        basics_per_color = num_basics_to_add // num_colors if num_colors > 0 else 0
        remainder = num_basics_to_add % num_colors if num_colors > 0 else num_basics_to_add
        for i, land_name in enumerate(commander_basics):
            count = basics_per_color + (1 if i < remainder else 0)
            decklist.extend([land_name] * count)
        if not commander_basics and remainder > 0:
            decklist.extend(["Wastes"] * remainder)
    if len(decklist) < 100:
        remaining_slots = 100 - len(decklist)
        fillers = candidates_df[~candidates_df["name"].isin(used_cards)].head(remaining_slots)
        decklist.extend(fillers["name"].tolist())
    return sorted(decklist[:100]), constraints


def generate_average_deck(df, commander_slug, color_identity):
    if df.empty:
        st.warning("Cannot generate average deck: No data available.")
        return None
    basic_land_names = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
    deck_counts = df.groupby("deck_id").size().reset_index(name="known_cards")
    deck_counts["inferred_basics"] = 100 - deck_counts["known_cards"]
    spell_df = df[~df["type"].str.contains("Land", na=False)]
    avg_spell_counts = (
        spell_df.groupby("deck_id")["type"].value_counts().unstack(fill_value=0).mean()
    )
    non_basic_land_df = df[(df["type"] == "Land") & (~df["name"].isin(basic_land_names))]
    avg_non_basic_count = (
        non_basic_land_df.groupby("deck_id").size().mean() if not non_basic_land_df.empty else 0
    )
    avg_basic_count = deck_counts["inferred_basics"].mean()
    template = avg_spell_counts.round().astype(int)
    template["Non-Basic Land"] = round(avg_non_basic_count)
    template["Basic Land"] = round(avg_basic_count)
    total_cards = template.sum()
    if total_cards == 0:
        st.warning("Cannot generate average deck: Calculated template is empty.")
        return None
    scaled_template = (template / total_cards * 99).round().astype(int)
    diff = 99 - scaled_template.sum()
    if diff != 0 and not scaled_template.empty:
        scaled_template[scaled_template.idxmax()] += diff
    commander_name = commander_slug.replace("-", " ").title()
    decklist = [commander_name]
    used_cards = {commander_name}
    for card_type, count in scaled_template.items():
        if count <= 0 or "Land" in card_type:
            continue
        candidates = df[df["type"] == card_type]
        pop_table = popularity_table(candidates)
        top_cards = pop_table[~pop_table["name"].isin(used_cards)].head(int(count))
        decklist.extend(top_cards["name"].tolist())
        used_cards.update(top_cards["name"].tolist())
    num_non_basics = int(scaled_template.get("Non-Basic Land", 0))
    if num_non_basics > 0:
        pop_non_basics = popularity_table(non_basic_land_df)
        top_lands = pop_non_basics[~pop_non_basics["name"].isin(used_cards)].head(num_non_basics)
        decklist.extend(top_lands["name"].tolist())
        used_cards.update(top_lands["name"].tolist())
    num_basics = int(scaled_template.get("Basic Land", 0))
    if num_basics > 0:
        basic_land_map = {
            "W": "Plains",
            "U": "Island",
            "B": "Swamp",
            "R": "Mountain",
            "G": "Forest",
        }
        commander_basics = [basic_land_map[c] for c in color_identity if c in basic_land_map]
        if not commander_basics:
            commander_basics = ["Wastes"]
        num_colors = len(commander_basics)
        basics_per_color = num_basics // num_colors if num_colors > 0 else 0
        remainder = num_basics % num_colors if num_colors > 0 else num_basics
        for i, land_name in enumerate(commander_basics):
            count = basics_per_color + (1 if i < remainder else 0)
            decklist.extend([land_name] * count)
        if not commander_basics and remainder > 0:
            decklist.extend(["Wastes"] * remainder)  # Add Wastes if colorless
    if len(decklist) < 100:
        remaining = 100 - len(decklist)
        all_pop = popularity_table(df)
        fillers = all_pop[~all_pop["name"].isin(used_cards)].head(remaining)
        decklist.extend(fillers["name"].tolist())
    return sorted(decklist)


# --- REMOVED import_edhrec_categories FUNCTION ---


@st.cache_data(ttl=2592000)
def scrape_scryfall_tagger(card_names: list, junk_tags_from_sheet: list):
    BASE_EXCLUDED_TAGS = {"abrade", "modal", "single english word name"}
    user_junk_tags = set(str(tag).lower() for tag in junk_tags_from_sheet)
    EXCLUDED_TAGS = BASE_EXCLUDED_TAGS.union(user_junk_tags)
    scraped_data = {}
    progress_bar = st.progress(0, text="Initializing Scryfall Tagger scrape...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page()
        for i, name in enumerate(card_names):
            progress_text = f"Scraping '{name}' ({i+1}/{len(card_names)})..."
            progress_bar.progress((i + 1) / len(card_names), text=progress_text)
            card_tags = set()
            try:
                encoded_name = urllib.parse.quote_plus(name)
                api_url = f"https://api.scryfall.com/cards/named?fuzzy={encoded_name}"
                response = requests.get(api_url)
                response.raise_for_status()
                card_data = response.json()
                set_code = card_data["set"]
                collector_num = card_data["collector_number"]
                tagger_url = f"https://tagger.scryfall.com/card/{set_code}/{collector_num}"
                page.goto(tagger_url, timeout=30000)
                page.wait_for_selector("a[href^='/tags/card/']", timeout=20000)
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")
                card_header = soup.find("h2", string=re.compile(r"^\s*Card\s*$"))
                if card_header:
                    tag_container = card_header.find_next_sibling("div")
                    if tag_container:
                        tags = tag_container.find_all("a", href=re.compile(r"^/tags/card/"))
                        for tag in tags:
                            tag_text = tag.get_text(strip=True)
                            if tag_text not in EXCLUDED_TAGS:
                                card_tags.add(tag_text.replace("-", " ").capitalize())
                if not card_tags:  # Fallback if specific header wasn't found
                    all_tags = soup.find_all("a", href=re.compile(r"^/tags/card/"))
                    for tag in all_tags:
                        tag_text = tag.get_text(strip=True)
                        if tag_text not in EXCLUDED_TAGS:
                            card_tags.add(tag_text.replace("-", " ").capitalize())
                if card_tags:
                    scraped_data[name] = sorted(list(card_tags))
                time.sleep(random.uniform(0.1, 0.25))
            except Exception as e:
                st.warning(f"Could not scrape '{name}'. (Error: {e})")
                continue
        browser.close()
    progress_bar.empty()
    if not scraped_data:
        st.error("Could not scrape any tags from the Scryfall Tagger.")
        return pd.DataFrame()
    final_data = [{"name": name, "category": "|".join(tags)} for name, tags in scraped_data.items()]
    return pd.DataFrame(final_data)


# ===================================================================
# 4. STREAMLIT UI & APP LOGIC
# ===================================================================


# (The `main` function code you provided is here)
def main():
    st.title("MTG Deckbuilding Analysis Tool")

    DEFAULT_FUNCTIONAL_CATEGORIES = [
        "Ramp",
        "Card Advantage",
        "Removal",
        "Sweeper",
        "Tutor",
        "Protection",
        "Recursion",
        "Counterspell",
    ]

    # --- Initialize Google Sheets Connection ---
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        st.session_state.gsheets_connected = True
    except Exception:
        st.session_state.gsheets_connected = False
        st.sidebar.warning("Google Sheets connection failed. Category Editor will be disabled.")

    df_raw = None

    # --- DATA SOURCE SELECTION ---
    st.sidebar.header("Deck Data Source")
    data_source_option = st.sidebar.radio(
        "Choose a data source:", ("Upload CSV", "Scrape New Data"), key="data_source"
    )

    commander_slug_for_tools = "ojer-axonil-deepest-might"
    new_data_loaded = False  # Flag to track if we need to clear constraints

    if data_source_option == "Upload CSV":
        decklist_file = st.sidebar.file_uploader("Upload Combined Decklists CSV", type=["csv"])
        if decklist_file:
            if (
                "last_uploaded_filename" not in st.session_state
                or st.session_state.last_uploaded_filename != decklist_file.name
            ):
                try:  # Add try-except for robustness
                    commander_slug_for_tools = decklist_file.name.split("_combined_decklists.csv")[
                        0
                    ]
                    st.session_state.commander_colors = get_commander_color_identity(
                        commander_slug_for_tools
                    )
                    df_raw = pd.read_csv(decklist_file)
                    st.session_state.scraped_df = df_raw
                    st.session_state.last_uploaded_filename = decklist_file.name
                    new_data_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing uploaded CSV: {e}")
                    df_raw = None  # Ensure df_raw is None on error
            else:
                df_raw = st.session_state.scraped_df

    elif data_source_option == "Scrape New Data":
        commander_slug = st.sidebar.text_input("Enter Commander Slug", "ojer-axonil-deepest-might")
        bracket_options = {
            "All Decks": "",
            "Budget": "budget",
            "Upgraded": "upgraded",
            "Optimized": "optimized",
            "cEDH": "cedh",
        }
        selected_bracket_name = st.sidebar.selectbox(
            "Select Bracket Level:", options=list(bracket_options.keys())
        )
        selected_bracket_slug = bracket_options[selected_bracket_name]
        deck_limit = st.sidebar.slider("Number of decks to scrape", 10, 200, 50)

        if st.sidebar.button("🚀 Start Scraping"):
            if (
                "last_scraped_slug" not in st.session_state
                or st.session_state.last_scraped_slug != commander_slug
                or "last_scraped_bracket" not in st.session_state
                or st.session_state.last_scraped_bracket != selected_bracket_slug
            ):
                with st.spinner("Scraping in progress... this may take several minutes."):
                    df_scraped, colors = run_scraper(
                        commander_slug,
                        deck_limit,
                        bracket_slug=selected_bracket_slug,
                        bracket_name=selected_bracket_name,
                    )
                    st.session_state.scraped_df = df_scraped
                    st.session_state.commander_colors = colors
                    st.session_state.last_scraped_slug = commander_slug
                    st.session_state.last_scraped_bracket = selected_bracket_slug
                    new_data_loaded = True
                    st.rerun()
            else:
                st.sidebar.info("Scrape parameters haven't changed. Using existing data.")
                if "scraped_df" in st.session_state:  # Load if scrape wasn't run but data exists
                    df_raw = st.session_state.scraped_df

    # Load data from session state if it exists (e.g., after rerun or if scrape wasn't needed)
    if (
        df_raw is None
        and "scraped_df" in st.session_state
        and st.session_state.scraped_df is not None
    ):
        df_raw = st.session_state.scraped_df
        if not new_data_loaded:
            st.sidebar.success("Scraped data is loaded.")
        # Determine commander slug based on how data was loaded
        if "last_scraped_slug" in st.session_state:
            commander_slug_for_tools = st.session_state.last_scraped_slug
        elif "last_uploaded_filename" in st.session_state:
            try:
                commander_slug_for_tools = st.session_state.last_uploaded_filename.split(
                    "_combined_decklists.csv"
                )[0]
            except:
                pass

    # --- Clear constraints if new data was loaded ---
    if new_data_loaded:
        st.sidebar.info("New data loaded. Clearing deck template constraints.")
        if "func_constraints" in st.session_state:
            del st.session_state["func_constraints"]
        if "type_constraints" in st.session_state:
            del st.session_state["type_constraints"]
        if "active_func_categories" in st.session_state:  # Also clear the active list
            del st.session_state["active_func_categories"]

    # --- RESET BUTTON ---
    st.sidebar.divider()
    if st.sidebar.button("🧹 Clear All Data & Reset"):
        keys_to_clear = [
            "scraped_df",
            "commander_colors",
            "master_categories",
            "junk_tags",
            "imported_tags",
            "func_constraints",
            "type_constraints",
            "active_func_categories",  # Added active_func_categories
            "last_uploaded_filename",
            "last_scraped_slug",
            "last_scraped_bracket",
        ]
        st.sidebar.write("Clearing session state keys...")
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.sidebar.write("Clearing application data cache...")
        st.cache_data.clear()
        # st.cache_resource.clear()

        st.success("All session data and caches cleared!")
        time.sleep(1)
        st.rerun()

    # ===============================================================
    # CARD CATEGORY DATA LOADING LOGIC
    # ===============================================================
    st.sidebar.header("Card Categories")

    # --- REMOVED EDHREC IMPORT BUTTON ---

    if "master_categories" not in st.session_state and st.session_state.gsheets_connected:
        with st.spinner("Loading your categories from Google Sheets..."):
            try:
                st.session_state.master_categories = conn.read(worksheet="Categories")
            except Exception as e:
                st.sidebar.error(f"Failed to load 'Categories' sheet: {e}")
                st.session_state.master_categories = pd.DataFrame(columns=["name", "category"])

    if "junk_tags" not in st.session_state and st.session_state.gsheets_connected:
        with st.spinner("Loading junk tag list..."):
            try:
                junk_df = conn.read(worksheet="JunkTags")
                if not junk_df.empty and "tag" in junk_df.columns:
                    st.session_state.junk_tags = junk_df["tag"].dropna().tolist()
                else:
                    st.session_state.junk_tags = []
            except Exception:
                st.sidebar.info("No 'JunkTags' worksheet found or it's empty. Using defaults.")
                st.session_state.junk_tags = []

    if st.session_state.gsheets_connected:
        if st.sidebar.button("Scrape Missing Tagger Categories 🔎"):
            if "master_categories" not in st.session_state:
                st.session_state.master_categories = pd.DataFrame(columns=["name", "category"])

            if not st.session_state.master_categories.empty:
                gsheet_df = st.session_state.master_categories.copy()
                if "category" not in gsheet_df.columns:
                    gsheet_df["category"] = ""
                gsheet_df["category"] = gsheet_df["category"].fillna("")
                cards_to_scrape_df = gsheet_df[gsheet_df["category"] == ""]

                if not cards_to_scrape_df.empty and "name" in cards_to_scrape_df.columns:
                    cards_to_scrape = cards_to_scrape_df["name"].dropna().unique().tolist()
                    st.sidebar.info(
                        f"Found {len(cards_to_scrape)} cards in GSheet without categories."
                    )
                else:
                    st.sidebar.info(
                        "All cards in GSheet already have categories or 'name' column missing. No scraping needed."
                    )
                    cards_to_scrape = []

                if cards_to_scrape:
                    junk_tags_list = st.session_state.get("junk_tags", [])
                    with st.spinner(
                        f"Scraping Scryfall Tagger for {len(cards_to_scrape)} cards..."
                    ):
                        tagger_df = scrape_scryfall_tagger(cards_to_scrape, junk_tags_list)

                    if not tagger_df.empty:
                        st.toast(f"Scraped tags for {len(tagger_df)} cards!", icon="✅")
                        st.sidebar.write("Merging scraped tags with GSheet data...")
                        master_df_copy = st.session_state.master_categories.copy()
                        if "category" not in master_df_copy.columns:
                            master_df_copy["category"] = ""
                        master_df_copy["category"] = master_df_copy["category"].fillna("")
                        master_df_copy.set_index("name", inplace=True)
                        tagger_df.set_index("name", inplace=True)
                        master_df_copy.update(tagger_df)
                        updated_master = master_df_copy.reset_index().sort_values("name")
                        cols = ["name", "category"] + [
                            col for col in updated_master.columns if col not in ["name", "category"]
                        ]
                        updated_master = updated_master[cols]

                        st.sidebar.write("Saving updated categories to Google Sheet...")
                        try:
                            conn.update(worksheet="Categories", data=updated_master)
                            st.session_state.master_categories = updated_master
                            st.sidebar.success("Google Sheet updated successfully!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.sidebar.error(f"Failed to update Google Sheet: {e}")
                    else:
                        st.sidebar.warning("Scraping finished, but no new tags were found.")
            else:
                st.sidebar.warning(
                    "Your 'Categories' GSheet is empty or not loaded. Cannot determine which cards to scrape."
                )

    # --- Robust category loading and merging ---
    categories_df_master = pd.DataFrame(columns=["name", "category"])
    imported_df = st.session_state.get("imported_tags", pd.DataFrame())
    gsheets_df = st.session_state.get("master_categories", pd.DataFrame())

    if (
        gsheets_df is None
        or not isinstance(gsheets_df, pd.DataFrame)
        or "name" not in gsheets_df.columns
    ):
        if gsheets_df is not None and not gsheets_df.empty:
            st.warning(
                "Your 'Categories' Google Sheet is missing the 'name' column or is invalid and will be ignored."
            )
        gsheets_df = pd.DataFrame(columns=["name", "category"])

    if (
        imported_df is None
        or not isinstance(imported_df, pd.DataFrame)
        or "name" not in imported_df.columns
    ):
        imported_df = pd.DataFrame(columns=["name", "category"])

    if not gsheets_df.empty or not imported_df.empty:
        merged_df = pd.merge(
            gsheets_df,
            imported_df,
            on="name",
            how="outer",
            suffixes=("_gsheet", "_imported"),
        )
        merged_df["category_gsheet"] = merged_df["category_gsheet"].fillna("")
        merged_df["category_imported"] = merged_df["category_imported"].fillna("")
        merged_df["category"] = np.where(
            merged_df["category_gsheet"] != "",
            merged_df["category_gsheet"],
            merged_df["category_imported"],
        )
        categories_df_master = (
            merged_df[["name", "category"]]
            .drop_duplicates(subset=["name"], keep="first")
            .sort_values("name")
            .reset_index(drop=True)
        )
        st.sidebar.info("Combined GSheet & Imported tags.")

    st.sidebar.divider()

    # --- MAIN APP DISPLAY LOGIC ---
    if df_raw is not None:
        df, FUNCTIONAL_ANALYSIS_ENABLED, NUM_DECKS, POP_ALL = clean_and_prepare_data(
            df_raw, categories_df_master
        )
        st.success(f"Data loaded with {NUM_DECKS} unique decks. Ready for analysis.")

        all_func_categories = []  # Define scope outside if block
        if FUNCTIONAL_ANALYSIS_ENABLED:
            if "active_func_categories" not in st.session_state:
                if "category" in df.columns:
                    all_individual_categories = df["category"].astype(str).str.split("|").explode()
                    all_func_categories = sorted(
                        [
                            cat
                            for cat in all_individual_categories.unique()
                            if pd.notna(cat) and cat not in ["Uncategorized", ""]
                        ]
                    )
                    st.session_state.active_func_categories = [
                        cat for cat in all_func_categories if cat in DEFAULT_FUNCTIONAL_CATEGORIES
                    ]
                else:
                    st.session_state.active_func_categories = []
                    all_func_categories = []
            else:
                if "category" in df.columns:
                    all_individual_categories = df["category"].astype(str).str.split("|").explode()
                    all_func_categories = sorted(
                        [
                            cat
                            for cat in all_individual_categories.unique()
                            if pd.notna(cat) and cat not in ["Uncategorized", ""]
                        ]
                    )
                else:
                    all_func_categories = []

        st.header("Dashboard & Analysis")
        col1, col2 = st.columns(2)
        with col1:
            price_cap = st.number_input("Price cap ($):", min_value=0.0, value=5.0, step=0.5)
            main_top_n = st.slider("Top N Staples:", 5, 100, 25, 5)
            exclude_top = st.checkbox("Exclude Top N Staples", False)
        with col2:
            unique_types_raw = df["type"].unique()
            unique_types = sorted([t for t in unique_types_raw if pd.notna(t) and t != "Unknown"])
            exclude_types = st.multiselect("Exclude Types:", options=unique_types, default=[])

        filtered_df = df.copy()
        if price_cap > 0:
            filtered_df = filtered_df[
                (filtered_df["price_clean"].isna()) | (filtered_df["price_clean"] <= price_cap)
            ]
        if exclude_top:
            filtered_df = filtered_df[
                ~filtered_df["name"].isin(POP_ALL["name"].head(main_top_n).tolist())
            ]
        if exclude_types:
            filtered_df = filtered_df[~filtered_df["type"].isin(exclude_types)]

        spells_df = filtered_df[~filtered_df["type"].str.contains("Land", na=False)]
        lands_df = filtered_df[filtered_df["type"].str.contains("Land", na=False)]

        st.subheader("Top Spells, Lands & Curves")
        c1, c2, c3 = st.columns(3)
        with c1:
            pop_spells = popularity_table(spells_df)
            if not pop_spells.empty:
                fig1 = px.bar(
                    pop_spells.head(25),
                    y="name",
                    x="count",
                    orientation="h",
                    title=f"Top {min(25, len(pop_spells))} Spells",
                )
                fig1.update_layout(yaxis=dict(autorange="reversed"), height=600)
                st.plotly_chart(fig1, use_container_width=True)
        with c2:
            pop_lands = popularity_table(lands_df)
            if not pop_lands.empty:
                fig_lands = px.bar(
                    pop_lands.head(25),
                    y="name",
                    x="count",
                    orientation="h",
                    title="Top 25 Lands",
                )
                fig_lands.update_layout(yaxis=dict(autorange="reversed"), height=600)
                st.plotly_chart(fig_lands, use_container_width=True)
        with c3:
            if "cmc" in spells_df.columns and pd.api.types.is_numeric_dtype(spells_df["cmc"]):
                curve = spells_df.groupby("cmc").size().reset_index(name="count")
                if not curve.empty:
                    fig2 = px.bar(curve, x="cmc", y="count", title="Mana Curve (Spells Only)")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.write("No spell data for mana curve.")
            else:
                st.write("CMC data missing or invalid for mana curve.")

        # --- Average Deck Statistics Section ---
        with st.expander("📊 Average Deck Statistics", expanded=True):
            if FUNCTIONAL_ANALYSIS_ENABLED:
                st.subheader("Functional Category Tracking")
                active_categories = st.session_state.get("active_func_categories", [])
                st.write("**Add/Remove Functional Categories to Track:**")
                if "all_func_categories" not in locals():
                    all_func_categories = []
                add_options = sorted(
                    [cat for cat in all_func_categories if cat not in active_categories]
                )
                col1_add, col2_add = st.columns([3, 1])
                new_func_to_add = col1_add.selectbox(
                    "Select category to add:",
                    options=add_options,
                    key="add_func_select_stats",
                    index=None,
                    placeholder="Choose function...",
                )
                if col2_add.button("Add Category", key="add_new_func_btn_stats"):
                    if new_func_to_add:
                        st.session_state.active_func_categories.append(new_func_to_add)
                        if "func_constraints" in st.session_state:
                            st.session_state.func_constraints[new_func_to_add] = (
                                (8, 12) if new_func_to_add in ["Ramp", "Card Advantage"] else (2, 8)
                            )
                        st.rerun()
                st.write("**Currently Tracked Categories:**")
                if not active_categories:
                    st.info("No functional categories are currently being tracked. Add some above.")
                else:
                    cols_per_row = 4
                    num_rows = (len(active_categories) + cols_per_row - 1) // cols_per_row
                    category_list_sorted = sorted(active_categories)
                    for i in range(num_rows):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            idx = i * cols_per_row + j
                            if idx < len(category_list_sorted):
                                func = category_list_sorted[idx]
                                with cols[j]:
                                    st.markdown(f"- **{func}**")
                                    if st.button(
                                        "Remove",
                                        key=f"del_active_func_stats_{func}",
                                        help=f"Stop tracking {func}",
                                    ):
                                        st.session_state.active_func_categories.remove(func)
                                        if (
                                            "func_constraints" in st.session_state
                                            and func in st.session_state.func_constraints
                                        ):
                                            del st.session_state.func_constraints[func]
                                        st.rerun()
                st.divider()

            active_categories_for_stats = st.session_state.get("active_func_categories", [])
            avg_stats = calculate_average_stats(df, NUM_DECKS, active_categories_for_stats)

            if avg_stats:
                st.subheader("Overall Averages")
                col_s1, col_s2, col_s3 = st.columns(3)
                avg_cmc_value = avg_stats.get("avg_cmc_non_land", 0)
                col_s1.metric("Avg. CMC (Non-Lands)", value=round(avg_cmc_value, 2))
                col_s2.metric("Avg. Total Lands", f"{avg_stats.get('avg_total_lands', 0):.1f}")
                col_s3.metric(
                    "Avg. Deck Price ($)",
                    (
                        f"${avg_stats.get('avg_deck_price', 0):.2f}"
                        if "avg_deck_price" in avg_stats
                        else "N/A"
                    ),
                )

                st.subheader("Average Functional Category Counts")
                func_counts_data = avg_stats.get("avg_functional_counts", {})
                if func_counts_data:
                    func_data = pd.DataFrame(
                        {
                            "Function": list(func_counts_data.keys()),
                            "Average Count": list(func_counts_data.values()),
                        }
                    ).sort_values("Average Count", ascending=False)
                    if not func_data.empty:
                        fig_func_dist = px.bar(
                            func_data,
                            x="Function",
                            y="Average Count",
                            title="Average Card Counts per Tracked Function",
                        )
                        st.plotly_chart(fig_func_dist, use_container_width=True)
                    else:
                        st.info("No data available for tracked functional categories.")
                else:
                    st.info("Functional category counts not calculated or none are tracked.")

                st.subheader("Average Card Type Distribution")
                type_data = pd.DataFrame(
                    {
                        "Type": list(avg_stats.get("avg_type_counts", {}).keys()),
                        "Average Count": list(avg_stats.get("avg_type_counts", {}).values()),
                    }
                ).sort_values("Average Count", ascending=False)
                if not type_data.empty:
                    fig_type_dist = px.bar(
                        type_data,
                        x="Type",
                        y="Average Count",
                        title="Average Card Counts per Type",
                    )
                    st.plotly_chart(fig_type_dist, use_container_width=True)

                st.subheader("Land Breakdown")
                st.write(
                    f"- Average Non-Basic Lands: {avg_stats.get('avg_non_basic_lands', 0):.1f}"
                )
                st.write(f"- Average Basic Lands: {avg_stats.get('avg_basic_lands', 0):.1f}")

                st.subheader("Mana Curve (Average Non-Land CMC Distribution)")
                cmc_dist_data = avg_stats.get("cmc_distribution", {})
                if cmc_dist_data:
                    cmc_df = pd.DataFrame(
                        list(cmc_dist_data.items()), columns=["CMC", "Count"]
                    ).sort_values("CMC")
                    cmc_df["Average Count per Deck"] = cmc_df["Count"] / NUM_DECKS
                    fig_cmc_dist = px.bar(
                        cmc_df,
                        x="CMC",
                        y="Average Count per Deck",
                        title="Average Non-Land Mana Curve",
                    )
                    st.plotly_chart(fig_cmc_dist, use_container_width=True)

                if "avg_deck_price" in avg_stats:
                    st.subheader("Deck Price Range")
                    st.write(f"- Minimum Price: ${avg_stats.get('min_deck_price', 0):.2f}")
                    st.write(f"- Median Price: ${avg_stats.get('median_deck_price', 0):.2f}")
                    st.write(f"- Average Price: ${avg_stats.get('avg_deck_price', 0):.2f}")
                    st.write(f"- Maximum Price: ${avg_stats.get('max_deck_price', 0):.2f}")
            else:
                st.info("Could not calculate average stats. Ensure data is loaded.")
        # --- END Average Deck Statistics ---

        with st.expander("Personal Deckbuilding Tools", expanded=True):
            st.subheader("Analyze Your Decklist")
            decklist_input = st.text_area(
                "Paste your decklist here:", height=200, key="deck_analyzer_input"
            )
            if st.button("🔍 Analyze My Deck"):
                user_decklist = parse_decklist(decklist_input)
                if user_decklist:
                    st.write(f"Analyzing {len(user_decklist)} cards...")
                    if (
                        not POP_ALL.empty
                        and "inclusion_rate" in POP_ALL.columns
                        and "name" in POP_ALL.columns
                    ):
                        staples = POP_ALL[POP_ALL["inclusion_rate"] >= 75]
                        missing_staples = staples[~staples["name"].isin(user_decklist)]
                        st.write("Popular Staples Missing From Your Deck (>75% inclusion):")
                        st.dataframe(missing_staples[["name", "inclusion_rate"]].round(1))
                    else:
                        st.warning("Popularity data unavailable for staple analysis.")
                else:
                    st.warning("Please paste a decklist to analyze.")

            st.subheader("Generate Average Deck")
            if "price_clean" in df.columns:
                deck_prices = df.groupby("deck_id")["price_clean"].sum()
                if not deck_prices.empty:
                    min_p, max_p = float(deck_prices.min()), float(deck_prices.max())
                    if min_p > max_p:
                        min_p = max_p
                    price_range = st.slider(
                        "Filter decks by Total Price for Average Deck:",
                        min_value=min_p,
                        max_value=max_p,
                        value=(min_p, max_p),
                    )
                    if st.button("📊 Generate Average Deck"):
                        with st.spinner("Generating average deck..."):
                            decks_in_range = deck_prices[
                                (deck_prices >= price_range[0]) & (deck_prices <= price_range[1])
                            ].index
                            filtered_price_df = df[df["deck_id"].isin(decks_in_range)]
                            avg_deck = generate_average_deck(
                                filtered_price_df,
                                commander_slug_for_tools,
                                st.session_state.get("commander_colors", []),
                            )
                            if avg_deck:
                                st.info(
                                    f"Detected Commander Color Identity: {', '.join(st.session_state.get('commander_colors', ['None']))}"
                                )
                                st.dataframe(pd.DataFrame(avg_deck, columns=["Card Name"]))
                else:
                    st.warning("No price data available to filter for average deck.")
            else:
                st.warning("Price data column ('price_clean') not found.")

            st.subheader("Generate a Deck Template")
            if FUNCTIONAL_ANALYSIS_ENABLED:
                st.write(
                    "First, build your list of constraints. Then, set their ranges and generate the deck inside the form below."
                )
                if "func_constraints" not in st.session_state:
                    st.session_state.func_constraints = {}
                if "type_constraints" not in st.session_state:
                    st.session_state.type_constraints = {}
                func_categories_list = []
                type_categories_list = []
                if "category" in df.columns:
                    all_individual_categories = df["category"].astype(str).str.split("|").explode()
                    func_categories_list = sorted(
                        [
                            cat
                            for cat in all_individual_categories.unique()
                            if pd.notna(cat) and cat not in ["Uncategorized", ""]
                        ]
                    )
                if "type" in df.columns:
                    type_categories_list_raw = df["type"].unique()
                    type_categories_list = sorted(
                        [t for t in type_categories_list_raw if pd.notna(t)]
                    )

                with st.expander("Step 2: Configure Card Type Constraints"):
                    available_types = [
                        t
                        for t in type_categories_list
                        if t not in st.session_state.type_constraints
                    ]
                    if available_types:
                        col1, col2 = st.columns([3, 1])
                        new_type = col1.selectbox(
                            "Add card type:",
                            options=available_types,
                            key="new_type_select",
                            index=None,
                            placeholder="Choose a type...",
                        )
                        if col2.button("Add Type", key="add_type_btn") and new_type:
                            st.session_state.type_constraints[new_type] = (
                                (25, 35) if new_type == "Creature" else (5, 15)
                            )
                            # --- REMOVED st.rerun() ---
                    elif type_categories_list:
                        st.info("All available card types have been added.")
                    if st.session_state.type_constraints:
                        st.write("**Configured Type Constraints:**")
                        for ctype in list(st.session_state.type_constraints.keys()):
                            col1_disp, col2_disp = st.columns([4, 1])
                            col1_disp.write(f"- **{ctype}**")
                            if col2_disp.button("Remove", key=f"del_type_{ctype}"):
                                del st.session_state.type_constraints[ctype]
                                st.rerun()  # Use st.rerun()

                with st.form(key="template_form"):
                    st.write("---")
                    st.write("**Step 3: Define Must-Haves, Exclusions, Ranges, and Generate**")
                    col_must, col_exclude = st.columns(2)
                    with col_must:
                        template_must_haves = st.text_area(
                            "Must-Include Cards (one per line):",
                            height=150,
                            key="template_must_haves",
                        )
                    with col_exclude:
                        template_must_excludes = st.text_area(
                            "Must-Exclude Cards (one per line):",
                            height=150,
                            key="template_must_excludes",
                        )
                    available_func_categories = sorted(
                        st.session_state.get("active_func_categories", [])
                    )
                    st.write("**Select Functional Constraints to Apply:**")
                    if not available_func_categories:
                        st.info(
                            "No functional categories are being tracked in the 'Average Deck Statistics' section."
                        )
                    if "selected_funcs_for_form" not in st.session_state:
                        st.session_state.selected_funcs_for_form = []
                    st.session_state.selected_funcs_for_form = st.multiselect(
                        "Choose functions to set ranges for:",
                        options=available_func_categories,
                        default=st.session_state.selected_funcs_for_form,
                        key="func_selector_form",
                    )

                    st.write("**Set Functional Constraint Ranges:**")
                    if not st.session_state.selected_funcs_for_form:
                        st.info("No functional categories selected above to set ranges for.")
                    for func in st.session_state.selected_funcs_for_form:
                        if func not in st.session_state.func_constraints:
                            st.session_state.func_constraints[func] = (5, 15)
                        value = st.session_state.func_constraints[func]
                        current_value = (
                            value
                            if isinstance(value, (list, tuple)) and len(value) == 2
                            else (5, 15)
                        )
                        c1, c2, c3 = st.columns([4, 1, 1])
                        with c1:
                            new_range = st.slider(
                                f"'{func}' count",
                                0,
                                40,
                                current_value,
                                key=f"slider_func_{func}",
                            )
                        with c2:
                            min_val = st.number_input(
                                f"{func} Min",
                                0,
                                40,
                                new_range[0],
                                key=f"num_min_func_{func}",
                                label_visibility="collapsed",
                            )
                        with c3:
                            max_val = st.number_input(
                                f"{func} Max",
                                0,
                                40,
                                new_range[1],
                                key=f"num_max_func_{func}",
                                label_visibility="collapsed",
                            )
                        if (min_val, max_val) != new_range:
                            st.session_state.func_constraints[func] = (min_val, max_val)
                            st.rerun()

                    st.write("**Set Card Type Constraint Ranges:**")
                    if not st.session_state.type_constraints:
                        st.info("No card type constraints configured in Step 2.")
                    for ctype, value in st.session_state.type_constraints.items():
                        current_value = (
                            value
                            if isinstance(value, (list, tuple)) and len(value) == 2
                            else (5, 15)
                        )
                        c1, c2, c3 = st.columns([4, 1, 1])
                        with c1:
                            new_range = st.slider(
                                f"'{ctype}' count",
                                0,
                                60,
                                current_value,
                                key=f"slider_type_{ctype}",
                            )
                        with c2:
                            min_val = st.number_input(
                                f"{ctype} Min",
                                0,
                                60,
                                new_range[0],
                                key=f"num_min_type_{ctype}",
                                label_visibility="collapsed",
                            )
                        with c3:
                            max_val = st.number_input(
                                f"{ctype} Max",
                                0,
                                60,
                                new_range[1],
                                key=f"num_max_type_{ctype}",
                                label_visibility="collapsed",
                            )
                        if (min_val, max_val) != new_range:
                            st.session_state.type_constraints[ctype] = (
                                min_val,
                                max_val,
                            )
                            st.rerun()

                    submitted = st.form_submit_button("📋 Generate Deck With Constraints")

                    if submitted:
                        if (
                            POP_ALL.empty
                            or "name" not in POP_ALL.columns
                            or "count" not in POP_ALL.columns
                        ):
                            st.error("Popularity data is missing or invalid.")
                        else:
                            constraints = {"functions": {}, "types": {}}
                            for func in st.session_state.selected_funcs_for_form:
                                if func in st.session_state.func_constraints:
                                    val_tuple = st.session_state.func_constraints[func]
                                    if isinstance(val_tuple, (list, tuple)) and len(val_tuple) == 2:
                                        constraints["functions"][func] = {
                                            "target": [val_tuple[0], val_tuple[1]],
                                            "current": 0,
                                        }
                                    else:
                                        st.warning(f"Invalid range for function '{func}'.")
                                        constraints["functions"][func] = {
                                            "target": [5, 15],
                                            "current": 0,
                                        }
                            for (
                                ctype,
                                val_tuple,
                            ) in st.session_state.type_constraints.items():
                                if isinstance(val_tuple, (list, tuple)) and len(val_tuple) == 2:
                                    constraints["types"][ctype] = {
                                        "target": [val_tuple[0], val_tuple[1]],
                                        "current": 0,
                                    }
                                else:
                                    st.warning(f"Invalid range for type '{ctype}'.")
                                    constraints["types"][ctype] = {
                                        "target": [5, 15],
                                        "current": 0,
                                    }
                            must_haves = parse_decklist(template_must_haves)
                            must_excludes = parse_decklist(template_must_excludes)
                            commander_colors = st.session_state.get("commander_colors", [])
                            with st.spinner("Generating decklists..."):
                                if (
                                    "name" in df.columns
                                    and "category" in df.columns
                                    and "cmc" in df.columns
                                ):
                                    base_candidates = (
                                        df[~df["name"].isin(must_excludes)]
                                        .drop_duplicates(subset=["name"])
                                        .copy()
                                        .merge(
                                            POP_ALL[["name", "count"]],
                                            on="name",
                                            how="left",
                                        )
                                    )
                                    base_candidates["count"] = base_candidates["count"].fillna(0)
                                    base_candidates["category_list"] = (
                                        base_candidates["category"].astype(str).str.split("|")
                                    )
                                    candidates_pop = base_candidates[
                                        ~base_candidates["name"].isin(must_haves)
                                    ].sort_values("count", ascending=False)
                                    candidates_eff = base_candidates[
                                        ~base_candidates["name"].isin(must_haves)
                                    ].copy()
                                    filtered_lands_df = (
                                        lands_df[~lands_df["name"].isin(must_excludes)]
                                        if "lands_df" in locals() and not lands_df.empty
                                        else pd.DataFrame()
                                    )
                                    median_cmc = (
                                        candidates_eff["cmc"].median()
                                        if not candidates_eff["cmc"].isnull().all()
                                        else 3
                                    )
                                    candidates_eff["cmc_filled"] = pd.to_numeric(
                                        candidates_eff["cmc"], errors="coerce"
                                    ).fillna(median_cmc)
                                    candidates_eff["efficiency_score"] = candidates_eff["count"] / (
                                        candidates_eff["cmc_filled"] + 1
                                    ).clip(lower=1)
                                    candidates_eff = candidates_eff.sort_values(
                                        "efficiency_score", ascending=False
                                    )
                                    pop_deck, _ = _fill_deck_slots(
                                        candidates_pop,
                                        deepcopy(constraints),
                                        initial_decklist=must_haves,
                                        lands_df=filtered_lands_df,
                                        color_identity=commander_colors,
                                    )
                                    eff_deck, _ = _fill_deck_slots(
                                        candidates_eff,
                                        deepcopy(constraints),
                                        initial_decklist=must_haves,
                                        lands_df=filtered_lands_df,
                                        color_identity=commander_colors,
                                    )
                                    pop_df = pd.DataFrame(pop_deck, columns=["Popularity Build"])
                                    eff_df = pd.DataFrame(eff_deck, columns=["Efficiency Build"])
                                    t1, t2 = st.tabs(["Popularity Build", "Efficiency Build"])
                                    with t1:
                                        st.dataframe(pop_df)
                                    with t2:
                                        st.dataframe(eff_df)
                                else:
                                    st.error(
                                        "Required columns ('name', 'category', 'cmc') not found in data."
                                    )
            else:  # Corresponds to if FUNCTIONAL_ANALYSIS_ENABLED
                st.warning(
                    "Import categories or connect to Google Sheets to enable the Deck Template Generator."
                )

            # --- Card Category Editor ---
            if st.session_state.gsheets_connected:
                with st.expander("Card Category Editor", expanded=False):
                    st.info("Edit categories below. Changes save to Google Sheet.")
                    if "name" in df.columns:
                        unique_cards_df = (
                            pd.DataFrame(df["name"].unique(), columns=["name"])
                            .sort_values("name")
                            .reset_index(drop=True)
                        )
                        if "name" in categories_df_master.columns:
                            editor_df = pd.merge(
                                unique_cards_df,
                                categories_df_master,
                                on="name",
                                how="left",
                            ).fillna("")
                        else:
                            st.warning("Master category data missing 'name'.")
                            editor_df = unique_cards_df.copy()
                            editor_df["category"] = ""
                        if not editor_df.empty:
                            edited_df = st.data_editor(
                                editor_df,
                                key="category_editor",
                                num_rows="dynamic",
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.warning("No unique card names.")
                            edited_df = pd.DataFrame(columns=["name", "category"])
                        if st.button("💾 Save Changes to Google Sheet"):
                            with st.spinner("Saving to Google Sheet..."):
                                if "master_categories" not in st.session_state or not isinstance(
                                    st.session_state.master_categories, pd.DataFrame
                                ):
                                    st.session_state.master_categories = pd.DataFrame(
                                        columns=["name", "category"]
                                    )
                                if "name" not in st.session_state.master_categories.columns:
                                    st.session_state.master_categories = pd.DataFrame(
                                        columns=["name", "category"]
                                    )
                                if edited_df is not None and "name" in edited_df.columns:
                                    updated_master = pd.concat(
                                        [
                                            st.session_state.master_categories[
                                                ~st.session_state.master_categories["name"].isin(
                                                    edited_df["name"]
                                                )
                                            ],
                                            edited_df,
                                        ]
                                    ).drop_duplicates(subset=["name"], keep="last")
                                    updated_master = updated_master[
                                        updated_master["name"].astype(str) != ""
                                    ].sort_values("name")
                                    if "category" not in updated_master.columns:
                                        updated_master["category"] = ""
                                    try:
                                        conn.update(worksheet="Categories", data=updated_master)
                                        st.session_state.master_categories = updated_master
                                        st.success("Categories saved!")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to save to Google Sheet: {e}")
                                else:
                                    st.error("Edited data invalid.")
                    else:
                        st.warning("Card data missing 'name'.")

            # --- Advanced Synergy Tools --- REMOVED ---

    else:  # Corresponds to if df_raw is not None
        st.info(
            "👋 Welcome! Please upload a CSV or scrape new data using the sidebar to get started."
        )


if __name__ == "__main__":
    if setup_complete:
        main()
