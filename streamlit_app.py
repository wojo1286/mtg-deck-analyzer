# ===================================================================
# 1. SETUP & IMPORTS
# ===================================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import warnings
import random
from copy import deepcopy
import time
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import subprocess
import sys
import urllib.parse

import json
from pathlib import Path

# --- Visualization ---
import plotly.express as px
import plotly.graph_objects as go

# --- Advanced Analytics ---
from mlxtend.frequent_patterns import apriori
from sklearn.manifold import TSNE

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
                timeout=600  # Set a 10-minute timeout for the installation
            )
        st.success("Playwright environment is ready!")
        with st.expander("Show installation logs"):
            st.code(process.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        st.error("Failed to install Playwright dependencies. The application cannot continue.")
        st.error(f"Error: {e}")
        st.code(e.stderr if hasattr(e, 'stderr') else "No stderr output.")
        st.stop()
    return True

# This line executes the setup function.
setup_complete = setup_playwright()

# ===================================================================
# 3. DATA SCRAPING & PROCESSING FUNCTIONS
# ===================================================================

TYPE_KEYWORDS = [
    "Creature", "Instant", "Sorcery", "Artifact", "Enchantment",
    "Planeswalker", "Land", "Battle", "Tribal", "Conspiracy",
    "Phenomenon", "Plane", "Scheme", "Vanguard", "Dungeon"
    "Planeswalker", "Land", "Battle"
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

    # Final fallback: inspect the whole row text for any recognizable keyword.
    row_text = tr.get_text(" ", strip=True) if tr else ""
    return _extract_primary_type(row_text)
    cleaned = text.strip()
    lowered = cleaned.lower()
    for keyword in TYPE_KEYWORDS:
        if keyword.lower() in lowered:
            return keyword
    return cleaned if cleaned else None


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
            elif "price" in header_text:
                price_idx = idx
            elif "cmc" in header_text or "cost" in header_text:
                cmc_idx = idx

        rows = table.find_all("tr")
        data_rows = rows[1:] if has_header else rows

        for tr in data_rows:
        for tr in table.find_all("tr")[1:]:
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

            ctype = _extract_type_from_row(tr, tds, type_idx)
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
                    (td.get_text(strip=True) for td in reversed(tds) if td.get_text(strip=True).startswith("$")),
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
        return data.get("container", {}).get("json_dict", {}).get("card", {}).get("color_identity", [])
    except (requests.RequestException, KeyError, IndexError):
        return []

def run_scraper(commander_slug, deck_limit, bracket_slug="", budget_slug="", bracket_name="All Decks"):
    st.info(f"🔍 Fetching deck metadata for '{commander_slug}' (Bracket: {bracket_name})...")
    
    base_url = f"https://json.edhrec.com/pages/decks/{commander_slug}"
    if bracket_slug: base_url += f"/{bracket_slug}"
    if budget_slug: base_url += f"/{budget_slug}"
    json_url = base_url + ".json"

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(json_url, headers=headers); r.raise_for_status(); data = r.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch metadata. Error: {e}"); return None, []

    color_identity = data.get("container", {}).get("json_dict", {}).get("card", {}).get("color_identity", [])
    if not color_identity: color_identity = get_commander_color_identity(commander_slug)
    st.session_state.commander_colors = color_identity

    decks = data.get("table", [])
    if not decks:
        st.error(f"No decks found for '{commander_slug}' in '{bracket_name}'."); return None, []

    df_meta = pd.json_normalize(decks).head(deck_limit)
    df_meta["deckpreview_url"] = df_meta["urlhash"].apply(lambda x: f"https://edhrec.com/deckpreview/{x}")
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
                
                page.wait_for_selector("table tbody tr", timeout=20000)
                page.wait_for_timeout(500)

                html = page.content()
                src_el = BeautifulSoup(html, "html.parser").find("a", href=lambda x: x and any(d in x for d in ["moxfield", "archidekt"]))
                deck_source = src_el["href"] if src_el else "Unknown"
                cards = parse_table(html, deck_id, deck_source)

                if cards: 
                    all_cards.extend(cards)
                else:
                    st.warning(f"No cards parsed for {deck_url}, though page loaded.")
                        
                time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                status_text.text(f"⚠️ Skipping deck {deck_id} due to error: {e}")
            progress_bar.progress((i + 1) / len(df_meta))
        browser.close()
    
    if not all_cards: 
        st.error("Scraping complete, but no cards were parsed."); return None, []
    st.success("✅ Scraping complete!"); return pd.DataFrame(all_cards), color_identity


@st.cache_data
def clean_and_prepare_data(_df, _categories_df=None):
    dfc = _df.copy()
    dfc['price_clean'] = pd.to_numeric(dfc.get('price', '').astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    dfc['cmc'] = pd.to_numeric(dfc.get('cmc'), errors='coerce')
    dfc['type'] = dfc.get('type', 'Unknown').fillna('Unknown')
    
    functional_analysis_enabled = False
    if _categories_df is not None and not _categories_df.empty:
        dfc = pd.merge(dfc, _categories_df, on='name', how='left')
        dfc['category'] = dfc['category'].fillna('Uncategorized')
        functional_analysis_enabled = True
    
    num_decks = dfc['deck_id'].nunique()
    pop_all = (dfc.groupby('name').agg(count=('deck_id','nunique')).reset_index().sort_values('count', ascending=False))
    pop_all['inclusion_rate'] = (pop_all['count'] / num_decks) * 100
    
    return dfc, functional_analysis_enabled, num_decks, pop_all

def popularity_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty: return pd.DataFrame(columns=['name', 'count', 'avg_price', 'avg_cmc', 'type'])
    return (frame.groupby('name')
            .agg(count=('deck_id','nunique'), avg_price=('price_clean','mean'), avg_cmc=('cmc','mean'), type=('type','first'))
            .reset_index().sort_values('count', ascending=False))

def parse_decklist(text: str) -> list:
    lines = text.strip().split('\n')
    return [re.sub(r'^\d+\s*x?\s*', '', line).strip() for line in lines if line.strip()]

def build_cococcurrence(source_df: pd.DataFrame, topN: int, exclude_staples_n: int, pop_all_df: pd.DataFrame) -> pd.DataFrame:
    working_df = source_df.copy()
    if exclude_staples_n > 0:
        staples_to_exclude = pop_all_df['name'].head(exclude_staples_n).tolist()
        working_df = working_df[~working_df['name'].isin(staples_to_exclude)]
    top_cards = working_df['name'].value_counts().head(topN).index
    f = working_df[working_df['name'].isin(top_cards)]
    if f.empty or 'deck_id' not in f: return pd.DataFrame()
    pivot = pd.crosstab(f['deck_id'], f['name']).reindex(columns=top_cards, fill_value=0)
    return pivot.T.dot(pivot)

def _fill_deck_slots(candidates_df, constraints, initial_decklist=[]):
    decklist, used_cards = list(initial_decklist), set(initial_decklist)
    
    initial_df = candidates_df[candidates_df['name'].isin(initial_decklist)].drop_duplicates(subset=['name'])
    for _, card in initial_df.iterrows():
        if card['type'] in constraints['types']: constraints['types'][card['type']]['current'] += 1
        if isinstance(card.get('category_list'), list):
            for func in card['category_list']:
                if func in constraints['functions']: 
                    constraints['functions'][func]['current'] += 1

    for _, card in candidates_df.iterrows():
        if len(decklist) >= 100 or card['name'] in used_cards: continue
        can_add = True
        card_type = card['type']
        if card_type in constraints['types'] and constraints['types'][card_type]['current'] >= constraints['types'][card_type]['target'][1]: can_add = False
        if can_add and isinstance(card.get('category_list'), list):
            for func in card['category_list']:
                if func in constraints['functions'] and constraints['functions'][func]['current'] >= constraints['functions'][func]['target'][1]: can_add = False; break
        if can_add:
            decklist.append(card['name']); used_cards.add(card['name'])
            if card_type in constraints['types']: constraints['types'][card_type]['current'] += 1
            if isinstance(card.get('category_list'), list):
                best_func, max_need = '', -1
                for func in card['category_list']:
                    if func in constraints['functions']:
                       need = constraints['functions'][func]['target'][1] - constraints['functions'][func]['current']
                       if need > max_need: max_need = need; best_func = func
                if best_func: constraints['functions'][best_func]['current'] += 1
    remaining = 100 - len(decklist)
    if remaining > 0:
        fillers = candidates_df[~candidates_df['name'].isin(used_cards)].head(remaining)
        decklist.extend(fillers['name'].tolist())
    return decklist, constraints

def generate_average_deck(df, commander_slug, color_identity):
    if df.empty:
        st.warning("Cannot generate average deck: No data available."); return None

    basic_land_names = ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes']
    
    deck_counts = df.groupby('deck_id').size().reset_index(name='known_cards')
    deck_counts['inferred_basics'] = 100 - deck_counts['known_cards']

    spell_df = df[~df['type'].str.contains('Land', na=False)]
    avg_spell_counts = spell_df.groupby('deck_id')['type'].value_counts().unstack(fill_value=0).mean()
    
    non_basic_land_df = df[(df['type'] == 'Land') & (~df['name'].isin(basic_land_names))]
    avg_non_basic_count = non_basic_land_df.groupby('deck_id').size().mean() if not non_basic_land_df.empty else 0
    avg_basic_count = deck_counts['inferred_basics'].mean()

    template = avg_spell_counts.round().astype(int)
    template['Non-Basic Land'] = round(avg_non_basic_count)
    template['Basic Land'] = round(avg_basic_count)

    total_cards = template.sum()
    if total_cards == 0:
        st.warning("Cannot generate average deck: Calculated template is empty."); return None
    
    scaled_template = (template / total_cards * 99).round().astype(int)
    diff = 99 - scaled_template.sum()
    if diff != 0 and not scaled_template.empty: scaled_template[scaled_template.idxmax()] += diff

    commander_name = commander_slug.replace('-', ' ').title()
    decklist = [commander_name]
    used_cards = {commander_name}

    for card_type, count in scaled_template.items():
        if count <= 0 or 'Land' in card_type: continue
        candidates = df[df['type'] == card_type]
        pop_table = popularity_table(candidates)
        top_cards = pop_table[~pop_table['name'].isin(used_cards)].head(int(count))
        decklist.extend(top_cards['name'].tolist())
        used_cards.update(top_cards['name'].tolist())
    
    num_non_basics = int(scaled_template.get('Non-Basic Land', 0))
    if num_non_basics > 0:
        pop_non_basics = popularity_table(non_basic_land_df)
        top_lands = pop_non_basics[~pop_non_basics['name'].isin(used_cards)].head(num_non_basics)
        decklist.extend(top_lands['name'].tolist())
        used_cards.update(top_lands['name'].tolist())

    num_basics = int(scaled_template.get('Basic Land', 0))
    if num_basics > 0:
        basic_land_map = {'W': 'Plains', 'U': 'Island', 'B': 'Swamp', 'R': 'Mountain', 'G': 'Forest'}
        commander_basics = [basic_land_map[c] for c in color_identity if c in basic_land_map]
        if not commander_basics: commander_basics = ['Wastes']
        
        basics_per_color = num_basics // len(commander_basics)
        remainder = num_basics % len(commander_basics)
        for i, land_name in enumerate(commander_basics):
            count = basics_per_color + (1 if i < remainder else 0)
            decklist.extend([land_name] * count)

    if len(decklist) < 100:
        remaining = 100 - len(decklist)
        all_pop = popularity_table(df)
        fillers = all_pop[~all_pop['name'].isin(used_cards)].head(remaining)
        decklist.extend(fillers['name'].tolist())
        
    return sorted(decklist)
@st.cache_data(ttl=604800) # Cache for 7 days
def import_edhrec_categories():
    """
    Builds a functional category list by fetching card recommendations
    from several popular and color-diverse commanders on EDHREC.
    THIS IS A PURE DATA FUNCTION and is safe to cache.
    """
    commander_slugs = [
        "atraxa-praetors-voice", "kenrith-the-returned-king", "korvold-fae-cursed-king",
        "chulane-teller-of-tales", "prosper-tome-bound", "yuriko-the-tigers-shadow",
        "meren-of-clan-nel-toth", "krenko-mob-boss", "urza-lord-high-artificer",
        "the-ur-dragon", "sythis-harvests-hand", "light-paws-emperors-voice",
        "tatyova-benthic-druid", "tergrid-god-of-fright", "ojer-axonil-deepest-might", "zedruu-the-greathearted"
    ]
    excluded_categories = ["high synergy cards", "top cards", "new cards", "utility lands"]
    all_card_tags = {}

    for slug in commander_slugs:
        try:
            url = f"https://json.edhrec.com/pages/commanders/{slug}.json"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            data = response.json()
            
            cardlists = data.get("container", {}).get("json_dict", {}).get("cardlists", [])
            if cardlists:
                for category_group in cardlists:
                    category_name = category_group.get('header', '').lower().strip()
                    if category_name and category_name not in excluded_categories:
                        clean_name = ' '.join(word.capitalize() for word in category_name.split())
                        if "Removal" in clean_name or "Wipe" in clean_name: 
                            clean_name = "Removal" 

                        for card in category_group.get('cardviews', []):
                            card_name = card.get('name')
                            if card_name:
                                if card_name not in all_card_tags:
                                    all_card_tags[card_name] = set()
                                all_card_tags[card_name].add(clean_name)
            time.sleep(0.25) # Be polite to the API
        except (requests.RequestException, json.JSONDecodeError):
            # We just skip failures in the cached function
            continue
            
    if not all_card_tags:
        return pd.DataFrame()

    final_data = [{"name": name, "category": "|".join(sorted(list(tags)))} for name, tags in all_card_tags.items()]
    return pd.DataFrame(final_data)

@st.cache_data(ttl=2592000) # Cache scraped tag data for 30 days
def scrape_scryfall_tagger(card_names: list, junk_tags_from_sheet: list):
    """
    Scrapes the Scryfall Tagger page for a given list of unique card names.
    Filters the results based on a user-provided list of junk tags.
    """
    BASE_EXCLUDED_TAGS = { 'abrade', 'modal', 'single english word name' }
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
                
                set_code = card_data['set']
                collector_num = card_data['collector_number']
                tagger_url = f"https://tagger.scryfall.com/card/{set_code}/{collector_num}"

                page.goto(tagger_url, timeout=30000)
                page.wait_for_selector("a[href^='/tags/card/']", timeout=20000)
                
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                card_header = soup.find('h2', string=re.compile(r'^\s*Card\s*$'))
                
                if card_header:
                    tag_container = card_header.find_next_sibling('div')
                    if tag_container:
                        tags = tag_container.find_all('a', href=re.compile(r'^/tags/card/'))
                        for tag in tags:
                            tag_text = tag.get_text(strip=True)
                            if tag_text not in EXCLUDED_TAGS:
                                card_tags.add(tag_text.replace('-', ' ').capitalize())
                
                if not card_tags:
                    all_tags = soup.find_all('a', href=re.compile(r'^/tags/card/'))
                    for tag in all_tags:
                        tag_text = tag.get_text(strip=True)
                        if tag_text not in EXCLUDED_TAGS:
                            card_tags.add(tag_text.replace('-', ' ').capitalize())
                
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
def main():
    st.title("MTG Deckbuilding Analysis Tool")

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
    data_source_option = st.sidebar.radio("Choose a data source:", ("Upload CSV", "Scrape New Data"), key="data_source")
    
    commander_slug_for_tools = "ojer-axonil-deepest-might"

    if data_source_option == "Upload CSV":
        decklist_file = st.sidebar.file_uploader("Upload Combined Decklists CSV", type=["csv"])
        if decklist_file:
            commander_slug_for_tools = decklist_file.name.split('_combined_decklists.csv')[0]
            st.session_state.commander_colors = get_commander_color_identity(commander_slug_for_tools)
            df_raw = pd.read_csv(decklist_file)
            st.session_state.scraped_df = df_raw
            
    elif data_source_option == "Scrape New Data":
        commander_slug = st.sidebar.text_input("Enter Commander Slug", "ojer-axonil-deepest-might")
        
        bracket_options = {
            "All Decks": "", "Budget": "budget", "Upgraded": "upgraded", 
            "Optimized": "optimized", "cEDH": "cedh"
        }
        selected_bracket_name = st.sidebar.selectbox("Select Bracket Level:", options=list(bracket_options.keys()))
        selected_bracket_slug = bracket_options[selected_bracket_name]

        deck_limit = st.sidebar.slider("Number of decks to scrape", 10, 200, 50)
        
        if st.sidebar.button("🚀 Start Scraping"):
            with st.spinner("Scraping in progress... this may take several minutes."):
                df_scraped, colors = run_scraper(
                    commander_slug, deck_limit, 
                    bracket_slug=selected_bracket_slug, 
                    bracket_name=selected_bracket_name
                )
                st.session_state.scraped_df = df_scraped
                st.session_state.commander_colors = colors
                st.rerun()

    if 'scraped_df' in st.session_state and st.session_state.scraped_df is not None:
        df_raw = st.session_state.scraped_df
        st.sidebar.success("Scraped data is loaded.")
        if data_source_option == "Scrape New Data":
            commander_slug_for_tools = commander_slug

    # --- RESET BUTTON ---
    st.sidebar.divider()
    if st.sidebar.button("🧹 Clear All Data & Reset"):
        keys_to_clear = [
            'scraped_df', 'commander_colors', 'master_categories', 'junk_tags',
            'imported_tags', 'func_constraints', 'type_constraints'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("All data cleared!")
        time.sleep(1)
        st.rerun()

    # ===============================================================
    # CARD CATEGORY DATA LOADING LOGIC
    # ===============================================================
    st.sidebar.header("Card Categories")

    if st.sidebar.button("Import Broad Categories from EDHREC 📋"):
        with st.spinner("Importing functional categories from EDHREC..."):
            edhrec_tags_df = import_edhrec_categories()
        if not edhrec_tags_df.empty:
            st.session_state.imported_tags = edhrec_tags_df
            st.toast(f"Successfully compiled categories for {len(edhrec_tags_df)} cards!", icon="✅")
            st.rerun()

    if 'master_categories' not in st.session_state and st.session_state.gsheets_connected:
        with st.spinner("Loading your categories from Google Sheets..."):
            st.session_state.master_categories = conn.read(worksheet="Categories")

    if 'junk_tags' not in st.session_state and st.session_state.gsheets_connected:
        with st.spinner("Loading junk tag list..."):
            try:
                junk_df = conn.read(worksheet="JunkTags")
                if not junk_df.empty and 'tag' in junk_df.columns:
                    st.session_state.junk_tags = junk_df['tag'].dropna().tolist()
                else: st.session_state.junk_tags = []
            except Exception:
                st.sidebar.info("No 'JunkTags' worksheet found. Using defaults.")
                st.session_state.junk_tags = []

    if st.session_state.gsheets_connected:
        if st.sidebar.button("Scrape Tagger for GSheet Cards 🔎"):
            if 'master_categories' in st.session_state and not st.session_state.master_categories.empty:
                unique_cards = st.session_state.master_categories['name'].dropna().unique().tolist()
                junk_tags_list = st.session_state.get('junk_tags', [])
                with st.spinner(f"Scraping Scryfall Tagger for {len(unique_cards)} cards..."):
                    tagger_df = scrape_scryfall_tagger(unique_cards, junk_tags_list)
                if not tagger_df.empty:
                    st.session_state.imported_tags = tagger_df
                    st.toast(f"Scraped tags for {len(tagger_df)} cards!", icon="✅")
                    st.rerun()
            else:
                st.sidebar.warning("Your 'Categories' GSheet is empty. Add card names first.")

    # --- MODIFIED: Robust category loading and merging ---
    categories_df_master = pd.DataFrame(columns=['name', 'category'])
    imported_df = st.session_state.get('imported_tags', pd.DataFrame())
    gsheets_df = st.session_state.get('master_categories', pd.DataFrame())
    
    # Gracefully handle DataFrames that are None or lack the 'name' column
    if gsheets_df is None or 'name' not in gsheets_df.columns:
        if gsheets_df is not None and not gsheets_df.empty:
            st.warning("Your 'Categories' Google Sheet is missing the 'name' column and will be ignored.")
        gsheets_df = pd.DataFrame(columns=['name', 'category'])

    if imported_df is None or 'name' not in imported_df.columns:
        imported_df = pd.DataFrame(columns=['name', 'category'])

    if not gsheets_df.empty or not imported_df.empty:
        merged_df = pd.merge(gsheets_df, imported_df, on='name', how='outer', suffixes=('_gsheet', '_imported'))
        merged_df['category_gsheet'] = merged_df['category_gsheet'].fillna('')
        merged_df['category_imported'] = merged_df['category_imported'].fillna('')
        merged_df['category'] = np.where(
            merged_df['category_gsheet'] != '', 
            merged_df['category_gsheet'], 
            merged_df['category_imported']
        )
        categories_df_master = merged_df[['name', 'category']].sort_values('name').reset_index(drop=True)
        st.sidebar.info("Combined GSheet & Imported tags.")
    # --- END MODIFICATION ---

    st.sidebar.divider()
    
    # --- MAIN APP DISPLAY LOGIC ---
    if df_raw is not None:
        df, FUNCTIONAL_ANALYSIS_ENABLED, NUM_DECKS, POP_ALL = clean_and_prepare_data(df_raw, categories_df_master)
        st.success(f"Data loaded with {NUM_DECKS} unique decks. Ready for analysis.")

        st.header("Dashboard & Analysis")
        col1, col2 = st.columns(2)
        with col1:
            price_cap = st.number_input('Price cap ($):', min_value=0.0, value=5.0, step=0.5)
            main_top_n = st.slider('Top N Staples:', 5, 100, 25, 5)
            exclude_top = st.checkbox('Exclude Top N Staples', False)
        with col2:
            unique_types = sorted([t for t in df['type'].unique() if t is not None])
            exclude_types = st.multiselect('Exclude Types:', options=unique_types, default=[])

        filtered_df = df.copy()
        if price_cap > 0: filtered_df = filtered_df[(filtered_df['price_clean'].isna()) | (filtered_df['price_clean'] <= price_cap)]
        if exclude_top: filtered_df = filtered_df[~filtered_df['name'].isin(POP_ALL['name'].head(main_top_n).tolist())]
        if exclude_types: filtered_df = filtered_df[~filtered_df['type'].isin(exclude_types)]

        spells_df = filtered_df[~filtered_df['type'].str.contains('Land', na=False)]
        lands_df = filtered_df[filtered_df['type'].str.contains('Land', na=False)]

        st.subheader("Top Spells, Lands & Curves")
        c1, c2, c3 = st.columns(3)
        with c1:
            pop_spells = popularity_table(spells_df)
            if not pop_spells.empty:
                fig1 = px.bar(pop_spells.head(25), y='name', x='count', orientation='h', title=f'Top {min(25, len(pop_spells))} Spells')
                fig1.update_layout(yaxis=dict(autorange='reversed'), height=600); st.plotly_chart(fig1, use_container_width=True)
        with c2:
            pop_lands = popularity_table(lands_df)
            if not pop_lands.empty:
                fig_lands = px.bar(pop_lands.head(25), y='name', x='count', orientation='h', title='Top 25 Lands')
                fig_lands.update_layout(yaxis=dict(autorange='reversed'), height=600); st.plotly_chart(fig_lands, use_container_width=True)
        with c3:
            curve = spells_df.groupby('cmc').size().reset_index(name='count')
            fig2 = px.bar(curve, x='cmc', y='count', title='Mana Curve (Spells Only)'); st.plotly_chart(fig2, use_container_width=True)

        if FUNCTIONAL_ANALYSIS_ENABLED and not spells_df.empty:
            with st.expander("Functional Analysis"):
                func_df = spells_df.copy()
                func_df['category'] = func_df['category'].fillna('Uncategorized')
                func_df['category_list'] = func_df['category'].str.split('|')
                func_df = func_df.explode('category_list').loc[lambda d: (d['category_list'] != 'Uncategorized') & (d['category_list'] != '')]
                
                if not func_df.empty:
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        sunburst_fig = px.sunburst(func_df, path=['category_list'], title='Functional Breakdown'); st.plotly_chart(sunburst_fig, use_container_width=True)
                    with fc2:
                        box_fig = px.box(func_df, x='category_list', y='cmc', title='CMC Distribution by Function'); st.plotly_chart(box_fig, use_container_width=True)
                else:
                    st.info("No categorized card functions found to display in the analysis.")

        with st.expander("Personal Deckbuilding Tools", expanded=True):
            st.subheader("Analyze Your Decklist")
            decklist_input = st.text_area("Paste your decklist here:", height=200, key="deck_analyzer_input")
            if st.button("🔍 Analyze My Deck"):
                user_decklist = parse_decklist(decklist_input)
                if user_decklist:
                    st.write(f"Analyzing {len(user_decklist)} cards...")
                    staples = POP_ALL[POP_ALL['inclusion_rate'] >= 75]
                    missing_staples = staples[~staples['name'].isin(user_decklist)]
                    st.write("Popular Staples Missing From Your Deck (>75% inclusion):")
                    st.dataframe(missing_staples[['name', 'inclusion_rate']].round(1))
                else:
                    st.warning("Please paste a decklist to analyze.")

            st.subheader("Generate Average Deck")
            deck_prices = df.groupby('deck_id')['price_clean'].sum()
            min_p, max_p = float(deck_prices.min()), float(deck_prices.max())
            price_range = st.slider("Filter decks by Total Price for Average Deck:", min_value=min_p, max_value=max_p, value=(min_p, max_p))
            if st.button("📊 Generate Average Deck"):
                with st.spinner("Generating average deck..."):
                    decks_in_range = deck_prices[(deck_prices >= price_range[0]) & (deck_prices <= price_range[1])].index
                    filtered_price_df = df[df['deck_id'].isin(decks_in_range)]
                    avg_deck = generate_average_deck(filtered_price_df, commander_slug_for_tools, st.session_state.get('commander_colors', []))
                    if avg_deck:
                        st.info(f"Detected Commander Color Identity: {', '.join(st.session_state.get('commander_colors', ['None']))}")
                        st.dataframe(pd.DataFrame(avg_deck, columns=["Card Name"]))
            
            st.subheader("Generate a Deck Template")
            if FUNCTIONAL_ANALYSIS_ENABLED:
                st.write("First, build your list of constraints. Then, set their ranges and generate the deck inside the form below.")
                
                if 'func_constraints' not in st.session_state: st.session_state.func_constraints = {}
                if 'type_constraints' not in st.session_state: st.session_state.type_constraints = {}

                all_individual_categories = df['category'].str.split('|').explode()
                func_categories_list = sorted([cat for cat in all_individual_categories.unique() if pd.notna(cat) and cat not in ['Uncategorized', '']])
                type_categories_list = sorted(df['type'].unique())

                with st.expander("Step 1: Configure Functional Constraints", expanded=True):
                    available_funcs = [f for f in func_categories_list if f not in st.session_state.func_constraints]
                    if available_funcs:
                        col1, col2 = st.columns([3, 1])
                        new_func = col1.selectbox("Add functional category:", options=available_funcs, key="new_func_select", index=None, placeholder="Choose a function...")
                        if col2.button("Add Function", key="add_func_btn") and new_func:
                            st.session_state.func_constraints[new_func] = (8, 12) if new_func in ['Ramp', 'Card Draw'] else (2, 8)
                            st.rerun()
                    
                    for func in list(st.session_state.func_constraints.keys()):
                        col1, col2 = st.columns([4, 1])
                        col1.write(f"- **{func}**")
                        if col2.button("Remove", key=f"del_func_{func}"):
                            del st.session_state.func_constraints[func]
                            st.rerun()

                with st.expander("Step 2: Configure Card Type Constraints"):
                    available_types = [t for t in type_categories_list if t not in st.session_state.type_constraints]
                    if available_types:
                        col1, col2 = st.columns([3, 1])
                        new_type = col1.selectbox("Add card type:", options=available_types, key="new_type_select", index=None, placeholder="Choose a type...")
                        if col2.button("Add Type", key="add_type_btn") and new_type:
                            st.session_state.type_constraints[new_type] = (25, 35) if new_type == 'Creature' else (5, 15)
                            st.rerun()

                    for ctype in list(st.session_state.type_constraints.keys()):
                        col1, col2 = st.columns([4, 1])
                        col1.write(f"- **{ctype}**")
                        if col2.button("Remove", key=f"del_type_{ctype}"):
                            del st.session_state.type_constraints[ctype]
                            st.rerun()
                
                with st.form(key='template_form'):
                    st.write("---")
                    st.write("**Step 3: Set Ranges and Generate**")
                    template_must_haves = st.text_area("Must-Include Cards (one per line):", key="template_must_haves")
                    
                    if not st.session_state.func_constraints and not st.session_state.type_constraints:
                        st.info("Add functional or card type constraints above to set their ranges here.")

                    for func, value in st.session_state.func_constraints.items():
                        st.session_state.func_constraints[func] = st.slider(f"Range for '{func}'", 0, 40, value, key=f"slider_func_{func}")
                    
                    for ctype, value in st.session_state.type_constraints.items():
                        st.session_state.type_constraints[ctype] = st.slider(f"Range for '{ctype}'", 0, 60, value, key=f"slider_type_{ctype}")
                    
                    submitted = st.form_submit_button("📋 Generate Deck With Constraints")

                    if submitted:
                        constraints = {'functions': {}, 'types': {}}
                        for func, (min_val, max_val) in st.session_state.func_constraints.items():
                            constraints['functions'][func] = {'target': [min_val, max_val], 'current': 0}
                        for ctype, (min_val, max_val) in st.session_state.type_constraints.items():
                            constraints['types'][ctype] = {'target': [min_val, max_val], 'current': 0}

                        with st.spinner("Generating decklists..."):
                            must_haves = parse_decklist(template_must_haves)
                            base_candidates = df.drop_duplicates(subset=['name']).copy().merge(POP_ALL[['name', 'count']], on='name')
                            base_candidates['category_list'] = base_candidates['category'].str.split('|')
                            candidates_pop = base_candidates[~base_candidates['name'].isin(must_haves)].sort_values('count', ascending=False)
                            candidates_eff = base_candidates[~base_candidates['name'].isin(must_haves)].copy()
                            median_cmc = candidates_eff['cmc'].median()
                            candidates_eff['efficiency_score'] = candidates_eff['count'] / (candidates_eff['cmc'].fillna(median_cmc) + 1)
                            candidates_eff = candidates_eff.sort_values('efficiency_score', ascending=False)

                            pop_deck, _ = _fill_deck_slots(candidates_pop, deepcopy(constraints), initial_decklist=must_haves)
                            eff_deck, _ = _fill_deck_slots(candidates_eff, deepcopy(constraints), initial_decklist=must_haves)

                            pop_df = pd.DataFrame(pop_deck, columns=["Popularity Build"])
                            eff_df = pd.DataFrame(eff_deck, columns=["Efficiency Build"])
                            
                            t1, t2 = st.tabs(["Popularity Build", "Efficiency Build"])
                            with t1: st.dataframe(pop_df)
                            with t2: st.dataframe(eff_df)
            else:
                st.warning("Import categories or connect to Google Sheets to enable the Deck Template Generator.")

        if st.session_state.gsheets_connected:
            with st.expander("Card Category Editor", expanded=False):
                st.info("Here you can add or edit categories for all unique cards found in the current dataset. Changes will be saved to your Google Sheet.")
                unique_cards_df = pd.DataFrame(df['name'].unique(), columns=['name']).sort_values('name').reset_index(drop=True)
                editor_df = pd.merge(unique_cards_df, categories_df_master, on='name', how='left').fillna('')
                
                st.write("Edit categories below (use '|' to separate multiple functions):")
                edited_df = st.data_editor(editor_df, key='category_editor', num_rows="dynamic", use_container_width=True, hide_index=True)
                
                if st.button("💾 Save Changes to Google Sheet"):
                    with st.spinner("Saving to Google Sheet..."):
                        if 'master_categories' not in st.session_state or st.session_state.master_categories.empty:
                            st.session_state.master_categories = pd.DataFrame(columns=['name', 'category'])
                            
                        updated_master = pd.concat([
                            st.session_state.master_categories[~st.session_state.master_categories['name'].isin(edited_df['name'])],
                            edited_df
                        ]).drop_duplicates(subset=['name'], keep='last')
                        
                        updated_master = updated_master[updated_master['name'] != ''].sort_values('name')

                        conn.update(worksheet="Categories", data=updated_master)
                        st.session_state.master_categories = updated_master
                        st.success("Categories saved successfully!")
                        time.sleep(1)
                        st.rerun()

        with st.expander("Advanced Synergy Tools", expanded=False):
            st.subheader("Card Inspector")
            all_spells_list = sorted(spells_df['name'].unique())
            if all_spells_list:
                selected_card = st.selectbox("Select a card to inspect:", all_spells_list)
                if selected_card:
                    decks_with_card = filtered_df[filtered_df['name'] == selected_card]['deck_id'].unique()
                    synergy_df = filtered_df[filtered_df['deck_id'].isin(decks_with_card)]
                    synergy_pop = popularity_table(synergy_df)
                    synergy_pop = synergy_pop[synergy_pop['name'] != selected_card]
                    st.write(f"Top 20 cards played with '{selected_card}':")
                    st.dataframe(synergy_pop.head(20))
            
            st.subheader("Synergy Map")
            if st.button("🗺️ Create Synergy Map"):
                with st.spinner("Generating Synergy Map..."):
                    deck_card_matrix = pd.crosstab(spells_df['deck_id'], spells_df['name'])
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(deck_card_matrix.columns)-1), max_iter=1000)
                    embedding = tsne.fit_transform(deck_card_matrix.T)
                    plot_df = pd.DataFrame(embedding, columns=['x', 'y'])
                    plot_df['card_name'] = deck_card_matrix.columns
                    plot_df = pd.merge(plot_df, df[['name', 'type']].drop_duplicates().rename(columns={'name': 'card_name'}), on='card_name', how='left')
                    fig = px.scatter(plot_df, x='x', y='y', hover_name='card_name', color='type', title='Card Synergy Map', height=800)
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Synergy Heatmap")
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                heatmap_top_n = st.slider('Top N for Heatmap:', 10, 50, 25, 5, key="heatmap_top_n")
            with h_col2:
                heatmap_exclude_n = st.slider('Exclude Staples:', 0, 25, 0, 1, key="heatmap_exclude_n")
            if st.button("🔥 Build Heatmap"):
                with st.spinner("Building Heatmap..."):
                    co = build_cococcurrence(filtered_df, topN=heatmap_top_n, exclude_staples_n=heatmap_exclude_n, pop_all_df=POP_ALL)
                    if co.empty: st.warning("Co-occurrence matrix is empty.")
                    else:
                        title = f'Card Co-occurrence (Top {heatmap_top_n}, excluding {heatmap_exclude_n})'
                        fig = px.imshow(co, color_continuous_scale='Purples', title=title, height=700, width=700)
                        st.plotly_chart(fig, use_container_width=True)

            st.subheader("Synergy Packages")
            COMMON_STAPLES = ['Sol Ring', 'Arcane Signet', 'Command Tower', 'Lightning Greaves', 'Swiftfoot Boots']
            packages_exclude_staples = st.multiselect('Exclude Staples from Packages:', options=COMMON_STAPLES, default=['Sol Ring', 'Arcane Signet'])
            packages_support_slider = st.slider('Min Support %:', 0.1, 0.7, 0.3, 0.05)
            if st.button("📦 Find Synergy Packages"):
                with st.spinner("Finding packages..."):
                    spells_to_analyze = spells_df[~spells_df['name'].isin(packages_exclude_staples)]
                    deck_card_matrix = pd.crosstab(spells_to_analyze['deck_id'], spells_to_analyze['name']) > 0
                    frequent_itemsets = apriori(deck_card_matrix, min_support=packages_support_slider, use_colnames=True)
                    if frequent_itemsets.empty: st.warning("No packages found. Try lowering 'Min Support %'.")
                    else:
                        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
                        result = frequent_itemsets[frequent_itemsets['length'] >= 2].sort_values(['length', 'support'], ascending=False)
                        result['itemsets'] = result['itemsets'].apply(lambda x: ', '.join(list(x)))
                        st.write(f"Found {len(result)} Synergy Packages:")
                        st.dataframe(result)

    else:
        st.info("👋 Welcome! Please upload a CSV or scrape new data using the sidebar to get started.")

if __name__ == "__main__":
    if setup_complete:
        main()
