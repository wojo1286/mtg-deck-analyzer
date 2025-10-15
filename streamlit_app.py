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

# --- Visualization ---
import plotly.express as px
import plotly.graph_objects as go

# --- Advanced Analytics ---
from mlxtend.frequent_patterns import apriori
from sklearn.manifold import TSNE

# --- Page Config ---
st.set_page_config(layout="wide", page_title="MTG Deckbuilding Analysis Tool")

# ===================================================================
# 2. DATA SCRAPING FUNCTIONS (from edhrec_scraper.py)
# ===================================================================

def parse_table(html, deck_id, deck_source):
    """Parses the HTML of a deck table to extract card data."""
    soup = BeautifulSoup(html, "html.parser")
    cards = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 6: continue
            
            cmc_el = tr.find("span", class_="float-right")
            cmc = cmc_el.get_text(strip=True) if cmc_el else None
            name_el = tr.find("a")
            name = name_el.get_text(strip=True) if name_el else None
            
            ctype = None
            for td in tds:
                text = td.get_text(strip=True)
                if text in ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Land"]:
                    ctype = text; break
            
            price = None
            for td in reversed(tds):
                txt = td.get_text(strip=True)
                if txt.startswith("$"):
                    price = txt; break

            if name:
                cards.append({"deck_id": deck_id, "deck_source": deck_source, "cmc": cmc, "name": name, "type": ctype, "price": price})
    return cards

def run_scraper(commander_slug, deck_limit):
    """
    Main function to scrape decklists for a given commander from EDHREC.
    Shows progress in the Streamlit UI.
    """
    st.info(f"🔍 Fetching deck metadata for '{commander_slug}'...")
    json_url = f"https://json.edhrec.com/pages/decks/{commander_slug}/optimized.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(json_url, headers=headers)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch metadata from EDHREC. Error: {e}")
        return None

    decks = data.get("table", [])
    if not decks:
        st.error(f"No decks found for '{commander_slug}'. Please check the commander name.")
        return None

    df_meta = pd.json_normalize(decks)
    df_meta["deckpreview_url"] = df_meta["urlhash"].apply(lambda x: f"https://edhrec.com/deckpreview/{x}")
    sample_df = df_meta.head(deck_limit)
    st.success(f"Found {len(decks)} total decks. Scraping the first {len(sample_df)}.")

    all_cards = []
    progress_bar = st.progress(0)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for i, row in sample_df.iterrows():
            deck_id = row["urlhash"]
            deck_url = row["deckpreview_url"]
            status_text = st.empty()
            status_text.text(f"[{i+1}/{len(sample_df)}] Fetching {deck_url}")

            try:
                page.goto(deck_url, timeout=90000)
                page.wait_for_selector('button.nav-link[aria-controls*="table"]', timeout=15000)
                page.click('button.nav-link[aria-controls*="table"]')
                page.wait_for_selector("table", timeout=20000)
                
                html = page.content()
                src_el = BeautifulSoup(html, "html.parser").find("a", href=lambda x: x and any(d in x for d in ["moxfield", "archidekt"]))
                deck_source = src_el["href"] if src_el else "Unknown"
                
                cards = parse_table(html, deck_id, deck_source)
                if cards:
                    all_cards.extend(cards)
                
                time.sleep(random.uniform(1.0, 2.5)) # Be polite to the server
            except Exception as e:
                status_text.text(f"⚠️ Skipping deck {deck_id} due to error: {e}")
                time.sleep(1) # Pause on error
            
            progress_bar.progress((i + 1) / len(sample_df))
        browser.close()
    
    if not all_cards:
        st.error("Scraping complete, but no cards were successfully parsed.")
        return None
        
    final_df = pd.DataFrame(all_cards)
    st.success("✅ Scraping complete!")
    return final_df

# ===================================================================
# 3. DATA PROCESSING & ANALYSIS FUNCTIONS (from Colab notebook)
# ===================================================================
@st.cache_data
def clean_data(df, categories_df=None):
    """Applies cleaning, categorization, and pre-calculation to the raw DataFrame."""
    dfc = df.copy()
    dfc['price_clean'] = (dfc.get('price', pd.Series(dtype='str'))
                         .astype(str).str.replace(r'[$,]', '', regex=True)
                         .replace({'': np.nan}).astype(float))
    dfc['cmc'] = pd.to_numeric(dfc.get('cmc', np.nan), errors='coerce')
    dfc['type'] = dfc.get('type', 'Unknown').fillna('Unknown')
    
    functional_analysis_enabled = False
    if categories_df is not None:
        dfc = pd.merge(dfc, categories_df, on='name', how='left')
        dfc['category'] = dfc['category'].fillna('Uncategorized')
        functional_analysis_enabled = True
    
    return dfc, functional_analysis_enabled

# ... (All other analysis functions like popularity_table, apply_filters, etc. would go here) ...
# For brevity, these are assumed to exist and are the same as in the Colab notebook.
# Let's define a placeholder for one of them.
def popularity_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty: return pd.DataFrame(columns=['name', 'count', 'avg_price', 'avg_cmc', 'type'])
    return (frame.groupby('name')
            .agg(count=('deck_id','nunique'), avg_price=('price_clean','mean'), avg_cmc=('cmc','mean'), type=('type','first'))
            .reset_index().sort_values('count', ascending=False))

# ===================================================================
# 4. STREAMLIT UI LAYOUT
# ===================================================================
st.title("MTG Deckbuilding Analysis Tool")

# --- Sidebar for Controls & Data Loading ---
st.sidebar.header("Data Source")
data_source_option = st.sidebar.radio("Choose a data source:", ("Upload CSV", "Scrape New Data"))

df = None
categories_df = None

if 'scraped_df' not in st.session_state:
    st.session_state['scraped_df'] = None

if data_source_option == "Upload CSV":
    decklist_file = st.sidebar.file_uploader("Upload Combined Decklists CSV", type=["csv"])
    categories_file = st.sidebar.file_uploader("Upload Card Categories CSV (Optional)", type=["csv"])
    if decklist_file:
        df = pd.read_csv(decklist_file)
        if categories_file:
            categories_df = pd.read_csv(categories_file)
elif data_source_option == "Scrape New Data":
    commander_slug = st.sidebar.text_input("Enter Commander Slug (e.g., ojer-axonil-deepest-might)", "ojer-axonil-deepest-might")
    deck_limit = st.sidebar.slider("Number of decks to scrape", 10, 500, 150)
    if st.sidebar.button("🚀 Start Scraping"):
        with st.spinner("Scraping in progress... this may take several minutes."):
            scraped_data = run_scraper(commander_slug, deck_limit)
            st.session_state['scraped_df'] = scraped_data
    
    if st.session_state['scraped_df'] is not None:
        df = st.session_state['scraped_df']
        st.sidebar.success("Scraped data is loaded and ready for analysis.")
        # Allow uploading categories for scraped data
        categories_file = st.sidebar.file_uploader("Upload Card Categories CSV (Optional)", type=["csv"])
        if categories_file:
            categories_df = pd.read_csv(categories_file)


# --- Main Application Logic ---
if df is not None:
    # Clean the loaded data
    df, FUNCTIONAL_ANALYSIS_ENABLED = clean_data(df, categories_df)
    
    st.success(f"Data loaded with {df['deck_id'].nunique()} unique decks. Ready for analysis.")
    
    # Pre-calculate global stats for the loaded data
    num_decks = df['deck_id'].nunique()
    pop_all = popularity_table(df)
    pop_all['inclusion_rate'] = (pop_all['count'] / num_decks) * 100

    # --- RENDER THE REST OF THE UI (Ported from Colab) ---
    st.header("Dashboard & Analysis")
    
    # Create columns for filters
    col1, col2 = st.columns(2)
    with col1:
        price_cap = st.number_input('Price cap ($):', min_value=0.0, value=5.0, step=0.5)
        main_top_n = st.slider('Top N Staples:', min_value=5, max_value=100, value=25, step=5)
        exclude_top = st.checkbox('Exclude Top N Staples', value=False)
    with col2:
        unique_types = sorted(df['type'].unique())
        exclude_types = st.multiselect('Exclude Types:', options=unique_types, default=[])

    # Apply filters based on widgets
    # Note: Streamlit's reactive nature means we don't need a button.
    # The filtering logic would be applied directly before generating charts.
    
    # Placeholder for where the charts would be displayed
    st.subheader("Top Spells")
    
    # Create a filtered dataframe based on user input
    filtered_df = df.copy() # Start with the full dataframe
    if price_cap > 0:
        filtered_df = filtered_df[filtered_df['price_clean'] <= price_cap]
    if exclude_top:
        top_names = pop_all['name'].head(main_top_n).tolist()
        filtered_df = filtered_df[~filtered_df['name'].isin(top_names)]
    if exclude_types:
        filtered_df = filtered_df[~filtered_df['type'].isin(exclude_types)]
    
    spells_df = filtered_df[~filtered_df['type'].str.contains('Land', na=False)]
    pop_spells = popularity_table(spells_df)
    
    if not pop_spells.empty:
        fig1 = px.bar(pop_spells.head(25), y='name', x='count', orientation='h', title=f'Top {min(25, len(pop_spells))} Spells')
        fig1.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No spells match the current filter criteria.")

    # ... All other tools (Deck Analyzer, Template Generator, Synergy Map, etc.) would be ported here
    # using st.text_area, st.button, st.tabs, etc.
    
else:
    st.info("👋 Welcome! Please upload a CSV or scrape new data using the sidebar to get started.")
