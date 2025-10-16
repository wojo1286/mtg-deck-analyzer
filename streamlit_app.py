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

# --- Visualization ---
import plotly.express as px
import plotly.graph_objects as go

# --- Advanced Analytics ---
from mlxtend.frequent_patterns import apriori
from sklearn.manifold import TSNE

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

def parse_table(html, deck_id, deck_source):
    soup = BeautifulSoup(html, "html.parser")
    cards = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 6:
                continue
            cmc_el = tr.find("span", class_="float-right")
            cmc = cmc_el.get_text(strip=True) if cmc_el else None
            name_el = tr.find("a")
            name = name_el.get_text(strip=True) if name_el else None
            ctype = None
            for td in tds:
                text = td.get_text(strip=True)
                if text in ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Land", "Battle"]:
                    ctype = text
                    break
            price = None
            for td in reversed(tds):
                txt = td.get_text(strip=True)
                if txt.startswith("$"):
                    price = txt
                    break
            if name:
                cards.append({
                    "deck_id": deck_id,
                    "deck_source": deck_source,
                    "cmc": cmc,
                    "name": name,
                    "type": ctype,
                    "price": price
                })
    return cards

def run_scraper(commander_slug, deck_limit, bracket_slug="", budget_slug="", bracket_name="All Decks"):
    st.info(f"🔍 Fetching deck metadata for '{commander_slug}' (Bracket: {bracket_name})...")
    
    base_url = f"https://json.edhrec.com/pages/decks/{commander_slug}"
    if bracket_slug:
        base_url += f"/{bracket_slug}"
    if budget_slug:
        base_url += f"/{budget_slug}"
    json_url = base_url + ".json"

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(json_url, headers=headers); r.raise_for_status(); data = r.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch metadata from EDHREC. URL: {json_url}. Error: {e}"); return None

    decks = data.get("table", [])
    if not decks:
        st.error(f"No decks found for '{commander_slug}' in the '{bracket_name}' bracket."); return None

    df_meta = pd.json_normalize(decks)
    df_meta["deckpreview_url"] = df_meta["urlhash"].apply(lambda x: f"https://edhrec.com/deckpreview/{x}")
    sample_df = df_meta.head(deck_limit)
    st.success(f"Found {len(decks)} total decks. Scraping the first {len(sample_df)}.")

    all_cards = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page()
        for i, row in sample_df.iterrows():
            deck_id, deck_url = row["urlhash"], row["deckpreview_url"]
            status_text.text(f"[{i+1}/{len(sample_df)}] Fetching {deck_url}")
            try:
                page.goto(deck_url, timeout=90000)
                
                page.wait_for_selector('button.nav-link[aria-controls*="table"]', timeout=15000)
                page.click('button.nav-link[aria-controls*="table"]')
                page.wait_for_selector("table", timeout=20000)
                
                try:
                    page.click("button#dropdown-item-button.dropdown-toggle", timeout=10000)
                    page.wait_for_selector("button.dropdown-item", timeout=5000)
                    for btn in page.query_selector_all("button.dropdown-item"):
                        if "Type" in btn.inner_text():
                            btn.click()
                            break
                    page.wait_for_selector("th:has-text('Type')", timeout=5000)
                except Exception:
                    pass

                html = page.content()
                src_el = BeautifulSoup(html, "html.parser").find("a", href=lambda x: x and any(d in x for d in ["moxfield", "archidekt"]))
                deck_source = src_el["href"] if src_el else "Unknown"
                cards = parse_table(html, deck_id, deck_source)

                if cards: 
                    all_cards.extend(cards)
                else:
                    st.warning(f"No cards parsed for {deck_url}.")
                        
                time.sleep(random.uniform(1.0, 2.5))
            except Exception as e:
                status_text.text(f"⚠️ Skipping deck {deck_id} due to error: {e}")
            progress_bar.progress((i + 1) / len(sample_df))
        browser.close()
    
    if not all_cards: 
        st.error("Scraping complete, but no cards were successfully parsed across all decks."); 
        return None
    st.success("✅ Scraping complete!"); 
    return pd.DataFrame(all_cards)


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

# ===================================================================
# 4. STREAMLIT UI & APP LOGIC
# ===================================================================
def main():
    st.title("MTG Deckbuilding Analysis Tool")

    st.sidebar.header("Data Source")
    df_raw = None
    categories_df = None
    if 'scraped_df' not in st.session_state: st.session_state.scraped_df = None
    data_source_option = st.sidebar.radio("Choose a data source:", ("Upload CSV", "Scrape New Data"), key="data_source")

    if data_source_option == "Upload CSV":
        decklist_file = st.sidebar.file_uploader("Upload Combined Decklists CSV", type=["csv"])
        categories_file = st.sidebar.file_uploader("Upload Card Categories CSV (Optional)", type=["csv"])
        if decklist_file:
            df_raw = pd.read_csv(decklist_file)
            if categories_file: categories_df = pd.read_csv(categories_file)
    elif data_source_option == "Scrape New Data":
        commander_slug = st.sidebar.text_input("Enter Commander Slug", "ojer-axonil-deepest-might")
        
        bracket_options = {
            "All Decks": "", "Bracket 1 - Exhibition (Budget)": "exhibition", "Bracket 2 - Core": "core",
            "Bracket 3 - Upgraded": "upgraded", "Bracket 4 - Optimized": "optimized", "Bracket 5 - cEDH": "cedh"
        }
        selected_bracket_name = st.sidebar.selectbox("Select Bracket Level:", options=list(bracket_options.keys()))
        selected_bracket_slug = bracket_options[selected_bracket_name]

        budget_options = {"Any Budget": "", "Budget": "budget", "Expensive": "expensive"}
        selected_budget_name = st.sidebar.selectbox("Select Budget Option:", options=list(budget_options.keys()))
        selected_budget_slug = budget_options[selected_budget_name]

        deck_limit = st.sidebar.slider("Number of decks to scrape", 10, 500, 150)
        if st.sidebar.button("🚀 Start Scraping"):
            with st.spinner("Scraping in progress... this may take several minutes."):
                display_bracket_name = selected_bracket_name + (f" ({selected_budget_name})" if selected_budget_name != "Any Budget" else "")
                st.session_state.scraped_df = run_scraper(commander_slug, deck_limit, selected_bracket_slug, selected_budget_slug, display_bracket_name)
        if st.session_state.scraped_df is not None:
            df_raw = st.session_state.scraped_df
            st.sidebar.success("Scraped data is loaded.")
            categories_file = st.sidebar.file_uploader("Upload Card Categories CSV (Optional)", type=["csv"])
            if categories_file: categories_df = pd.read_csv(categories_file)

    if df_raw is not None:
        df, FUNCTIONAL_ANALYSIS_ENABLED, NUM_DECKS, POP_ALL = clean_and_prepare_data(df_raw, categories_df)
        st.success(f"Data loaded with {NUM_DECKS} unique decks. Ready for analysis.")

        st.header("Dashboard & Analysis")
        col1, col2 = st.columns(2)
        with col1:
            price_cap = st.number_input('Price cap ($):', min_value=0.0, value=5.0, step=0.5)
            main_top_n = st.slider('Top N Staples:', 5, 100, 25, 5)
            exclude_top = st.checkbox('Exclude Top N Staples', False)
        with col2:
            unique_types = sorted(df['type'].unique())
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
                func_df['category_list'] = func_df['category'].str.split('|')
                func_df = func_df.explode('category_list').loc[lambda d: d['category_list'] != 'Uncategorized']
                fc1, fc2 = st.columns(2)
                with fc1:
                    sunburst_fig = px.sunburst(func_df, path=['category_list'], title='Functional Breakdown'); st.plotly_chart(sunburst_fig, use_container_width=True)
                with fc2:
                    box_fig = px.box(func_df, x='category_list', y='cmc', title='CMC Distribution by Function'); st.plotly_chart(box_fig, use_container_width=True)

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

            st.subheader("Generate a Deck Template")
            if FUNCTIONAL_ANALYSIS_ENABLED:
                with st.form(key='template_form'):
                    st.write("Define your deck structure with the constraints below.")
                    template_must_haves = st.text_area("Must-Include Cards (one per line):", key="template_must_haves")
                    
                    func_categories_list = sorted([cat for cat in df.explode('category')['category'].unique() if cat != 'Uncategorized'])
                    type_categories_list = sorted(df['type'].unique())
                    
                    constraints = {'functions': {}, 'types': {}}
                    
                    with st.expander("Functional Constraints", expanded=True):
                        for cat in func_categories_list:
                            col1, col2 = st.columns([1, 3])
                            is_enabled = col1.checkbox(cat, value=True, key=f"check_{cat}")
                            if is_enabled:
                                min_val, max_val = col2.slider("", 0, 40, (8, 12) if cat in ['Ramp', 'Card Draw'] else (0, 10), key=f"slider_{cat}")
                                constraints['functions'][cat] = {'target': [min_val, max_val], 'current': 0}

                    with st.expander("Card Type Constraints"):
                         for cat in type_categories_list:
                            col1, col2 = st.columns([1, 3])
                            is_enabled = col1.checkbox(cat, value=False, key=f"check_type_{cat}")
                            if is_enabled:
                                min_val, max_val = col2.slider("", 0, 60, (30, 40) if cat == 'Creature' else (0, 20), key=f"slider_type_{cat}")
                                constraints['types'][cat] = {'target': [min_val, max_val], 'current': 0}
                    
                    submitted = st.form_submit_button("📋 Generate Deck With Constraints")

                    if submitted:
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
                            with t1:
                                st.dataframe(pop_df)
                            with t2:
                                st.dataframe(eff_df)
            else:
                st.warning("Upload a `card_categories.csv` to enable the Deck Template Generator.")

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
