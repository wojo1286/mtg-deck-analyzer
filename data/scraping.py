"""
Scraping utilities for EDHREC deck metadata.
Targets https://edhrec.com/decks/<commander> and extracts deck URLs + info.
"""

import time
import random
import pandas as pd
import streamlit as st
from playwright.sync_api import sync_playwright
from core.cache import ensure_playwright


@st.cache_resource
def _get_browser():
    """Ensures Playwright Chromium is installed and launches a shared browser."""
    ensure_playwright()
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    return browser, playwright


def _fetch_html(url: str, wait_selector: str = "table", retries: int = 3) -> str | None:
    """Fetches HTML and waits for deck table to load."""
    browser, playwright = _get_browser()
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()

    for attempt in range(1, retries + 1):
        try:
            page.goto(url, timeout=60000)
            page.wait_for_selector(wait_selector, timeout=25000)
            html = page.content()
            context.close()
            return html
        except Exception as e:
            st.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}")
            time.sleep(2 * attempt + random.uniform(0, 2))

    context.close()
    st.error(f"Failed to fetch {url}")
    return None


@st.cache_data(show_spinner=False)
def scrape_deck_metadata(commander_slug: str, bracket: str = "all", budget: str = "all") -> pd.DataFrame:
    """
    Scrapes the /decks/<commander> page to extract deck URLs and metadata.

    Returns:
        DataFrame with columns:
        [deck_name, deck_url, bracket, budget, uploaded, likes, comments]
    """
    base_url = f"https://edhrec.com/decks/{commander_slug}"
    url = f"{base_url}?p=1"

    html = _fetch_html(url, wait_selector="table")

    if not html:
        return pd.DataFrame(columns=["deck_name", "deck_url", "bracket", "budget", "uploaded", "likes", "comments"])

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    decks = []
    table = soup.find("table")
    if not table:
        st.warning(f"No deck table found for {commander_slug}")
        return pd.DataFrame()

    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        link_tag = cols[0].find("a", href=True)
        deck_name = link_tag.get_text(strip=True) if link_tag else None
        deck_url = f"https://edhrec.com{link_tag['href']}" if link_tag else None
        uploaded = cols[1].get_text(strip=True) if len(cols) > 1 else None
        stats_text = cols[2].get_text(" ", strip=True) if len(cols) > 2 else ""

        likes = comments = None
        if "likes" in stats_text:
            likes = stats_text.split("likes")[0].strip().split()[-1]
        if "comments" in stats_text:
            comments = stats_text.split("comments")[0].strip().split()[-1]

        decks.append({
            "deck_name": deck_name,
            "deck_url": deck_url,
            "bracket": bracket,
            "budget": budget,
            "uploaded": uploaded,
            "likes": likes,
            "comments": comments
        })

    df = pd.DataFrame(decks)
    st.success(f"✅ Found {len(df)} decks for {commander_slug}")
    return df
