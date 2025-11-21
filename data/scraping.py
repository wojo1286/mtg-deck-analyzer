"""
Synchronous scraping helpers for EDHREC deck metadata.

The primary deck scraping pipeline now lives in
``data.decklists.scrape_edhrec_decks_for_commander``. This module remains as a
lightweight helper for metadata scraping when needed.
"""

from __future__ import annotations

import random
import time

import pandas as pd
import streamlit as st
from playwright.sync_api import sync_playwright

from core.cache import ensure_playwright


@st.cache_resource
def _get_browser():
    """Ensure Playwright Chromium is installed and return a reusable browser/playwright pair."""

    ensure_playwright()
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    return browser, playwright


def _fetch_html(
    url: str,
    *,
    wait_selector: str = "table",
    retries: int = 3,
    page_size: int | None = None,
) -> str | None:
    """Fetch HTML and wait for the desired selector to appear (sync Playwright)."""

    browser, playwright = _get_browser()
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()

    for attempt in range(1, retries + 1):
        try:
            page.goto(url, timeout=60000)
            page.wait_for_selector(wait_selector, timeout=25000)

            if page_size:
                try:
                    if page_size == 100:
                        locator = page.locator("text='100'")
                    else:
                        locator = page.locator(f"text='{page_size}'")
                    if locator.count():
                        locator.first.click(timeout=5000)
                        page.wait_for_timeout(1000)
                    else:
                        selectors = (
                            "button:has-text('100')" if page_size == 100 else f"button:has-text('{page_size}')",
                            "a:has-text('100')" if page_size == 100 else f"a:has-text('{page_size}')",
                            "li:has-text('100')" if page_size == 100 else f"li:has-text('{page_size}')",
                        )
                        for sel in selectors:
                            if page.is_visible(sel):
                                page.click(sel, timeout=5000)
                                page.wait_for_timeout(1000)
                                break
                except Exception:
                    pass

            html = page.content()
            context.close()
            return html
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Attempt {attempt}/{retries} failed for {url}: {exc}")
            time.sleep(2 * attempt + random.uniform(0, 2))

    context.close()
    st.error(f"Failed to fetch {url}")
    return None


@st.cache_data(show_spinner=False)
def scrape_deck_metadata(
    commander_slug: str,
    *,
    max_decks: int = 100,
    bracket: str = "all",
    budget: str = "all",
) -> pd.DataFrame:
    """
    Scrape the EDHREC /decks/<commander> page to extract deck URLs and metadata.

    Returns a DataFrame with columns:
        [deck_name, deck_url, bracket, budget, uploaded, likes, comments]
    """

    base_url = f"https://edhrec.com/decks/{commander_slug}"
    url = f"{base_url}?p=1"

    html = _fetch_html(url, wait_selector="table", page_size=100)

    if not html:
        return pd.DataFrame(
            columns=[
                "deck_name",
                "deck_url",
                "bracket",
                "budget",
                "uploaded",
                "likes",
                "comments",
            ]
        )

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    decks = []
    table = soup.find("table")
    if not table:
        st.warning(f"No deck table found for {commander_slug}")
        return pd.DataFrame()

    rows = table.find_all("tr")[1:]
    if max_decks and max_decks > 0:
        rows = rows[: int(max_decks)]

    for row in rows:
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

        decks.append(
            {
                "deck_name": deck_name,
                "deck_url": deck_url,
                "bracket": bracket,
                "budget": budget,
                "uploaded": uploaded,
                "likes": likes,
                "comments": comments,
            }
        )

    df = pd.DataFrame(decks)
    st.success(f"✅ Found {len(df)} decks for {commander_slug}")
    return df
