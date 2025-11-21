from __future__ import annotations

import random
import time
from typing import Callable

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PWTimeout, sync_playwright

from core.cache import ensure_playwright
from data.parsing import parse_table


def _build_commander_json_url(
    commander_slug: str, bracket_slug: str = "", budget_slug: str = ""
) -> str:
    """Build the EDHREC JSON endpoint for commander + optional bracket/budget."""

    base = f"https://json.edhrec.com/pages/decks/{commander_slug}"
    if bracket_slug:
        base += f"/{bracket_slug}"
    if budget_slug:
        base += f"/{budget_slug}"
    return f"{base}.json"


def _get_commander_color_identity(commander_slug: str) -> list[str]:
    """Fetch commander color identity with graceful failure handling."""

    try:
        url = f"https://json.edhrec.com/pages/commanders/{commander_slug}.json"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("container", {})
            .get("json_dict", {})
            .get("card", {})
            .get("color_identity", [])
        )
    except Exception:
        return []


def _extract_deck_source(html: str) -> str:
    """Identify the originating deck site (e.g., Moxfield/Archidekt) from the preview HTML."""

    soup = BeautifulSoup(html, "html.parser")
    link = soup.find(
        "a",
        href=lambda x: x and any(host in x for host in ["moxfield", "archidekt", "deckstats"]),
    )
    return link["href"] if link and link.has_attr("href") else "Unknown"


def _load_deck_page_html(page, deck_url: str) -> str | None:
    """Navigate to a deckpreview URL and return page HTML after ensuring the table is visible."""

    if not deck_url:
        return None

    try:
        page.goto(deck_url, timeout=90000)
        # Switch to the Table tab when present
        for sel in [
            "button[data-rr-ui-event-key='table']",
            "button:has-text('Table')",
            "role=tab[name='Table']",
            "button[aria-controls='viewTabs-pane-table']",
        ]:
            try:
                if page.is_visible(sel):
                    page.click(sel, timeout=5000)
                    break
            except Exception:
                continue

        try:
            page.wait_for_selector("div[class*='TableView_table']", timeout=10000)
        except PWTimeout:
            page.wait_for_selector("table", timeout=8000)

        # Ensure the Type column is enabled via the Edit Columns menu.
        try:
            if not page.is_visible('th:has-text("Type")'):
                for open_sel in ["button:has-text('Edit Columns')", "button[aria-label='Edit Columns']"]:
                    try:
                        if page.is_visible(open_sel):
                            page.click(open_sel, timeout=5000)
                            break
                    except Exception:
                        continue

                try:
                    page.wait_for_selector('div[class*="dropdown-menu"][class*="show"]', timeout=5000)
                except Exception:
                    page.wait_for_timeout(250)

                for sel in [
                    'div[class*="dropdown-menu"][class*="show"] button:has-text("Type")',
                    'div[class*="dropdown-menu"][class*="show"] [role="menuitem"]:has-text("Type")',
                    'div[class*="dropdown-menu"][class*="show"] label:has-text("Type")',
                    'label:has-text("Type")',
                    'button:has-text("Type")',
                ]:
                    try:
                        page.click(sel, timeout=2000)
                        break
                    except Exception:
                        continue

                try:
                    page.wait_for_selector('th:has-text("Type")', timeout=10000)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            page.wait_for_selector("table tbody tr td", timeout=15000)
        except Exception:
            try:
                page.wait_for_selector("tr:has(td)", timeout=8000)
            except Exception:
                pass

        try:
            page.evaluate("window.scrollBy(0, 600)")
            page.wait_for_timeout(800)
        except Exception:
            pass

        return page.content()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"⚠️ Skipping deck {deck_url} due to navigation error: {exc}")
        return None


def scrape_edhrec_decks_for_commander(
    commander_slug: str,
    deck_limit: int,
    bracket_slug: str = "",
    bracket_name: str = "All Decks",
    budget_slug: str = "",
    _html_fetcher: Callable[[str], str | None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Scrape deckpreview pages for a commander using EDHREC's JSON metadata endpoint.

    Args:
        commander_slug: EDHREC slug for the commander.
        deck_limit: maximum number of decks to scrape.
        bracket_slug: optional EDHREC bracket (budget/upgraded/optimized/cedh).
        bracket_name: display name for the bracket (used in logging).
        budget_slug: optional budget level slug used by EDHREC.
        _html_fetcher: internal hook to override deck HTML retrieval (used in tests).

    Returns:
        Tuple of (cards DataFrame, commander color identity list).
    """

    if not commander_slug:
        st.warning("Commander slug is required for scraping.")
        return pd.DataFrame(), []

    json_url = _build_commander_json_url(commander_slug, bracket_slug, budget_slug)
    headers = {"User-Agent": "Mozilla/5.0"}
    st.info(f"🔍 Fetching deck metadata for '{commander_slug}' (Bracket: {bracket_name})...")

    try:
        resp = requests.get(json_url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:  # noqa: BLE001
        st.error(f"Failed to fetch metadata from {json_url}: {exc}")
        return pd.DataFrame(), []

    color_identity = (
        data.get("container", {})
        .get("json_dict", {})
        .get("card", {})
        .get("color_identity", [])
    ) or _get_commander_color_identity(commander_slug)

    decks = data.get("table", [])
    if not decks:
        st.error(f"No decks found for '{commander_slug}' in '{bracket_name}'.")
        return pd.DataFrame(), color_identity

    df_meta = pd.json_normalize(decks)
    df_meta["deckpreview_url"] = df_meta.get("urlhash", "").apply(
        lambda x: f"https://edhrec.com/deckpreview/{x}"
    )
    selected = df_meta.head(deck_limit)
    if selected.empty:
        st.warning("No deck metadata available to scrape.")
        return pd.DataFrame(), color_identity

    st.success(f"Found {len(df_meta)} decks. Scraping the first {len(selected)}.")

    html_fetcher = _html_fetcher
    all_cards: list[dict] = []

    if html_fetcher is None:
        ensure_playwright()
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            page = browser.new_page()
            for i, row in selected.iterrows():
                deck_id = row.get("urlhash")
                deck_url = row.get("deckpreview_url")
                st.info(f"[{i + 1}/{len(selected)}] Fetching {deck_url}")
                html = _load_deck_page_html(page, deck_url)
                if not html:
                    continue

                deck_source = _extract_deck_source(html)
                cards = parse_table(html, deck_id=str(deck_id), deck_source=deck_source)
                if cards:
                    all_cards.extend(cards)
                else:
                    st.warning(
                        f"No cards parsed for {deck_url}; skipping this deck instead of failing."
                    )
                time.sleep(random.uniform(0.5, 1.5))
            browser.close()
    else:
        for i, row in selected.iterrows():
            deck_id = row.get("urlhash")
            deck_url = row.get("deckpreview_url")
            st.info(f"[{i + 1}/{len(selected)}] Fetching {deck_url}")
            html = html_fetcher(deck_url)
            if not html:
                continue

            deck_source = _extract_deck_source(html)
            cards = parse_table(html, deck_id=str(deck_id), deck_source=deck_source)
            if cards:
                all_cards.extend(cards)
            else:
                st.warning(
                    f"No cards parsed for {deck_url}; skipping this deck instead of failing."
                )

    if not all_cards:
        st.warning("Scraping completed but no cards were parsed from any deck.")
        return pd.DataFrame(), color_identity

    df_cards = pd.DataFrame(all_cards)
    st.success(f"✅ Parsed {len(df_cards)} cards from {len(selected)} decks.")
    return df_cards, color_identity
