"""
Scraping utilities for EDHREC.

This module now covers two layers:
1) Deck metadata scraping from https://edhrec.com/decks/<commander> (sync Playwright) -> scrape_deck_metadata()
2) Shared async Playwright helpers for deckpreview-page scraping (used by data/decklists.py):
   - BrowserManager (shared Chromium/context)
   - backoff(), goto_with_backoff(), wait_for_selector_with_backoff()
"""

from __future__ import annotations

import asyncio
import random
from typing import Callable, Awaitable, TypeVar

import nest_asyncio
import pandas as pd
import streamlit as st
from playwright.sync_api import sync_playwright  # used by the sync metadata scraper
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from core.cache import ensure_playwright

# Apply AFTER imports, BEFORE any asyncio usage
nest_asyncio.apply()


@st.cache_resource
def _get_browser():
    """Ensures Playwright Chromium is installed and launches a shared browser (sync API)."""
    ensure_playwright()
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    return browser, playwright


def _fetch_html(url: str, wait_selector: str = "table", retries: int = 3) -> str | None:
    """Fetches HTML and waits for deck table to load (sync API)."""
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
def scrape_deck_metadata(
    commander_slug: str, bracket: str = "all", budget: str = "all"
) -> pd.DataFrame:
    """
    Scrapes the /decks/<commander> page to extract deck URLs and metadata (sync API).

    Returns:
        DataFrame with columns:
        [deck_name, deck_url, bracket, budget, uploaded, likes, comments]
    """
    base_url = f"https://edhrec.com/decks/{commander_slug}"
    url = f"{base_url}?p=1"

    html = _fetch_html(url, wait_selector="table")

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


T = TypeVar("T")


async def backoff(
    func: Callable[[], Awaitable[T]],
    *,
    tries: int = 4,
    base_delay: float = 0.75,
    max_delay: float = 6.0,
    exceptions: tuple[type[Exception], ...] = (PWTimeout,),
) -> T:
    """
    Exponential backoff with jitter for awaited operations (e.g., page.goto, wait_for_selector).
    """
    delay = base_delay
    last_err: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            return await func()
        except exceptions as e:
            last_err = e
            if attempt >= tries:
                break
            # jitter in [0.5, 1.5] * delay
            await asyncio.sleep(min(max_delay, delay * (0.5 + random.random())))
            delay *= 2
    if last_err:
        raise last_err
    raise RuntimeError("backoff terminated unexpectedly without returning or raising.")


class BrowserManager:
    """
    Shared async Chromium browser/context for multiple deckpreview fetches.

    Usage:
        async with BrowserManager() as mgr:
            async with mgr.page() as page:
                await page.goto(...)
    """

    def __init__(self, headless: bool = True):
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._context = None

    async def startup(self):
        ensure_playwright()
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        if self._browser is None:
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
        if self._context is None:
            self._context = await self._browser.new_context(ignore_https_errors=True)

    async def shutdown(self):
        try:
            if self._context:
                await self._context.close()
        finally:
            self._context = None
        try:
            if self._browser:
                await self._browser.close()
        finally:
            self._browser = None
        try:
            if self._playwright:
                await self._playwright.stop()
        finally:
            self._playwright = None

    async def __aenter__(self):
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

    @asynccontextmanager
    async def page(self):
        """Async context manager that yields a new page and auto-closes it."""
        await self.startup()
        page = await self._context.new_page()
        try:
            yield page
        finally:
            await page.close()


async def goto_with_backoff(page, url: str, *, timeout_ms: int = 90000):
    async def _go():
        return await page.goto(url, timeout=timeout_ms)

    return await backoff(_go)


async def wait_for_selector_with_backoff(page, selector: str, *, timeout_ms: int = 10000):
    async def _wait():
        return await page.wait_for_selector(selector, timeout=timeout_ms)

    return await backoff(_wait)
