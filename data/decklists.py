from __future__ import annotations

import asyncio
import nest_asyncio
import pandas as pd
import streamlit as st


from typing import List, Dict, Any
from urllib.parse import urlparse

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from core.cache import ensure_playwright
from data.parsing import _extract_primary_type
from data.scraping import BrowserManager, goto_with_backoff, wait_for_selector_with_backoff
from bs4 import BeautifulSoup

nest_asyncio.apply()

# ---------------------------------------------------------------------
# Debug / behavior toggles (flip to True if you want step-by-step logs)
# ---------------------------------------------------------------------
DEBUG_SHOW_STEPS = False  # stream step-by-step actions
DEBUG_HTML_SNAPSHOT = False  # show HTML slice for the FIRST deck
DEBUG_HTML_SLICE_LEN = 6000
SCROLL_AFTER_ROWS = True  # scroll to coax virtualized rows
SCROLL_PIXELS = 600
SCROLL_WAIT_MS = 800


# -----------------------------
# Small utilities
# -----------------------------
def _extract_deck_id(deck_url: str) -> str:
    """
    Extract the deckpreview hash as a unique deck identifier.
    Example: https://edhrec.com/deckpreview/kHw_wlJv_VzrnS04-t0vbA --> kHw_wlJv_VzrnS04-t0vbA
    """
    try:
        path = urlparse(deck_url).path
        return path.rstrip("/").split("/")[-1] or deck_url
    except Exception:
        return deck_url


# -----------------------------
# Async Playwright helpers
# -----------------------------
async def _click_table_tab(page) -> bool:
    """Click the Table tab using multiple selector fallbacks. Return True if clicked/visible."""
    selectors = [
        "button[data-rr-ui-event-key='table']",
        "button:has-text('Table')",
        "role=tab[name='Table']",
        "button[aria-controls='viewTabs-pane-table']",
    ]
    for sel in selectors:
        try:
            if await page.is_visible(sel):
                if DEBUG_SHOW_STEPS:
                    st.write(f"• Trying table tab selector: `{sel}`")
                await page.click(sel, timeout=5000)
                return True
        except Exception:
            pass
    return False


async def _ensure_table_container(page) -> None:
    """Wait for the table container or a plain <table> fallback."""
    try:
        await page.wait_for_selector('div[class*="TableView_table"]', timeout=10000)
        if DEBUG_SHOW_STEPS:
            st.write("• TableView_table container detected.")
        return
    except PWTimeout:
        if DEBUG_SHOW_STEPS:
            st.write("• TableView_table NOT detected, falling back to generic <table>.")
        await page.wait_for_selector("table", timeout=8000)


async def _ensure_type_column(page) -> None:
    """Open 'Edit Columns' and enable 'Type' if needed, using multiple fallbacks."""
    try:
        visible = await page.is_visible('th:has-text("Type")')
        if visible:
            if DEBUG_SHOW_STEPS:
                st.write("• 'Type' column already visible.")
            return

        # Open the menu
        for open_sel in [
            "button:has-text('Edit Columns')",
            "button[aria-label='Edit Columns']",
        ]:
            try:
                if await page.is_visible(open_sel):
                    if DEBUG_SHOW_STEPS:
                        st.write(f"• Opening edit columns via: `{open_sel}`")
                    await page.click(open_sel, timeout=5000)
                    break
            except Exception:
                pass

        # Wait for dropdown / menu (best-effort)
        try:
            await page.wait_for_selector('div[class*="dropdown-menu"][class*="show"]', timeout=5000)
        except PWTimeout:
            await page.wait_for_timeout(250)

        # Toggle "Type" with several possible widgets
        toggle_selectors = [
            'div[class*="dropdown-menu"][class*="show"] button:has-text("Type")',
            'div[class*="dropdown-menu"][class*="show"] [role="menuitem"]:has-text("Type")',
            'div[class*="dropdown-menu"][class*="show"] label:has-text("Type")',
            'label:has-text("Type")',
            'button:has-text("Type")',
        ]
        toggled = False
        for sel in toggle_selectors:
            try:
                if DEBUG_SHOW_STEPS:
                    st.write(f"• Trying to toggle 'Type' with: `{sel}`")
                await page.click(sel, timeout=2000)
                toggled = True
                break
            except Exception:
                pass

        if toggled:
            await page.wait_for_selector('th:has-text("Type")', timeout=10000)
            if DEBUG_SHOW_STEPS:
                st.write("• 'Type' column is now visible.")
        else:
            if DEBUG_SHOW_STEPS:
                st.warning("• Could not toggle 'Type' column (continuing).")

    except Exception as e:
        st.warning(f"• Error ensuring 'Type' column: {e}")


async def _wait_for_rows(page) -> int:
    """Wait for any rows to be present; return count seen. Also try gentle scroll for virtualization."""
    # Primary wait: real table cells
    try:
        await page.wait_for_selector("table tbody tr td", timeout=15000)
    except PWTimeout:
        # Fallback: any row with td
        try:
            await page.wait_for_selector("tr:has(td)", timeout=7000)
        except PWTimeout:
            pass

    locator = page.locator("table tbody tr")
    try:
        count = await locator.count()
    except Exception:
        count = 0

    if DEBUG_SHOW_STEPS:
        st.write(f"• Row count before scroll: {count}")

    if count == 0 and SCROLL_AFTER_ROWS:
        try:
            await page.evaluate(f"window.scrollBy(0, {SCROLL_PIXELS})")
            await page.wait_for_timeout(SCROLL_WAIT_MS)
            count = await locator.count()
            if DEBUG_SHOW_STEPS:
                st.write(f"• Row count after scroll: {count}")
        except Exception:
            pass

    return count


async def _fetch_deck_html_async(url: str) -> str | None:
    """Fetch deck HTML using the interaction flow your original app used, with fallbacks."""
    ensure_playwright()
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()

        try:
            if DEBUG_SHOW_STEPS:
                st.write(f"➡️  Goto: {url}")
            await page.goto(url, timeout=90000)

            # Try to switch to Table tab
            clicked = await _click_table_tab(page)
            if not clicked and DEBUG_SHOW_STEPS:
                st.write("• Table tab click not performed (maybe already active).")

            await _ensure_table_container(page)
            await _ensure_type_column(page)
            _ = await _wait_for_rows(page)

            html = await page.content()

            await context.close()
            await browser.close()

            return html
        except Exception as e:
            await context.close()
            await browser.close()
            st.error(f"❌ Failed to fetch deck HTML for {url}: {e}")
            return None


def _fetch_deck_html(url: str) -> str | None:
    """Run the async fetch safely under Streamlit's running event loop."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return loop.run_until_complete(_fetch_deck_html_async(url))
    else:
        return asyncio.run(_fetch_deck_html_async(url))


# -----------------------------
# HTML parsing (table + role-grid)
# -----------------------------
def _parse_deck_html(deck_html: str, deck_url: str, deck_name: str) -> List[Dict[str, Any]]:
    """
    Parse the deck HTML into card rows.
    Primary path: <table> parsing with header discovery.
    Fallback: ARIA/role-based grids (div[role='row'] / div[role='cell']).
    """
    soup = BeautifulSoup(deck_html, "html.parser")
    cards: List[Dict[str, Any]] = []
    deck_id = _extract_deck_id(deck_url)

    # --- PATH A: classic table(s)
    tables = soup.find_all("table")
    if tables:
        for table in tables:
            header_row = table.find("tr")
            if not header_row:
                continue
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all("th")]
            col_map = {h: i for i, h in enumerate(headers)}

            rows = table.find_all("tr")[1:]
            for tr in rows:
                tds = tr.find_all("td")
                if not tds:
                    continue

                # Card name (first link)
                name_el = tr.find("a")
                name = name_el.get_text(strip=True) if name_el else None
                if not name:
                    # Fallback: try first cell text
                    name = tds[0].get_text(" ", strip=True) if tds else None
                if not name:
                    continue

                # Type
                raw_type = None
                if "type" in col_map and len(tds) > col_map["type"]:
                    raw_type = tds[col_map["type"]].get_text(" ", strip=True)
                else:
                    # heuristic fallback: any td that looks like a type line
                    for td in tds:
                        txt = td.get_text(" ", strip=True)
                        ctype = _extract_primary_type(txt)
                        if ctype:
                            raw_type = txt
                            break
                ctype = _extract_primary_type(raw_type) if raw_type else None

                # CMC
                cmc = None
                if "cmc" in col_map and len(tds) > col_map["cmc"]:
                    cmc = tds[col_map["cmc"]].get_text(strip=True)
                else:
                    # Some markup uses a right-aligned span for CMC
                    rhs = tr.find("span", class_="float-right")
                    cmc = rhs.get_text(strip=True) if rhs else None

                # Price
                price = None
                if "price" in col_map and len(tds) > col_map["price"]:
                    price = tds[col_map["price"]].get_text(strip=True)
                else:
                    # last cell that starts with $
                    for td in reversed(tds):
                        txt = td.get_text(strip=True)
                        if txt.startswith("$"):
                            price = txt
                            break

                # Qty (rarely present)
                qty = None
                if "qty" in col_map and len(tds) > col_map["qty"]:
                    qty = tds[col_map["qty"]].get_text(strip=True)
                if not qty:
                    qty = "1"

                cards.append(
                    {
                        "deck_id": deck_id,  # <-- unique deck key
                        "deck_name": deck_name or "View Decklist",
                        "deck_url": deck_url,
                        "name": name,
                        "type": ctype or "Unknown",
                        "cmc": cmc,
                        "price": price,
                        "qty": qty,
                    }
                )

        if cards:
            return cards  # success via table path

    # --- PATH B: role-based grid fallback
    role_rows = soup.select("div[role='rowgroup'] div[role='row']")
    if role_rows:
        for rr in role_rows:
            cells = rr.select("div[role='cell']")
            if not cells:
                continue
            # name from first link or first cell
            link = rr.find("a")
            name = link.get_text(strip=True) if link else cells[0].get_text(" ", strip=True)
            if not name:
                continue

            # try to infer type from any cell values
            ctype = None
            for c in cells:
                txt = c.get_text(" ", strip=True)
                ctype = _extract_primary_type(txt)
                if ctype:
                    break

            # naive cmc/price hunt (best-effort)
            cmc = None
            price = None
            for c in cells:
                txt = c.get_text(" ", strip=True)
                if cmc is None and txt.isdigit():
                    cmc = txt
                if price is None and txt.startswith("$"):
                    price = txt

            cards.append(
                {
                    "deck_id": deck_id,  # <-- unique deck key
                    "deck_name": deck_name or "View Decklist",
                    "deck_url": deck_url,
                    "name": name,
                    "type": ctype or "Unknown",
                    "cmc": cmc,
                    "price": price,
                    "qty": "1",
                }
            )

    return cards


# -----------------------------
# Public API
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_decklists(deck_df: pd.DataFrame, max_decks: int = 10) -> pd.DataFrame:
    """
    Fetch and parse decklists from EDHREC deckpreview URLs.

    Args:
        deck_df: DataFrame returned from your deck metadata step; must have deck_url/deck_name
        max_decks: number of decks to scrape (user-controlled)

    Returns:
        DataFrame of card rows:
        [deck_id, deck_name, deck_url, name, type, cmc, price, qty]
    """
    if deck_df is None or deck_df.empty or "deck_url" not in deck_df.columns:
        st.warning("No deck metadata to scrape.")
        return pd.DataFrame(
            columns=[
                "deck_id",
                "deck_name",
                "deck_url",
                "name",
                "type",
                "cmc",
                "price",
                "qty",
            ]
        )

    all_cards: List[Dict[str, Any]] = []
    selected = deck_df.head(max_decks).copy()

    for idx, row in selected.iterrows():
        deck_name = row.get("deck_name") or "View Decklist"
        deck_url = row.get("deck_url")
        if not deck_url:
            continue

        st.info(f"Fetching decklist for: {deck_name}")
        html = _fetch_deck_html(deck_url)
        if not html:
            continue

        # Debug snapshot for the first deck
        if DEBUG_HTML_SNAPSHOT and idx == selected.index[0]:
            with st.expander("HTML snapshot (first deck)", expanded=False):
                st.code(html[:DEBUG_HTML_SLICE_LEN])

        cards = _parse_deck_html(html, deck_url, deck_name)
        if DEBUG_SHOW_STEPS:
            st.write(f"• Parsed {len(cards)} cards from this deck.")

        all_cards.extend(cards)

    df = pd.DataFrame(all_cards)
    if df.empty:
        st.warning("No cards were parsed from any deck.")
    else:
        st.success(f"✅ Parsed {len(df)} cards from {min(len(selected), max_decks)} decks.")
    return df


@st.cache_data(show_spinner=False)
def fetch_decklists_shared(deck_df: pd.DataFrame, max_decks: int = 10) -> pd.DataFrame:
    """
    Same output as fetch_decklists, but uses a single shared browser/context
    for all deck URLs in this call (faster and nicer to EDHREC).
    """
    if deck_df is None or deck_df.empty or "deck_url" not in deck_df.columns:
        st.warning("No deck metadata to scrape.")
        return pd.DataFrame(
            columns=[
                "deck_id",
                "deck_name",
                "deck_url",
                "name",
                "type",
                "cmc",
                "price",
                "qty",
            ]
        )

    selected = deck_df.head(max_decks).copy()

    async def _run() -> pd.DataFrame:
        mgr = BrowserManager(headless=True)
        rows: list[dict] = []
        try:
            await mgr.startup()
            for idx, row in selected.iterrows():
                deck_name = row.get("deck_name") or "View Decklist"
                deck_url = row.get("deck_url")
                if not deck_url:
                    continue
                st.info(f"Fetching decklist for: {deck_name}")
                async with mgr.page() as page:
                    # same interaction flow as your working version, but using goto_with_backoff
                    await goto_with_backoff(page, deck_url)
                    # switch to Table tab if needed
                    for sel in [
                        "button[data-rr-ui-event-key='table']",
                        "button:has-text('Table')",
                        "role=tab[name='Table']",
                        "button[aria-controls='viewTabs-pane-table']",
                    ]:
                        try:
                            if await page.is_visible(sel):
                                await page.click(sel, timeout=5000)
                                break
                        except Exception:
                            pass

                    # table container (or <table> fallback)
                    try:
                        await wait_for_selector_with_backoff(
                            page, 'div[class*="TableView_table"]', timeout_ms=10000
                        )
                    except Exception:
                        await wait_for_selector_with_backoff(page, "table", timeout_ms=8000)

                    # ensure Type column
                    try:
                        if not await page.is_visible('th:has-text("Type")'):
                            for open_sel in [
                                "button:has-text('Edit Columns')",
                                "button[aria-label='Edit Columns']",
                            ]:
                                try:
                                    if await page.is_visible(open_sel):
                                        await page.click(open_sel, timeout=5000)
                                        break
                                except Exception:
                                    pass
                            try:
                                await page.wait_for_selector(
                                    'div[class*="dropdown-menu"][class*="show"]',
                                    timeout=5000,
                                )
                            except Exception:
                                await page.wait_for_timeout(250)
                            for sel in [
                                'div[class*="dropdown-menu"][class*="show"] button:has-text("Type")',
                                'div[class*="dropdown-menu"][class*="show"] [role="menuitem"]:has-text("Type")',
                                'div[class*="dropdown-menu"][class*="show"] label:has-text("Type")',
                                'label:has-text("Type")',
                                'button:has-text("Type")',
                            ]:
                                try:
                                    await page.click(sel, timeout=2000)
                                    break
                                except Exception:
                                    pass
                            try:
                                await page.wait_for_selector('th:has-text("Type")', timeout=8000)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # wait for rows (best-effort + gentle scroll)
                    try:
                        await page.wait_for_selector("table tbody tr td", timeout=12000)
                    except Exception:
                        try:
                            await page.wait_for_selector("tr:has(td)", timeout=8000)
                        except Exception:
                            pass
                    try:
                        await page.evaluate("window.scrollBy(0, 600)")
                        await page.wait_for_timeout(800)
                    except Exception:
                        pass

                    html = await page.content()
                    rows.extend(_parse_deck_html(html, deck_url, deck_name))
            return pd.DataFrame(rows)
        finally:
            await mgr.shutdown()

    loop = asyncio.get_event_loop()
    if loop.is_running():
        df = loop.run_until_complete(_run())
    else:
        df = asyncio.run(_run())

    if df.empty:
        st.warning("No cards were parsed from any deck.")
    else:
        st.success(
            f"✅ Parsed {len(df)} cards from {min(len(selected), max_decks)} decks (shared browser)."
        )
    return df
