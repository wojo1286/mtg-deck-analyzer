# data/tags.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import streamlit as st
import time
import urllib.parse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from streamlit.errors import StreamlitSecretNotFoundError

from core.cache import ensure_playwright
from core.gsheets import safe_read_gsheet, safe_update_gsheet


def _empty_tags_df() -> pd.DataFrame:
    """Return a consistently shaped, empty tag DataFrame."""

    return pd.DataFrame(columns=["name", "category"])


def _sheet_id() -> str | None:
    """
    Return the configured Google Sheet ID from st.secrets, or None if
    secrets are missing or misconfigured.

    This MUST be safe to call when there is no secrets.toml at all.
    """
    # In some environments st.secrets may not even exist or may throw when accessed.
    try:
        secrets_obj = getattr(st, "secrets", None)
    except Exception:
        return None

    # IMPORTANT: do NOT do `if not secrets_obj:` – that calls __len__ and
    # forces Streamlit to parse secrets.toml, which raises when missing.
    if secrets_obj is None:
        return None

    try:
        cfg = secrets_obj.get("gsheets", {})
    except StreamlitSecretNotFoundError:
        # No secrets.toml or empty / unparsable secrets
        return None
    except Exception:
        # Any other parsing/lookup error → fail soft
        return None

    sheet_id = cfg.get("sheet_id") if isinstance(cfg, dict) else None
    if not sheet_id:
        return None

    sheet_id_str = str(sheet_id).strip()
    return sheet_id_str or None


def has_gsheet_categories() -> bool:
    """True if a Google Sheet is configured for tag categories; False otherwise."""
    try:
        return _sheet_id() is not None
    except StreamlitSecretNotFoundError:
        return False
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_card_tags() -> pd.DataFrame:
    """
    Load card tags exported from Scryfall Tagger, using a local CSV as
    a simple, environment-independent fallback.

    This file is expected to live alongside this module as `card_tags.csv`
    with at least columns: [name, category].
    """
    path = Path(__file__).with_name("card_tags.csv")
    if not path.exists():
        return _empty_tags_df()

    df = pd.read_csv(path)
    return normalize_tags_df(df)


@st.cache_data(show_spinner=False)
def load_tags_from_gsheet(worksheet: str = "Categories") -> pd.DataFrame:
    """
    Attempt to load tag categories from Google Sheets.

    If there is no secrets.toml, if the gsheets block is missing, or if
    the sheet cannot be read, this returns an empty DataFrame with the
    expected columns so the app can fall back to local CSV tags.
    """
    try:
        sheet_id = _sheet_id()
    except StreamlitSecretNotFoundError:
        return _empty_tags_df()
    except Exception:
        return _empty_tags_df()

    if not sheet_id:
        # No configured sheet → just return empty and let caller decide fallback
        return _empty_tags_df()

    try:
        df = safe_read_gsheet(sheet_id, worksheet)
    except StreamlitSecretNotFoundError:
        return _empty_tags_df()
    except Exception:
        return _empty_tags_df()

    if df is None or df.empty:
        return _empty_tags_df()

    return normalize_tags_df(df)


def save_tags_to_gsheet(df: pd.DataFrame, worksheet: str = "Categories") -> None:
    """
    Persist the provided DataFrame back to the configured Google Sheet.

    This still expects proper credentials and will raise if not configured,
    because "Save to GSheet" is an explicit user action.
    """
    sheet_id = _sheet_id()
    if not sheet_id:
        raise RuntimeError("Google Sheets credentials are not configured.")

    work = df.copy()
    if "name" not in work.columns or "category" not in work.columns:
        raise ValueError("Expected columns 'name' and 'category' in tags DataFrame.")

    work["name"] = work["name"].astype(str).str.strip()
    work["category"] = work["category"].fillna("").astype(str)

    safe_update_gsheet(sheet_id, worksheet, work)


def _merge_categories(values: pd.Series) -> str:
    """Helper to combine multiple category strings into a deduped pipe list."""

    tags: list[str] = []
    for raw in values:
        text = str(raw or "")
        parts = [p.strip() for p in text.split("|") if p.strip()]
        tags.extend(parts)
    if not tags:
        return ""
    # Preserve deterministic order while deduping
    seen: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.append(tag)
    return "|".join(seen)


def normalize_tags_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and deduplicate a tags DataFrame.

    Ensures columns ["name", "category"] exist, strips whitespace, fills NA,
    and merges duplicate names by aggregating their categories with "|".
    """

    if df is None or df.empty:
        return _empty_tags_df()

    work = df.copy()
    if "name" not in work.columns:
        work["name"] = ""
    if "category" not in work.columns:
        work["category"] = ""

    work["name"] = work["name"].astype(str).str.strip()
    work["category"] = work["category"].fillna("").astype(str).str.strip()

    aggregated = (
        work.groupby("name", as_index=False)["category"].apply(_merge_categories)
    )
    return aggregated[["name", "category"]]


@st.cache_data(show_spinner=False)
def load_tag_categories() -> pd.DataFrame:
    """
    Load card role tags with clear precedence rules.

    Precedence:
        1) Google Sheets (worksheet "Categories") when configured and non-empty
        2) Local CSV `card_tags.csv` as a fallback
        3) Empty DataFrame with columns ["name", "category"]

    Returns a cleaned, deduplicated DataFrame based on the first available source.
    """

    sheet_df = load_tags_from_gsheet()
    if sheet_df is not None and not sheet_df.empty:
        return normalize_tags_df(sheet_df)

    csv_df = load_card_tags()
    if csv_df is not None and not csv_df.empty:
        return normalize_tags_df(csv_df)

    return _empty_tags_df()


@st.cache_data(show_spinner=False)
def scrape_scryfall_tagger(
    card_names: Iterable[str],
    junk_tags: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Scrape Scryfall Tagger for the given card names and return tag data.

    Returns a DataFrame with columns:
        - name: card name
        - category: '|' separated tag string

    Network failures are caught and surfaced via Streamlit warnings.
    """
    names = [str(name).strip() for name in card_names if str(name).strip()]
    if not names:
        return pd.DataFrame(columns=["name", "category"])

    ensure_playwright()

    # Default noisy or non-functional tags to exclude; caller can extend
    excluded = {"abrade", "modal", "single english word name"}
    if junk_tags:
        excluded.update({str(tag).lower() for tag in junk_tags})

    progress = st.progress(0, text="Initializing Scryfall Tagger scrape...")
    scraped: dict[str, list[str]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page()

        total = len(names)
        for idx, card_name in enumerate(names, start=1):
            try:
                encoded = urllib.parse.quote_plus(card_name)
                response = requests.get(
                    f"https://api.scryfall.com/cards/named?fuzzy={encoded}",
                    timeout=15,
                )
                response.raise_for_status()
                card_data = response.json()
                set_code = card_data.get("set")
                collector = card_data.get("collector_number")
                if not set_code or not collector:
                    continue

                tagger_url = f"https://tagger.scryfall.com/card/{set_code}/{collector}"
                page.goto(tagger_url, timeout=30000)
                page.wait_for_selector("a[href^='/tags/card/']", timeout=20000)

                soup = BeautifulSoup(page.content(), "html.parser")
                tags = set()
                for link in soup.find_all("a", href=lambda x: x and x.startswith("/tags/card/")):
                    tag_text = link.get_text(strip=True)
                    if not tag_text:
                        continue
                    if tag_text.lower() in excluded:
                        continue
                    tags.add(tag_text.replace("-", " ").capitalize())

                if tags:
                    scraped[card_name] = sorted(tags)
            except Exception as exc:  # pragma: no cover - network variability
                st.warning(f"Could not scrape '{card_name}'. ({exc})")
            finally:
                progress.progress(
                    idx / total,
                    text=f"Scraping '{card_name}' ({idx}/{total})...",
                )
                time.sleep(0.1)

        browser.close()

    progress.empty()

    if not scraped:
        return pd.DataFrame(columns=["name", "category"])

    data = [
        {"name": name, "category": "|".join(tags)}
        for name, tags in scraped.items()
    ]
    return pd.DataFrame(data)
