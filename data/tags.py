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
    return _sheet_id() is not None


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
        return pd.DataFrame(columns=["name", "category"])

    df = pd.read_csv(path)

    if "name" not in df.columns:
        df["name"] = ""
    if "category" not in df.columns:
        df["category"] = ""

    df["name"] = df["name"].astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str)

    # Only return the columns we actually care about
    return df[["name", "category"]]


@st.cache_data(show_spinner=False)
def load_tags_from_gsheet(worksheet: str = "Categories") -> pd.DataFrame:
    """
    Attempt to load tag categories from Google Sheets.

    If there is no secrets.toml, if the gsheets block is missing, or if
    the sheet cannot be read, this returns an empty DataFrame with the
    expected columns so the app can fall back to local CSV tags.
    """
    sheet_id = _sheet_id()
    if not sheet_id:
        # No configured sheet → just return empty and let caller decide fallback
        return pd.DataFrame(columns=["name", "category"])

    df = safe_read_gsheet(sheet_id, worksheet)

    if df is None or df.empty:
        return pd.DataFrame(columns=["name", "category"])

    if "name" not in df.columns:
        df["name"] = ""
    if "category" not in df.columns:
        df["category"] = ""

    df["name"] = df["name"].astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str)

    return df[["name", "category"]]


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
