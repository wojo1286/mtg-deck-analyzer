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

from core.cache import ensure_playwright
from core.gsheets import safe_read_gsheet, safe_update_gsheet


def _sheet_id() -> str | None:
    """Return the configured Google Sheet id, if available."""
    cfg = st.secrets.get("gsheets", {}) if hasattr(st, "secrets") else {}
    sheet_id = cfg.get("sheet_id") if isinstance(cfg, dict) else None
    if sheet_id:
        return str(sheet_id)
    return None


def has_gsheet_categories() -> bool:
    return _sheet_id() is not None


@st.cache_data(show_spinner=False)
def load_card_tags() -> pd.DataFrame:
    """Load card tags exported from Scryfall Tagger."""
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
    return df


@st.cache_data(show_spinner=False)
def load_tags_from_gsheet(worksheet: str = "Categories") -> pd.DataFrame:
    """Attempt to load tag categories from Google Sheets."""
    sheet_id = _sheet_id()
    if not sheet_id:
        return pd.DataFrame(columns=["name", "category"])
    df = safe_read_gsheet(sheet_id, worksheet)
    if df.empty:
        return pd.DataFrame(columns=["name", "category"])
    if "name" not in df.columns:
        df["name"] = ""
    if "category" not in df.columns:
        df["category"] = ""
    df["name"] = df["name"].astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str)
    return df


def save_tags_to_gsheet(df: pd.DataFrame, worksheet: str = "Categories") -> None:
    """Persist the provided DataFrame back to the configured Google Sheet."""
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
def scrape_scryfall_tagger(card_names: Iterable[str], junk_tags: Iterable[str] | None = None) -> pd.DataFrame:
    """Scrape Scryfall Tagger for the given card names and return tag data."""
    names = [str(name).strip() for name in card_names if str(name).strip()]
    if not names:
        return pd.DataFrame(columns=["name", "category"])

    ensure_playwright()

    excluded = {"abrade", "modal", "single english word name"}
    if junk_tags:
        excluded.update({str(tag).lower() for tag in junk_tags})

    progress = st.progress(0, text="Initializing Scryfall Tagger scrape...")
    scraped: dict[str, list[str]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page()
        for idx, card_name in enumerate(names, start=1):
            try:
                encoded = urllib.parse.quote_plus(card_name)
                response = requests.get(f"https://api.scryfall.com/cards/named?fuzzy={encoded}", timeout=15)
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
                progress.progress(idx / len(names), text=f"Scraping '{card_name}' ({idx}/{len(names)})...")
                time.sleep(0.1)
        browser.close()

    progress.empty()

    if not scraped:
        return pd.DataFrame(columns=["name", "category"])

    data = [{"name": name, "category": "|".join(tags)} for name, tags in scraped.items()]
    return pd.DataFrame(data)
