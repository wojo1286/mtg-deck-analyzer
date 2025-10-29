"""
Parsing utilities for EDHREC and related deck sources.
Handles HTML table extraction and card type inference.
"""

from bs4 import BeautifulSoup
import re
import streamlit as st
from core.config import TYPE_KEYWORDS


def _extract_primary_type(text: str | None) -> str | None:
    """Extracts the main card type (Creature, Land, Instant, etc.) from a text line."""
    if not text:
        return None

    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return None

    lowered = cleaned.lower()
    for keyword in TYPE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword.lower())}\b", lowered):
            return keyword

    # Handle special separators (e.g., "Creature — Elf")
    for separator in ("—", "-", "/"):
        if separator in cleaned:
            prefix = cleaned.split(separator, 1)[0].strip()
            if prefix:
                return _extract_primary_type(prefix)
    return None


def _extract_type_from_row(tr, tds, type_idx: int | None) -> str | None:
    """
    Attempts to extract a card's type from an HTML table row.
    Looks through typical table attributes and text patterns.
    """
    attribute_keys = {
        "data-type-line",
        "data-typeline",
        "data-type",
        "data-card-type",
        "data-cardtype",
        "data-card-types",
        "data-cardtypes",
        "data-type_line",
    }
    type_hint_attrs = {
        "data-title",
        "title",
        "aria-label",
        "data-tooltip",
        "data-tooltip-content",
        "data-tooltip-title",
        "data-label",
        "data-th",
        "headers",
    }

    candidate_texts: list[str] = []

    if type_idx is not None and len(tds) > type_idx:
        candidate_texts.append(tds[type_idx].get_text(" ", strip=True))

    def record_value(value):
        if not value:
            return
        if isinstance(value, (list, tuple, set)):
            for v in value:
                record_value(v)
        else:
            candidate_texts.append(str(value))

    # Collect attribute-based hints
    for attr in attribute_keys:
        record_value(tr.get(attr))

    def collect_from_tag(tag):
        if tag is None:
            return
        for attr, value in tag.attrs.items():
            attr_lower = attr.lower()
            if attr_lower in attribute_keys:
                record_value(value)
            elif attr_lower in type_hint_attrs:
                joined = " ".join(value) if isinstance(value, (list, tuple, set)) else str(value)
                if "type" in joined.lower():
                    record_value(joined)
                    text_value = tag.get_text(" ", strip=True)
                    if text_value:
                        candidate_texts.append(text_value)
        class_list = tag.get("class", [])
        if isinstance(class_list, str):
            class_list = [class_list]
        if any("type" in cls for cls in class_list):
            candidate_texts.append(tag.get_text(" ", strip=True))

    for td in tds:
        collect_from_tag(td)
        for child in td.find_all(True):
            collect_from_tag(child)

    if tr:
        collect_from_tag(tr)

    # Return first valid match
    for text in candidate_texts:
        ctype = _extract_primary_type(text)
        if ctype:
            return ctype

    row_text = tr.get_text(" ", strip=True) if tr else ""
    return _extract_primary_type(row_text)


@st.cache_data(show_spinner=False)
def parse_table(html: str, deck_id: str, deck_source: str) -> list[dict]:
    """
    Parses an EDHREC deck table and returns card data.

    Returns a list of dicts:
        [{'deck_id':..., 'deck_source':..., 'cmc':..., 'name':..., 'type':..., 'price':...}]
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = []

    for table in soup.find_all("table"):
        header_row = table.find("tr")
        header_cells = header_row.find_all(["th", "td"]) if header_row else []
        has_header = bool(header_row and header_row.find_all("th"))

        type_idx = price_idx = cmc_idx = None
        for idx, cell in enumerate(header_cells):
            header_text = cell.get_text(strip=True).lower()
            if "type" in header_text:
                type_idx = idx
            elif "price" in header_text or "card kingdom" in header_text:
                price_idx = idx
            elif "cmc" in header_text or "cost" in header_text:
                cmc_idx = idx

        rows = table.find_all("tr")
        data_rows = rows[1:] if has_header else rows

        if type_idx is None:
            st.warning(f"⚠️ Could not find 'Type' column in deck {deck_id}.")

        for tr in data_rows:
            tds = tr.find_all("td")
            if not tds:
                continue

            name_el = tr.find("a")
            name = name_el.get_text(strip=True) if name_el else None

            cmc = None
            if cmc_idx is not None and len(tds) > cmc_idx:
                cmc = tds[cmc_idx].get_text(strip=True)
            else:
                cmc_el = tr.find("span", class_="float-right")
                if cmc_el:
                    cmc = cmc_el.get_text(strip=True)

            raw_type = None
            if type_idx is not None and len(tds) > type_idx:
                raw_type = tds[type_idx].get_text(strip=True)
            else:
                raw_type = next(
                    (
                        td.get_text(strip=True)
                        for td in tds
                        if _extract_primary_type(td.get_text(strip=True))
                    ),
                    None,
                )
            ctype = _extract_primary_type(raw_type)

            price = None
            if price_idx is not None and len(tds) > price_idx:
                price = tds[price_idx].get_text(strip=True)
            else:
                price = next(
                    (
                        td.get_text(strip=True)
                        for td in reversed(tds)
                        if td.get_text(strip=True).startswith("$")
                    ),
                    None,
                )

            if name:
                cards.append(
                    {
                        "deck_id": deck_id,
                        "deck_source": deck_source,
                        "cmc": cmc,
                        "name": name,
                        "type": ctype,
                        "price": price,
                    }
                )

    if not cards:
        st.warning(f"No cards parsed from deck {deck_id}.")
    return cards
