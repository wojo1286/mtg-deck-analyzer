# api_server.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List

app = FastAPI()

# TODO: restrict this later to your deployed AI Studio app origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or ["https://aistudio.google.com", "..."]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- EDHREC decks endpoint ----

@app.get("/edhrec/decks")
def get_edhrec_decks(
    commander_slug: str = Query(..., alias="commanderSlug"),
    bracket: str = Query("", alias="bracket"),
    limit: int = Query(20, alias="limit"),
) -> Dict[str, Any]:
    """
    Return decks + card rows for a commander from EDHREC.

    Response:
    {
      "decks": [{ "id": str, "source": str }],
      "cards": [{ "deckId": str, "cardName": str, "typeLine": str, "cmc": float, "price": float }]
    }
    """

    # --- 1. Use your existing scraping code here ---
    # Pseudocode: call into your existing Python functions
    #
    # from data.decklists import scrape_edhrec_decks_for_commander
    # decks_df, cards_df = scrape_edhrec_decks_for_commander(
    #     commander_slug=commander_slug,
    #     bracket=bracket,
    #     limit=limit,
    # )
    #
    # Where:
    #  - decks_df has columns: ["deck_id", "source", ...]
    #  - cards_df has columns: ["deck_id", "card_name", "type_line", "cmc", "price"]
    #
    # For now, I'll show a simple skeleton you can adapt:

    import requests
    from bs4 import BeautifulSoup

    def build_commander_decks_json_url(slug: str, bracket_slug: str = "") -> str:
        base = f"https://json.edhrec.com/pages/decks/{slug}"
        if bracket_slug:
            base += f"/{bracket_slug}"
        return f"{base}.json"

    # Map UI bracket to EDHREC slug if needed
    bracket_slug = {
        "All Decks": "",
        "Budget": "budget",
        "Upgraded": "upgraded",
        "Optimized": "optimized",
        "cEDH": "cedh",
    }.get(bracket, "")

    decks_json_url = build_commander_decks_json_url(commander_slug, bracket_slug)
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(decks_json_url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # EDHREC JSON structure may differ; adjust this to match your existing scraper.
    deck_hashes: List[str] = []
    for section in data.get("container", {}).get("json_dict", {}).get("decks", []):
        for d in section.get("decks", []):
            url_hash = d.get("urlhash")
            if url_hash:
                deck_hashes.append(url_hash)

    deck_hashes = deck_hashes[:limit]

    decks: List[Dict[str, Any]] = []
    cards: List[Dict[str, Any]] = []

    for idx, url_hash in enumerate(deck_hashes):
        deck_id = f"deck-{idx}"
        decks.append({"id": deck_id, "source": "edhrec"})

        deck_url = f"https://edhrec.com/deckpreview/{url_hash}"
        html_resp = requests.get(deck_url, headers=headers, timeout=30)
        html_resp.raise_for_status()
        soup = BeautifulSoup(html_resp.text, "html.parser")

        # Find the main card table – you may need to tweak these selectors
        table = soup.find("table")
        if not table:
            continue

        header_cells = [th.get_text(strip=True).lower() for th in table.find("thead").find_all("th")]
        name_idx = header_cells.index("card") if "card" in header_cells else header_cells.index("name")
        type_idx = header_cells.index("type") if "type" in header_cells else None
        cmc_idx = header_cells.index("cmc") if "cmc" in header_cells else None
        price_idx = header_cells.index("price") if "price" in header_cells else None

        for row in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cells:
                continue
            card_name = cells[name_idx]
            type_line = cells[type_idx] if type_idx is not None and len(cells) > type_idx else ""
            cmc_str = cells[cmc_idx] if cmc_idx is not None and len(cells) > cmc_idx else "0"
            price_str = cells[price_idx] if price_idx is not None and len(cells) > price_idx else "0"

            try:
                cmc = float(cmc_str)
            except ValueError:
                cmc = 0.0
            try:
                price = float(price_str.replace("$", "").replace(",", "")) if price_str else 0.0
            except ValueError:
                price = 0.0

            cards.append(
                {
                    "deckId": deck_id,
                    "cardName": card_name,
                    "typeLine": type_line,
                    "cmc": cmc,
                    "price": price,
                }
            )

    return {"decks": decks, "cards": cards}


# ---- Scryfall Tagger endpoint ----

@app.get("/tags/scryfall")
def get_scryfall_tags(
    card_name: str = Query(..., alias="cardName"),
) -> Dict[str, Any]:
    """
    Look up tags for a card from Scryfall Tagger (or your own tag DB).
    For now, you can stub this or implement scraping from tagger.scryfall.com.
    """
    # TODO: replace this stub with real Scryfall Tagger logic
    # e.g. requests.get(tag_url), parse HTML with BeautifulSoup, extract tag pills

    # Simple stub so the AI Studio app works:
    tags = []
    if card_name.lower() in {"sol ring", "arcane signet"}:
        tags = ["Ramp", "Mana Rock"]

    return {"cardName": card_name, "tags": tags}
