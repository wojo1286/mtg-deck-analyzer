from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List

app = FastAPI()

# For now, allow all origins (you can tighten this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://aistudio.google.com"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- EDHREC Decks Endpoint ----------

@app.get("/edhrec/decks")
def get_edhrec_decks(
    commander_slug: str = Query(..., alias="commanderSlug"),
    bracket: str = Query("", alias="bracket"),  # you can use this later
    limit: int = Query(20, alias="limit"),
) -> Dict[str, Any]:
    """
    TEMP implementation: return a hard-coded test payload so the frontend can wire up.
    Then you will replace this body with your real EDHREC scraping logic.
    """
    decks = [
        {"id": "test-deck-1", "source": "test"},
        {"id": "test-deck-2", "source": "test"},
    ]

    cards = [
        {
            "deckId": "test-deck-1",
            "cardName": "Sol Ring",
            "typeLine": "Artifact",
            "cmc": 1,
            "price": 2.50,
        },
        {
            "deckId": "test-deck-1",
            "cardName": "Arcane Signet",
            "typeLine": "Artifact",
            "cmc": 2,
            "price": 0.80,
        },
        {
            "deckId": "test-deck-1",
            "cardName": "Command Tower",
            "typeLine": "Land",
            "cmc": 0,
            "price": 0.50,
        },
        {
            "deckId": "test-deck-2",
            "cardName": "Sol Ring",
            "typeLine": "Artifact",
            "cmc": 1,
            "price": 2.50,
        },
        {
            "deckId": "test-deck-2",
            "cardName": "Cultivate",
            "typeLine": "Sorcery",
            "cmc": 3,
            "price": 1.00,
        },
    ]

    return {"decks": decks, "cards": cards}


# ---------- Scryfall Tagger Endpoint ----------

@app.get("/tags/scryfall")
def get_scryfall_tags(
    card_name: str = Query(..., alias="cardName"),
) -> Dict[str, Any]:
    """
    TEMP implementation: simple stub you can replace with real Scryfall Tagger scraping.
    """
    # Fake some tags so your Tag Editor can test the call
    tags: List[str] = []
    lower = card_name.lower()
    if lower in {"sol ring", "arcane signet"}:
        tags = ["Ramp", "Mana Rock"]
    elif lower == "command tower":
        tags = ["Land", "Fixing"]

    return {"cardName": card_name, "tags": tags}
