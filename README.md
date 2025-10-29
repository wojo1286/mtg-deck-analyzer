# MTG Deck Analyzer 🧙‍♂️🃏

A full-stack Python application that scrapes Magic: The Gathering Commander decklists from EDHREC, parses and transforms card data into a normalized format, and provides interactive visual analysis through a Streamlit dashboard. Built for MTG players looking to explore deck archetypes, card synergies, and pricing trends across formats.
---

## 🌐 Features

- 🔍 **Decklist Scraping** via Playwright (EDHREC HTML)
- 🧠 **Card Parsing** with fallback logic for non-table formats
- 🗃 **Deck Building Engine** to filter, enrich, and consolidate scraped card data
- 📊 **Interactive Dashboard** powered by Streamlit + Plotly
- 🧪 **Test Coverage** for all scrape → parse → build components
- ⚙️ **Modular Architecture** with reusable constants and configuration

---

## 📁 Project Structure

mtg-deck-analyzer/
│
├── analysis/ # Deck generation and scoring logic
│ └── deckgen.py
│
├── core/ # Caching, config, constants
│ ├── cache.py
│ ├── constants.py # Centralized constant definitions
│ └── config.yaml # Easily editable tuning values
│
├── data/ # Scraping and parsing logic
│ ├── decklists.py # EDHREC scraping logic
│ └── parsing.py # HTML parsing to card list
│
├── tests/ # Pytest tests
│ ├── test_scrape.py
│ ├── test_parse.py
│ └── test_deckgen.py
│
├── ui/ # Streamlit front-end
│ └── dashboard.py
│
├── app.py # Entry point for Streamlit
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourname/mtg-deck-analyzer.git
cd mtg-deck-analyzer
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Install Playwright Browsers
bash
Copy code
playwright install
4. Run the App
bash
Copy code
streamlit run app.py
🧪 Run Tests
bash
Copy code
pytest tests/
🔧 Configuration
All tuning parameters (e.g., rarity weights, synergy scores, color preferences) are stored in:

bash
Copy code
core/config.yaml
Update this file to change global behavior of deck generation and analysis.

📌 TODO (Phase 1)
 Finalize constants.py and config.yaml

 Write robust unit tests for all core modules

 Improve fallback parsing for atypical EDHREC decks

 Add test coverage for scraper failover logic

📄 License
MIT License

🧙‍♀️ Powered By
Python 3.11+

Streamlit

Pandas

Playwright

Plotly

Pytest

yaml
Copy code

---
