# MTG Deck Analyzer – Development Roadmap & AI Guide

This document is for **developers and AI coding assistants** (Codex, ChatGPT, etc.).  
It describes how the project is structured, what we’re building next, and how to make
changes safely.

Users of the app should read **`README.md`** instead.

---

## 1. Project Overview

The MTG Deck Analyzer is a **Streamlit-based Commander (EDH) analysis and deckbuilding tool**.

High-level capabilities:

- Scrape decklists from **EDHREC** using Playwright.
- Parse EDHREC HTML into normalized card tables.
- Enrich cards with types, roles/tags (Ramp, Draw, Removal, etc.), and prices.
- Provide interactive analysis: popularity, mana curve, co-occurrence, synergy.
- Generate **average** or **constrained** decklists based on targets and tags.
- Export generated decks for use on other platforms.

Long-term, the app should also:

- Infer archetypes from data.
- Simulate opening hands and show odds.
- Offer AI-assisted upgrade / replacement suggestions.

---

## 2. Repository Layout (current)

All application code lives under `mtg_app/`:

```text
mtg-deck-analyzer/
│
├── mtg_app/
│   ├── app.py                # Streamlit entry point
│   │
│   ├── core/
│   │   ├── cache.py          # Playwright installation/cache helpers
│   │   └── constants.py      # Loads YAML config from constants.yaml
│   │
│   ├── analysis/
│   │   ├── deckgen.py        # Deck generation & summarization utilities
│   │   ├── stats.py          # Popularity, mana curve, type breakdown, etc.
│   │   └── synergy.py        # Co-occurrence & synergy metrics
│   │
│   ├── data/
│   │   ├── scraping.py       # Low-level scraping helpers (Playwright)
│   │   ├── decklists.py      # EDHREC deck metadata scraping
│   │   ├── parsing.py        # HTML → card table parsing/cleaning
│   │   └── tags.py           # Tag loading (GSheet/CSV), tag helpers
│   │
│   ├── ui/
│   │   └── dashboard.py      # Streamlit UI & visualization layout
│   │
│   └── tests/
│       └── test_pipeline.py  # Smoke test: scrape → parse → build
│
├── mtg_app/constants.yaml    # TYPE_KEYWORDS, default function categories, etc.
├── requirements.txt
└── README.md                 # User-facing docs

Other relevant files:

mtg_app/data/card_tags.csv – local fallback tag definitions.

.streamlit/secrets.toml – ignored by git, used only for local GSheet credentials.

The main development branch is currently refactor-v2.
AI assistants should create feature branches following this pattern:

codex/{short-feature-name}
# e.g. codex/fix-deck-summary, codex/land-logic, codex/add-budget-presets

3. Data Flow

The core pipeline is:

Scrape deck metadata
data/decklists.py / data/scraping.py use Playwright to pull deck URLs and
summary info from EDHREC (/decks/<commander-slug>).

Fetch & parse deck contents
deckpreview pages are fetched and parsed into card tables by data/parsing.py.

Clean & enrich card data
analysis/stats.py and helpers:

Normalize card names and types (via TYPE_KEYWORDS in constants.yaml).

Attach prices.

Attach functional categories (Ramp, Card Advantage, etc.) from tags.

Analyze

Popularity / inclusion rates.

Mana curve (spells only).

Type breakdown per deck and on average.

Co-occurrence and synergy metrics.

Generate decks (analysis/deckgen.py)

Average decks based on inclusion frequency + land rules.

Constrained decks that hit type and function targets.

Render UI (ui/dashboard.py + app.py)

Sidebar: commander, scrape controls, presets.

Tabs/expanders: parsed cards, popularity, synergy, deck generator, summaries.

4. Phase Roadmap

The roadmap is organized into phases. We are currently focusing on Phase 1.

Checkboxes are for humans and AI assistants to mark progress in PR descriptions.

Phase 1 – Foundation & Stability (current)

Goal: Correctness, robustness, and predictable behavior before building more features.

1.1 Configuration & constants

 Modularize into mtg_app/ with core/, data/, analysis/, ui/, tests/.

 Move constants into constants.yaml, loaded via core/constants.py.

 Eliminate hardcoded type/function tags:

_extract_primary_type() and related logic must use TYPE_KEYWORDS.

Functional categories must be driven by DEFAULT_FUNCTIONAL_CATEGORIES.

1.2 Tags & Google Sheets

 Finish safe_read_gsheet() and safe_update_gsheet() in data/tags.py:

Wrap all GSheet access in these helpers.

Gracefully handle missing or malformed st.secrets (gsheets block).

Do not crash Streamlit if secrets are missing; instead, fall back to CSV.

Add clear logging / st.warning messages on failure.

 Ensure tag loading behavior is consistent:

Use GSheet only if available and valid.

Otherwise use card_tags.csv as local fallback.

Treat category as a pipe-separated list of tags ("Ramp|Mana Rock").

1.3 Playwright & scraping stability

 Make Playwright setup robust:

Use core/cache.ensure_playwright() for install.

Avoid stale browser objects across reruns (fix greenlet.error).

When a browser/context fails, close and recreate it safely.

 Support multi-page EDHREC deck metadata scraping:

Iterate p=1,2,3... with page_size=100 until max_decks or no more rows.

Handle cases where page size controls differ (button, link, etc.).

1.4 Streamlit state & reruns

 Audit and wrap st.rerun() calls:

Only rerun when the user explicitly performs an action (e.g., Fetch decks, Clear data).

Use st.session_state flags to avoid infinite rerun loops.

 Ensure the “Clear data” button resets:

Cached scrape/deck data.

Any session_state keys dependent on the current commander.

1.5 Deck generation & summary correctness

 Fix summarize_deck to use the actual generated deck list:

Build a DataFrame for the 99/100 cards in the generated deck.

Join on a deduplicated card metadata table.

Recompute:

Counts by type.

Functions covered.

Land breakdown (basics vs nonbasics).

Total estimated price.

 Implement explicit land handling in generate_average_deck:

Choose a land target (e.g., 36–38 for 100-card decks).

Derive nonbasic lands from the scraped meta when present.

Fill remaining land slots with basics based on color identity.

Ensure the final list actually includes the claimed number of lands.

 Verify that functions covered and tag-level synergy rely on each card’s own tags, not global defaults that accidentally tag the whole pool as Ramp/Mana Rock.

1.6 Tests & tooling

 Extend tests/test_pipeline.py:

Include sanity checks for types, land counts, and tag application.

Add regression tests for generate_average_deck and summarize_deck.

 Ensure python -m compileall . and pytest both pass before merging.

Phase 2 – Performance & UX

Goal: Make the app feel fast and pleasant to use on typical EDHREC deck counts.

2.1 Caching & memoization

 Use @st.cache_data for pure, expensive computations:

Popularity/inclusion tables.

Co-occurrence matrix.

Any aggregate stats that depend only on scraped card data.

 Use @st.cache_resource for shared resources when appropriate.

2.2 Vectorization & performance

 Replace row-wise .apply() where possible:

_extract_primary_type() → compiled regex over df["type"].

Tag and function classification via merges or .isin() rather than repeated string work.

 Optimize _fill_deck_slots():

Precompute lookups for card type, tags, and price.

Avoid recomputing filters each time a slot is filled.

2.3 UX polish

 Add friendly empty-state messages:

“No decks scraped yet – click ‘Fetch decks’ first.”

“No tags found; using built-in defaults.”

“Filters removed all cards; try lowering constraints.”

 Ensure error messages from scraping and GSheets are human-readable and visible in the UI.

Phase 3 – Data Enrichment & Insights

Goal: Provide richer guidance beyond basic stats.

 Refine card synergy and co-occurrence tools:

Allow filtering by minimum decks, card type, or tag.

Surface “synergy outliers” (cards that perform above/below expectation).

 Implement budget filter presets:

e.g., “Budget,” “Midrange,” “High-end” radio buttons that map to price caps.

 Add replacement suggestions:

For each card, suggest cheaper or more synergistic alternatives with similar tags.

UI view: “If you cut X, consider Y or Z.”

 Improve export options:

Copy-to-clipboard text.

Moxfield/Archidekt import formats (1 Card Name format).

Keep existing CSV export.

Phase 4 – Advanced Tools & Interactivity

Goal: Make the app a powerful interactive workbench for deck tuning.

 Add a searchable card browser (st.data_editor) over the candidate pool:

Filter by type, tag, price, inclusion rate.

Allow toggling “include/exclude” flags that feed back into deck generation.

 Add interactive tags and Scryfall links:

Hover or click on a card to see:

Scryfall link.

Tags/functional roles.

Basic stats (deck count, avg price).

 Implement raw commander input with fuzzy slugification:

Allow users to type a commander name and resolve it to an EDHREC slug.

Let the user override if multiple matches are found.

 Add a live “slots remaining” tracker during constrained deck generation:

Show progress towards target ranges:

e.g., “Ramp 7 / (8–12), Draw 4 / (6–10), Removal 5 / (6–10), Creatures 28 / (20–32).”

Phase 5 – Intelligent Features

Goal: Layer in higher-level understanding and AI assistance.

 Archetype inference:

Use Jaccard/cosine similarity on cards and tags to cluster decks into archetypes.

Show archetype labels and how close the current build is to each.

 Simulated opening hands:

Monte Carlo simulation of draws to show:

Land count probabilities.

Chance to have ramp/draw by turn X.

 AI-assisted suggestions:

Use co-occurrence + tags as the data backbone.

Use an LLM (e.g., OpenAI model) to:

Suggest upgrades within a target budget.

Explain reasoning in natural language.

5. Guidelines for AI Assistants (Codex / ChatGPT)

When modifying this repo:

Respect boundaries

Do not commit or invent real secrets. Never hardcode credentials.

Assume .streamlit/secrets.toml is local-only; never modify it.

Use existing abstractions

Use core/constants.py instead of repeating constants.

For Google Sheets, always use safe_read_gsheet() / safe_update_gsheet().

For scraping, go through helpers in data/scraping.py and data/decklists.py.

Keep changes focused

Create small, cohesive PRs tied to a specific roadmap task.

Update or add tests when behavior changes.

Code style

Prefer clear, modular functions over huge monoliths.

Use type hints (-> pd.DataFrame, etc.) where practical.

Keep docstrings updated, especially around deck generation and tag logic.

Validate

Before merging, ensure:

python -m compileall . passes.

pytest passes (or newly added tests are green).

This document is the single source of truth for project direction.
If implementing new features, align them with the phases and tasks defined here.
