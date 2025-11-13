# data/staples.py
from __future__ import annotations

from pathlib import Path
import json

import streamlit as st


@st.cache_data(show_spinner=False)
def load_ci_staples(ci_key: str | None) -> set[str]:
    """Load color-identity staples defined in data/ci_staples.json."""
    if not ci_key:
        return set()

    path = Path(__file__).with_name("ci_staples.json")
    if not path.exists():
        return set()

    data = json.loads(path.read_text("utf-8"))
    names = data.get(ci_key, [])
    return {str(n).strip() for n in names}
