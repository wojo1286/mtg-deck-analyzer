import streamlit as st
from core.cache import ensure_playwright
from core.config import DEFAULT_FUNCTIONAL_CATEGORIES

st.set_page_config(layout="wide", page_title="MTG Deckbuilding Analysis Tool")

if ensure_playwright():
    st.title("MTG Deckbuilding Analysis Tool - Modular Version")
    st.write("✅ Core structure initialized successfully.")
    st.write("Default Functional Categories:")
    st.write(DEFAULT_FUNCTIONAL_CATEGORIES)
else:
    st.error("Playwright setup failed.")
