# ===================================================================
# 1. SETUP & IMPORTS
# ===================================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import warnings
import random
from copy import deepcopy
import time
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import subprocess
import sys

# --- Visualization ---
import plotly.express as px
import plotly.graph_objects as go

# --- Advanced Analytics ---
from mlxtend.frequent_patterns import apriori
from sklearn.manifold import TSNE

# --- Page Config ---
st.set_page_config(layout="wide", page_title="MTG Deckbuilding Analysis Tool")

# ===================================================================
# 2. PLAYWRIGHT INSTALLATION (RUNS ONCE PER CONTAINER START)
# ===================================================================

@st.cache_resource
def setup_playwright():
    """Ensures Playwright Chromium is installed without using sudo or --with-deps."""
    st.write("Verifying and (if needed) installing Playwright dependencies...")
    try:
        command = [sys.executable, "-m", "playwright", "install", "chromium"]
        with st.spinner("Installing Chromium browser (first time only)..."):
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=600
            )
        st.success("Playwright environment is ready.")
        with st.expander("Show installation logs"):
            st.code(process.stdout)
    except subprocess.CalledProcessError as e:
        st.error("Failed to install Playwright dependencies. The application cannot continue.")
        st.error(f"Error: {e}")
        st.code(e.stderr if hasattr(e, 'stderr') else "No stderr output.")
        st.stop()
    return True

setup_complete = setup_playwright()

# ===================================================================
# 3. TEST SCRAPER FOR DEPLOYMENT VERIFICATION
# ===================================================================

def run_test_scraper():
    """Test function to verify Playwright + Chromium works in Streamlit Cloud."""
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        except Exception:
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])

        page = browser.new_page()
        page.goto("https://example.com")
        html = page.content()
        st.write("✅ Page loaded successfully. Title:", page.title())
        browser.close()

# ===================================================================
# 4. STREAMLIT UI
# ===================================================================

def main():
    st.title("Playwright Chromium Test on Streamlit Cloud")

    st.write("This app verifies that Playwright can run inside Streamlit Cloud.")

    if st.button("🚀 Run Test Scrape"):
        run_test_scraper()

if __name__ == "__main__":
    if setup_complete:
        main()
