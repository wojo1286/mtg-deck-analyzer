import subprocess, sys
import streamlit as st
from pathlib import Path

@st.cache_resource
def ensure_playwright():
    chromium_path = Path.home() / ".cache/ms-playwright/chromium"
    if chromium_path.exists():
        return True
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True
        )
        return True
    except Exception as e:
        st.error(f"Playwright installation failed: {e}")
        st.stop()
