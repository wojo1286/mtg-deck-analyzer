import streamlit as st
import subprocess
import sys
import os
from playwright.sync_api import sync_playwright

# -------------------------------------------------------------------
# 1. CACHED INSTALLER FOR PLAYWRIGHT
# -------------------------------------------------------------------
@st.cache_resource
def setup_playwright():
    """
    Ensures that the Playwright Chromium browser and OS-level dependencies
    are installed within the container at runtime. Only runs once per container.
    """
    st.write("Verifying and (if needed) installing Playwright dependencies...")

    try:
        # Run installation command for Playwright + OS deps
        command = [
            sys.executable,
            "-m",
            "playwright",
            "install",
            "--with-deps",
            "chromium"
        ]

        with st.spinner("Installing Chromium browser (this only runs on first boot)..."):
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # Allow ample time for first-time install
            )

        st.success("Playwright environment is ready.")
        with st.expander("Show installation logs"):
            st.code(process.stdout)

    except subprocess.CalledProcessError as e:
        st.error("Failed to install Playwright dependencies. The application cannot continue.")
        st.error(f"Error: {e}")
        st.code(e.stderr if hasattr(e, 'stderr') else "No stderr output.")
        st.stop()

    except subprocess.TimeoutExpired:
        st.error("Playwright installation timed out. Please try again or check your connection.")
        st.stop()

    return True


# -------------------------------------------------------------------
# 2. SIMPLE TEST CASE (REPLACE THIS WITH YOUR APP LOGIC)
# -------------------------------------------------------------------
def run_test_scraper():
    """Minimal test to verify Chromium install and scraping works."""
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        except Exception as e:
            st.warning("First launch failed, trying to reinstall Playwright browser binaries...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=False)
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])

        page = browser.new_page()
        page.goto("https://example.com")
        html = page.content()
        st.write("✅ Page loaded successfully. Title:", page.title())
        browser.close()


# -------------------------------------------------------------------
# 3. STREAMLIT UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(layout="centered", page_title="Playwright Test App")
    st.title("Playwright Chromium Test on Streamlit Cloud")

    st.write("This app verifies that Playwright can run inside Streamlit Cloud.")
    if st.button("🚀 Run Test Scrape" ):
        run_test_scraper()


# -------------------------------------------------------------------
# 4. ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    if setup_playwright():
        main()
