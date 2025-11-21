import pandas as pd
import pytest
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from analysis.synergy import card_tag_synergy
from data import tags


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_load_tags_from_gsheet_handles_missing_secrets(monkeypatch):
    def _raise_missing():
        raise StreamlitSecretNotFoundError("secrets missing")

    monkeypatch.setattr(tags, "_sheet_id", _raise_missing)

    df = tags.load_tags_from_gsheet()
    assert df.empty
    assert list(df.columns) == ["name", "category"]
    assert tags.has_gsheet_categories() is False


def test_load_tag_categories_prefers_gsheet(monkeypatch):
    sheet_df = pd.DataFrame({"name": ["Sol Ring", "Sol Ring"], "category": ["Ramp", "Mana Rock"]})
    csv_df = pd.DataFrame({"name": ["Arcane Signet"], "category": ["Ramp"]})

    monkeypatch.setattr(tags, "load_tags_from_gsheet", lambda worksheet="Categories": sheet_df)
    monkeypatch.setattr(tags, "load_card_tags", lambda: csv_df)

    result = tags.load_tag_categories()
    assert set(result["name"]) == {"Sol Ring"}
    assert result.loc[result["name"] == "Sol Ring", "category"].iloc[0] == "Ramp|Mana Rock"


def test_load_tag_categories_falls_back_to_csv(monkeypatch):
    monkeypatch.setattr(tags, "load_tags_from_gsheet", lambda worksheet="Categories": pd.DataFrame())
    csv_df = pd.DataFrame({"name": ["Cultivate"], "category": ["Ramp|Fixing"]})
    monkeypatch.setattr(tags, "load_card_tags", lambda: csv_df)

    result = tags.load_tag_categories()
    assert result.equals(tags.normalize_tags_df(csv_df))


def test_card_tag_synergy_filters_to_current_cards():
    df = pd.DataFrame(
        {
            "deck_id": ["d1", "d1", "d2"],
            "name": ["Sol Ring", "Beast Within", "Sol Ring"],
            "category": ["Ramp|Mana Rock", "Removal", "Ramp|Mana Rock"],
        }
    )

    synergy_df = card_tag_synergy(df)

    assert set(synergy_df["tag"]) == {"Ramp", "Mana Rock", "Removal"}

    ramp_row = synergy_df[(synergy_df["name"] == "Sol Ring") & (synergy_df["tag"] == "Ramp")].iloc[0]
    assert pytest.approx(ramp_row["p_tag"], rel=1e-6) == 1.0
    assert pytest.approx(ramp_row["p_tag_given"], rel=1e-6) == 1.0

    removal_row = synergy_df[(synergy_df["name"] == "Beast Within") & (synergy_df["tag"] == "Removal")].iloc[0]
    assert pytest.approx(removal_row["p_tag"], rel=1e-6) == 0.5
    assert pytest.approx(removal_row["delta"], rel=1e-6) == 0.5
