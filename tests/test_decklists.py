import pandas as pd

from data.decklists import scrape_edhrec_decks_for_commander
from data.parsing import parse_table


SAMPLE_HTML = """
<html>
  <body>
    <a href="https://moxfield.com/decks/example">View source</a>
    <table>
      <tr><th>Name</th><th>Type</th><th>CMC</th><th>Price</th></tr>
      <tr><td><a>Sol Ring</a></td><td>Artifact</td><td>1</td><td>$1.00</td></tr>
      <tr><td><a>Forest</a></td><td>Basic Land — Forest</td><td></td><td>$0.10</td></tr>
    </table>
  </body>
</html>
"""


def test_parse_table_extracts_fields():
    rows = parse_table(SAMPLE_HTML, deck_id="deck123", deck_source="https://moxfield.com/decks/example")
    assert len(rows) == 2
    df = pd.DataFrame(rows)
    assert set(df.columns) >= {"deck_id", "deck_source", "name", "type", "cmc", "price"}
    sol_ring = df[df["name"] == "Sol Ring"].iloc[0]
    assert sol_ring["type"] == "Artifact"
    assert sol_ring["cmc"] == "1"


def test_scrape_edhrec_decks_for_commander_uses_metadata(monkeypatch):
    meta_payload = {
        "container": {"json_dict": {"card": {"color_identity": ["U", "R"]}}},
        "table": [
            {"urlhash": "abc123", "name": "Deck One"},
            {"urlhash": "def456", "name": "Deck Two"},
        ],
    }

    class FakeResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, *args, **kwargs):
        if "commanders" in url:
            return FakeResponse({"container": {"json_dict": {"card": {"color_identity": ["G"]}}}})
        return FakeResponse(meta_payload)

    monkeypatch.setattr("data.decklists.requests.get", fake_get)

    df, colors = scrape_edhrec_decks_for_commander(
        "test-commander",
        deck_limit=2,
        _html_fetcher=lambda _: SAMPLE_HTML,
    )

    assert colors == ["U", "R"]
    assert not df.empty
    assert set(df.columns) >= {"deck_id", "deck_source", "name", "type", "cmc", "price"}
    assert df["deck_id"].nunique() == 2
