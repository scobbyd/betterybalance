import pathlib

from betterybalance.core.feeds import tennet_settlement as ts

FIX = (pathlib.Path(__file__).parent / "fixtures" / "tennet_settlement_fragment.html").read_text()


def test_parse_fragment_keeps_all_rows_and_states():
    rows = ts.parse_day_fragment(FIX)
    assert len(rows) == 8
    states = {r["regulation_state"] for r in rows}
    assert 2 in states
    inverted = [r for r in rows if r["buy_eur_mwh"] < r["sell_eur_mwh"]]
    assert len(inverted) == 1  # survives, never asserted away
    assert all(r["ts"] % 900 == 0 for r in rows)


def test_fetch_settlement_pages_caches_and_aligns(tmp_path, monkeypatch):
    import datetime as dt

    import numpy as np
    from betterybalance.core.feeds import base

    page = (
        '<input type="hidden" id="pInstance" value="123">'
        '<input type="hidden" id="pSalt" value="9">'
        '<input type="hidden" id="pPageItemsProtected" value="abc">'
        "apex.widget.report.paginate('42', 'AJAXID'"
    )
    posts = []

    def fake_get(url, **k):
        return page.encode()

    def fake_post(url, data, **k):
        posts.append(data)
        return FIX.encode()

    monkeypatch.setattr(ts, "http_get", fake_get)
    monkeypatch.setattr(ts, "http_post", fake_post)
    monkeypatch.setattr(ts, "PACE_S", 0.0)
    cache = base.DayCache(tmp_path / "s.jsonl")
    day = dt.date(2026, 10, 28)
    t = ts.fetch_settlement(day, day, cache)
    assert len(posts) == 1 and len(t["ts"]) == 96
    assert int(np.isfinite(t["buy_eur_mwh"]).sum()) == 8
    assert np.isnan(t["dispatch_up_eur_mwh"][0]) and t["regulation_state"][1] == 2.0
    again = ts.fetch_settlement(day, day, cache, offline=True)
    assert len(posts) == 1
    assert np.array_equal(t["buy_eur_mwh"], again["buy_eur_mwh"], equal_nan=True)


def test_malformed_cell_raises_feederror():
    import pytest
    from betterybalance.core.feeds.base import FeedError

    bad = FIX.replace('headers="PRICE_SHORTAGE">180<', 'headers="PRICE_SHORTAGE">n/a<', 1)
    assert bad != FIX
    with pytest.raises(FeedError, match="2026-10-28"):
        ts.parse_day_fragment(bad)
    with pytest.raises(FeedError):
        ts.parse_day_fragment(FIX.replace('headers="PTU">1<', 'headers="PTU"><', 1))
    inf = FIX.replace('headers="REGULATION_STATE">-1<', 'headers="REGULATION_STATE">inf<', 1)
    assert inf != FIX
    with pytest.raises(FeedError):
        ts.parse_day_fragment(inf)


def test_fetch_settlement_survives_one_malformed_day(tmp_path, monkeypatch):
    import datetime as dt

    import numpy as np
    from betterybalance.core.feeds import base

    page = (
        '<input type="hidden" id="pInstance" value="1">'
        '<input type="hidden" id="pSalt" value="9">'
        '<input type="hidden" id="pPageItemsProtected" value="a">'
        "apex.widget.report.paginate('42', 'AJAXID'"
    )
    bad = FIX.replace('headers="PRICE_SHORTAGE">180<', 'headers="PRICE_SHORTAGE">n/a<', 1)

    def fake_post(url, data, **k):
        return bad.encode() if b"27-10-2026" in data else FIX.encode()

    monkeypatch.setattr(ts, "http_get", lambda url, **k: page.encode())
    monkeypatch.setattr(ts, "http_post", fake_post)
    monkeypatch.setattr(ts, "PACE_S", 0.0)
    cache = base.DayCache(tmp_path / "s.jsonl")
    t = ts.fetch_settlement(dt.date(2026, 10, 27), dt.date(2026, 10, 28), cache)
    assert len(t["ts"]) == 192
    assert np.isnan(t["buy_eur_mwh"][:96]).all()
    assert int(np.isfinite(t["buy_eur_mwh"][96:]).sum()) == 8
    assert cache.get(dt.date(2026, 10, 27)) is None


def test_adapter_sends_no_user_agent_of_its_own():
    import inspect

    from betterybalance.core.feeds import tennet_settlement as ts

    source = inspect.getsource(ts)
    assert "User-Agent" not in source and "Mozilla" not in source
