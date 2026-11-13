# tests/test_feeds_base.py
import datetime as dt

import pytest
from betterybalance.core.feeds import base


def test_daycache_roundtrip_last_write_wins(tmp_path):
    c = base.DayCache(tmp_path / "x.jsonl")
    d = dt.date(2026, 10, 1)
    assert c.get(d) is None
    c.put(d, [{"a": 1}])
    c.put(d, [{"a": 2}])
    assert c.get(d) == [{"a": 2}] and c.days() == [d]

def test_http_get_raises_feederror(monkeypatch):
    def boom(*a, **k): raise OSError("down")
    monkeypatch.setattr(base.urllib.request, "urlopen", boom)
    with pytest.raises(base.FeedError):
        base.http_get("https://example.invalid/")

def test_local_days_inclusive():
    assert len(base.local_days(dt.date(2026, 10, 30), dt.date(2026, 11, 1))) == 3


def test_http_get_sends_honest_user_agent_and_caller_can_override(monkeypatch):
    seen = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def fake(request, timeout=None):
        seen.append(request)
        return Resp()

    monkeypatch.setattr(base.urllib.request, "urlopen", fake)
    base.http_get("https://example.invalid/")
    assert seen[0].get_header("User-agent") == base.USER_AGENT
    base.http_get("https://example.invalid/", headers={"User-Agent": "custom/1"})
    assert seen[1].get_header("User-agent") == "custom/1"
