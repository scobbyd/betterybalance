import datetime as dt
import importlib

fh = importlib.import_module("tools.fetch_history")


def test_plan_requests_months_and_days():
    plan = fh.plan_requests(dt.date(2025, 12, 20), dt.date(2026, 1, 5), only=("da", "ida"))
    assert ("da", dt.date(2025, 12, 1), dt.date(2025, 12, 31)) in plan
    assert sum(1 for p in plan if p[0] == "ida_IDA3") == 17
