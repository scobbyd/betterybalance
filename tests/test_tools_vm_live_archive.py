from tools.vm_live_archive import start_labels


def test_start_labels_shift_each_window_end_to_its_minute_start():
    # avg_over_time(...[1m]) at T covers (T - 60, T]: the minute that starts at T - 60
    body = {
        "data": {
            "result": [{"metric": {}, "values": [[1_800_000_060, "10.5"], [1_800_000_120.0, "11"]]}]
        }
    }
    assert start_labels(body) == {1_800_000_000: 10.5, 1_800_000_060: 11.0}
    assert start_labels({"data": {"result": []}}) == {}


def test_to_is_inclusive_local_day():
    import datetime as dt

    from tools.vm_live_archive import ZONE, window

    start, end = window("2026-10-31", "2026-10-31")
    assert end - start == 86400
    assert end == int(dt.datetime(2026, 11, 1, tzinfo=ZONE).timestamp())
    # a two-day window names both days; the clock-change day is 25 h long
    start, end = window("2026-10-24", "2026-10-25")
    assert end - start == 86400 + 90000
