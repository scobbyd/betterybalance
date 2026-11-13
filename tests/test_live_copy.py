# tests/test_live_copy.py (moved from Task 17: state_view needs live/copy.py in this task)
from betterybalance.live import copy as CP


def test_every_reason_and_flag_has_both_languages():
    for lang in CP.LANGS:
        assert all(CP.verdict(r, "hold", 0.0, 18.4, None, lang) for r in CP.REASONS)
        assert all(CP.status_line(f, lang) for f in CP.FLAGS)
    assert CP.status_line("pv_drift", "en") == "PV forecast off by more than 15 % this week"


def test_verdict_lines():
    assert (
        CP.verdict("commit", "discharge", -9.6, 18.4, None, "en")
        == "Selling 9,6 kW this quarter: the price clears 18,40 ct/kWh."
    )
    assert (
        CP.verdict("commit", "discharge", -9.6, 18.4, None, "nl")
        == "Verkoopt 9,6 kW dit kwartier: de prijs ligt boven 18,40 ct/kWh."
    )


def test_unknown_language_falls_back_to_english_and_windows_read_amsterdam_time():
    line = CP.verdict("commit", "discharge", -9.6, 18.4, None, "de")
    assert line == CP.verdict("commit", "discharge", -9.6, 18.4, None, "en")
    # 2026-11-02 01:00 UTC is 02:00 in Amsterdam (CET)
    window = {"start": 1793581200, "end": 1793581200 + 9000, "side": "charge"}
    assert "02:00" in CP.verdict("wait", "hold", 0.0, 18.4, window, "en")
    assert "04:30" in CP.verdict("wait", "hold", 0.0, 18.4, window, "nl")


def test_copy_has_no_em_dash():
    for lang in CP.LANGS:
        for r in CP.REASONS:
            for action, kw in (("hold", 0.0), ("charge", 3.0), ("discharge", -3.0)):
                assert "—" not in CP.verdict(r, action, kw, 18.4, None, lang)
        for f in CP.FLAGS:
            assert "—" not in CP.status_line(f, lang)


def test_every_site_flag_has_a_status_line_in_both_languages():
    from betterybalance.live import site as LS

    assert LS.FLAGS is CP.FLAGS
    for lang in CP.LANGS:
        assert all(CP.status_line(f, lang) for f in LS.FLAGS)
