from tools.study import report as R


def _study(**over):
    s = {
        "test_start": "2027-03-01",
        "test_end": "2027-03-15",
        "test_last_day": "2027-03-14",
        "month_end": "2027-04-01",
        "partial": True,
        "train_days": 200,
        "refit_every_days": 7,
        "tiers": ["A", "B", "C", "D"],
        "horizons": [1, 2],
        "folds": [{}, {}],
        "facts": {
            "master_first_day": "2026-08-01",
            "master_last_day": "2027-03-20",
            "label_last_day": "2027-03-16",
            "label_missing_from": "2027-03-17",
            "live_first_day": "2027-01-05",
            "train_start_day": "2026-08-13",
            "train_days_per_tier": {
                "A": {"min": 200.0, "max": 200.0},
                "B": {"min": 200.0, "max": 200.0},
                "C": {"min": 21.0, "max": 28.0},
                "D": {"min": 20.5, "max": 28.0},
            },
        },
        "ida": {"first_day": "2026-08-01", "train_start_day": "2026-08-13", "clipped": False},
        "rules": {"load_bearing_gain": 0.05, "adopt_min_gain": 0.05, "min_rows": 50},
        "results": {
            "A": {1: {"buy": {"mae_q50": 50.0, "n": 10}}, 2: {"buy": {"mae_q50": 50.0}}},
            "naive_da": {1: {"buy": {"mae": 52.0}}, 2: {"buy": {"mae": 51.0}}},
            "baseline": {1: {"buy": {"mae_q50": 55.0}}, 2: {"buy": {"mae_q50": 50.5}}},
        },
        "reading": {},
        "decay": {},
        "headroom": None,
        "gap_test": {"buy": {"mean": None, "median": None, "n": 0}},
    }
    s.update(over)
    return s


def test_training_sentences_read_dates_and_weeks_from_the_data():
    text = " ".join(R.training_sentences(_study()))
    assert "Fold 0 trains from 2026-08-13" in text and "200-day window by 7 days" in text
    assert "starts on 2027-01-05" in text and "2,9 to 4,0 weeks" in text
    assert "28,6 weeks for tiers A and B" in text and "conservative" in text


def test_label_ida_and_optimism_sentences():
    s = _study()
    assert "up to 2027-03-16" in R.label_sentence(s)
    assert "From 2027-03-17 to 2027-03-20" in R.label_sentence(s)
    s["facts"]["label_missing_from"] = None
    assert "From " not in R.label_sentence(s)
    assert "no training window shrinks" in R.ida_sentence(s)
    clipped = _study(
        ida={"first_day": "2026-09-01", "train_start_day": "2026-08-13", "clipped": True}
    )
    assert "starts on 2026-09-01, after the fold-0 train start 2026-08-13" in R.ida_sentence(
        clipped
    )
    opt = R.optimism_sentence(s)
    assert (
        "one day in 200 for tiers A and B" in opt and "one day in 20 to 28 for tiers C and D" in opt
    )


def test_baseline_sentence_and_window_lines():
    s = _study()
    b = R.baseline_sentence(s)
    assert "at 1 of 2 horizons" in b and "tier A beats naive day-ahead by only 2,0 EUR/MWh" in b
    lines = "\n".join(R.window_lines(s))
    assert "after 2027-04-01 writes docs/reports/2027-04-01-march-price-study.md" in lines
    assert "2026-09-02" not in lines and "2025-10-01" not in lines


def test_conclusion_last_sentence_uses_the_run():
    last = R.conclusion(_study())[-1]
    assert "2 weekly folds of March 2027" in last and "at most 4,0 weeks" in last


def test_restore_keys_round_trip():
    assert R.restore_keys({"A": {"0": {"1": 2}, "-1": [{"3": 4}]}}) == {
        "A": {0: {1: 2}, -1: [{3: 4}]}
    }


def test_rule_sentence_names_a_flat_change():
    r = {"ref": "C", "mean": -0.0004, "load_bearing": False}
    assert "no change in buy q50 MAE (below 0,1 %)" in R.rule_sentence("D", r)
    r = {"ref": "B", "mean": 0.096, "load_bearing": True}
    assert R.rule_sentence("C", r).endswith("9,6 % lower buy q50 MAE, load-bearing yes.")
