"""Markdown report of the October 2026 price study, rendered from the results JSON.

Every date, span and count in the prose comes from the results dict (see run_study in
tools/study/october_2026.py). The JSON round trip turns int keys into strings and NaN into
null: restore_keys undoes the first, and every number below goes through _num or eu.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
from betterybalance.core.evaluate import render_markdown
from betterybalance.core.features import ELAPSED_MIN
from betterybalance.core.fmt import eu

_TARGETS = ("buy", "sell")
_QNAMES = ("q10", "q50", "q90")
_DROP_HORIZONS = (0, 1, 4, 24)
_TIER_TEXT = {
    "A": "day-ahead and calendar",
    "B": "the intraday auctions IDA1 to IDA3",
    "C": "the live end-of-quarter prices of the last quarters",
    "D": "the live price at the start of the forecast quarter",
}


# --- JSON -----------------------------------------------------------------------------------


def restore_keys(x):
    """Turn digit-string dict keys back into ints, recursively."""
    if isinstance(x, dict):
        return {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): restore_keys(v)
            for k, v in x.items()
        }
    if isinstance(x, list):
        return [restore_keys(v) for v in x]
    return x


def load_results(path: Path) -> dict:
    return restore_keys(json.loads(Path(path).read_text()))


# --- helpers --------------------------------------------------------------------------------


def _num(x) -> float:
    return float("nan") if x is None else float(x)


def pct(x, dec: int = 1) -> str:
    x = _num(x)
    return eu(100.0 * x, dec) + " %" if math.isfinite(x) else eu(float("nan"))


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return lines


def _cell_node(results: dict, group: str, h: int, e: int | None) -> dict:
    node = results.get(group, {}).get(h, {})
    return node.get(e, {}) if h == 0 else node


def buy_mae(results: dict, tier: str, h: int, e: int | None = None) -> float:
    e = (e or ELAPSED_MIN[0]) if h == 0 else None
    return _num(_cell_node(results, tier, h, e).get("buy", {}).get("mae_q50"))


def _row_labels(horizons) -> list[tuple[str, int, int | None]]:
    rows = []
    for h in horizons:
        if h == 0:
            rows.extend((f"0 ({e} min)", 0, e) for e in ELAPSED_MIN)
        else:
            rows.append((str(h), h, None))
    return rows


def _weeks(days: float) -> str:
    return eu(days / 7.0, 1)


def _span(lo: float, hi: float, fmt) -> str:
    a, b = fmt(lo), fmt(hi)
    return a if a == b else f"{a} to {b}"


def _flat(m: float) -> bool:
    """A change that rounds to 0,0 % in the report."""
    return abs(m) < 0.0005


def rule_sentence(tier: str, r: dict) -> str:
    m = _num(r.get("mean"))
    if r.get("load_bearing") is None or not math.isfinite(m):
        return f"Tier {tier} over tier {r['ref']} at h1 to h4: no complete comparison, no verdict."
    verdict = "yes" if r["load_bearing"] else "no"
    if _flat(m):
        change = "no change in buy q50 MAE (below 0,1 %)"
    else:
        change = f"{pct(abs(m))} {'lower' if m >= 0 else 'higher'} buy q50 MAE"
    return f"Tier {tier} over tier {r['ref']} at h1 to h4: {change}, load-bearing {verdict}."


def _month(s: dict) -> str:
    return dt.date.fromisoformat(s["test_start"]).strftime("%B %Y")


# --- derived sentences ----------------------------------------------------------------------


def training_sentences(s: dict) -> list[str]:
    """Fold-0 train start, the live-history span of tiers C and D and the size gap (M8)."""
    f, tiers = s["facts"], s["tiers"]
    days = f["train_days_per_tier"]
    out = [
        f"Fold 0 trains from {f['train_start_day']}; each later fold slides the "
        f"{s['train_days']}-day window by {s['refit_every_days']} days."
    ]
    live = [t for t in ("C", "D") if t in tiers and days.get(t)]
    if live:
        lo = min(days[t]["min"] for t in live)
        hi = max(days[t]["max"] for t in live)
        ab = [days[t]["max"] for t in ("A", "B") if t in tiers and days.get(t)]
        tail = (
            f", against about {_weeks(max(ab))} weeks for tiers A and B. This gap in training size "
            "works against tiers C and D, so their passes are conservative."
            if ab
            else "."
        )
        out.append(
            f"Tiers {' and '.join(live)} need the live feed, whose history starts on "
            f"{f['live_first_day']}, so they train on {_span(lo, hi, _weeks)} weeks of it" + tail
        )
    return out


IDA_NOTE = " The IDA history itself begins 2024-06-14. The study sees only the master table."


def ida_sentence(s: dict) -> str:
    ida = s["ida"]
    if ida["first_day"] is None:
        return "No IDA value exists in the data; tier B has nothing to add."
    if ida["clipped"]:
        return (
            f"The master table starts on {ida['first_day']}, after the fold-0 train start "
            f"{ida['train_start_day']}, so the study shrinks the training window of every tier to "
            "start there: tiers B to D cannot learn IDA before it, and tier A shrinks with them so "
            "that the B over A comparison stays on the same training span." + IDA_NOTE
        )
    return (
        f"The master table starts on {ida['first_day']}, no later than the fold-0 train start "
        f"{ida['train_start_day']}, so no training window shrinks." + IDA_NOTE
    )


def optimism_sentence(s: dict) -> str:
    days = s["facts"]["train_days_per_tier"]
    parts = []
    for group in (("A", "B"), ("C", "D")):
        have = [t for t in group if t in s["tiers"] and days.get(t)]
        if not have:
            continue
        lo = min(days[t]["min"] for t in have)
        hi = max(days[t]["max"] for t in have)
        span = _span(lo, hi, lambda d: eu(d, 0))
        parts.append(
            f"one day in {span} for tier{'s' if len(have) > 1 else ''} {' and '.join(have)}"
        )
    return (
        "Each refit trains on every label before the fold's first quarter, although TenneT "
        "publishes the last of those days only the next day. That is one day of optimism per "
        "refit: " + ", ".join(parts) + "."
    )


def label_sentence(s: dict) -> str:
    f = s["facts"]
    text = (
        f"Settled prices are labels only and exist up to {f['label_last_day']}. TenneT publishes "
        "settlement in bulk the next day, so no feature reads a settled price."
    )
    if f["label_missing_from"]:
        text += (
            f" From {f['label_missing_from']} to {f['master_last_day']} the master table has no "
            "settled price yet."
        )
    return text


def baseline_sentence(s: dict) -> str | None:
    """Name how the table baseline and tier A compare with naive day-ahead on buy (M9)."""
    res = s["results"]
    hs = [h for h in s["horizons"] if h >= 1]
    diffs = []
    for h in hs:
        b = _num(_cell_node(res, "baseline", h, None).get("buy", {}).get("mae_q50"))
        n = _num(_cell_node(res, "naive_da", h, None).get("buy", {}).get("mae"))
        if math.isfinite(b) and math.isfinite(n):
            diffs.append(b - n)
    if not diffs:
        return None
    worse = sum(d > 0 for d in diffs)
    where = "every horizon" if worse == len(diffs) else f"{worse} of {len(diffs)} horizons"
    text = (
        f"The day-ahead decile by hour-block table baseline loses to naive day-ahead on buy at "
        f"{where} ({eu(min(diffs), 1)} to {eu(max(diffs), 1)} EUR/MWh worse)"
    )
    a = buy_mae(res, "A", 1)
    n1 = _num(_cell_node(res, "naive_da", 1, None).get("buy", {}).get("mae"))
    if "A" in s["tiers"] and math.isfinite(a) and math.isfinite(n1):
        verb = "beats" if a < n1 else "loses to"
        text += f", and tier A {verb} naive day-ahead by only {eu(abs(n1 - a), 1)} EUR/MWh at h1"
    return text + ". This study names it and does not investigate why."


def conclusion(s: dict) -> list[str]:
    """At most ten plain sentences, every number and date taken from the results."""
    res, rd, tiers, rules = s["results"], s["reading"], s["tiers"], s["rules"]
    gate = pct(rules["load_bearing_gain"], 0)
    out = []
    n1 = _cell_node(res, "A", 1, None).get("buy", {}).get("n")
    out.append(
        f"Over {eu(_num(n1), 0)} quarter-hours between {s['test_start']} and "
        f"{s['test_last_day']}, a forecast from the day-ahead price and the calendar alone "
        f"(tier A) missed the settled buy price by {eu(buy_mae(res, 'A', 1), 1)} EUR/MWh on "
        "average, one quarter ahead."
    )
    for tier in ("B", "C", "D"):
        r = rd.get(tier)
        if r is None or r.get("load_bearing") is None:
            continue
        m = _num(r["mean"])
        if _flat(m):
            move = "left that error unchanged (below 0,1 %)"
        else:
            move = f"{'lowered' if m >= 0 else 'raised'} that error by {pct(abs(m))}"
        enough = "enough" if r["load_bearing"] else "not enough"
        out.append(
            f"Adding {_TIER_TEXT[tier]} (tier {tier}) {move} one to four quarters ahead compared "
            f"with tier {r['ref']}, {enough} to pass the {gate} rule."
        )
    if "D" in tiers and "C" in tiers and 0 in s["horizons"]:
        e = ELAPSED_MIN[-1]
        out.append(
            f"Inside the running quarter the live minutes count most: after {e} minutes tier D "
            f"misses by {eu(buy_mae(res, 'D', 0, e), 1)} EUR/MWh against "
            f"{eu(buy_mae(res, 'C', 0, e), 1)} for tier C."
        )
    best = tiers[-1]
    dec = s["decay"].get(best, {})
    hs = [h for h in dec if h >= 1]
    far = max((h for h in hs if h >= 4), default=None)
    if best != "A" and 1 in dec and far is not None:
        out.append(
            f"Fresh information fades with the horizon: tier {best} is {pct(dec[1])} better than "
            f"tier A one quarter ahead and {pct(dec[far])} at {far} quarters ahead."
        )
    ws = {
        g: _cell_node(res, g, 1, None).get("buy", {}).get("wrong_side")
        for g in ("A", best, "naive_persist")
    }
    if best != "A" and None not in ws.values():
        out.append(
            f"On direction, which is what a battery is paid for, tier {best} puts the buy price on "
            f"the wrong side of the day-ahead price in {pct(ws[best], 0)} of quarters one quarter "
            f"ahead, against {pct(ws['A'], 0)} for tier A and {pct(ws['naive_persist'], 0)} for "
            "simply repeating the running price."
        )
    hd = s["headroom"]
    if hd is not None and math.isfinite(_num(hd["mean_gain"])):
        g = _num(hd["mean_gain"])
        adopt_gate = pct(rules["adopt_min_gain"], 0)
        if hd["adopt"]:
            out.append(
                f"A gradient-boosted model beat the linear model by {pct(g)} on average at h0 to "
                f"h4, so it passes the {adopt_gate} adoption rule."
            )
        else:
            move = "lowered" if g >= 0 else "raised"
            out.append(
                f"A gradient-boosted model {move} the error by {pct(abs(g))} on average at h0 to "
                f"h4, short of the {adopt_gate} rule, so the simple linear model stays."
            )
    gap = s["gap_test"]["buy"]
    if gap["n"]:
        mean = _num(gap["mean"])
        out.append(
            "The live feed's last price of a quarter usually equals the settled price (median gap "
            f"{eu(_num(gap['median']), 1)} EUR/MWh) but sits {eu(abs(mean), 1)} EUR/MWh "
            f"{'below' if mean < 0 else 'above'} it on average, the cost of using the live "
            "feed as a stand-in for prices that settle the next day."
        )
    nf = len(s["folds"])
    days = s["facts"]["train_days_per_tier"]
    live = [days[t]["max"] for t in ("C", "D") if t in tiers and days.get(t)]
    history = (
        f" with tiers C and D trained on at most {_weeks(max(live))} weeks of live history"
        if live
        else ""
    )
    out.append(
        "These are price errors, not money (drop 2 measures what a battery earns), and the test "
        f"covers {nf} weekly fold{'s' if nf != 1 else ''} of {_month(s)}{history}, so the numbers "
        "can move as the month completes and history grows."
    )
    return out[:10]


# --- the report -----------------------------------------------------------------------------


def window_lines(s: dict) -> list[str]:
    """The bullets of the windows-and-data section."""
    arc = s.get("archive", {})
    timing = s.get("timing", {})
    lines = [
        f"- Test window: {s['test_start']} 00:00 to {s['test_last_day']} 24:00 local, "
        f"walk-forward with a refit every {s['refit_every_days']} days on the trailing "
        f"{s['train_days']} days.",
    ]
    if s.get("partial"):
        name = dt.date.fromisoformat(s["test_start"]).strftime("%B").lower()
        lines.append(
            f"- The month is not over. The full-month re-run after {s['month_end']} writes "
            f"docs/reports/{s['month_end']}-{name}-price-study.md."
        )
    lines += [
        f"- {label_sentence(s)}",
        f"- {' '.join(training_sentences(s))}",
        f"- {ida_sentence(s)} Unknown IDA values enter as zero plus a known-flag.",
        f"- Live archive: {eu(_num(arc.get('rows')), 0)} minutes from {arc.get('first')} to "
        f"{arc.get('last')}, {pct(arc.get('valid_share'))} with both prices, "
        f"{pct(arc.get('stale_share'), 2)} masked as a stalled page (unchanged for more than "
        "30 minutes).",
        "- Minute convention: a minute labelled t holds the samples taken in [t, t + 60 s) and is "
        "known at t + 60 s. So e_m, the mean of the first m minutes of a quarter, is known at "
        "exactly the quarter start plus m minutes, which is the forecast time, and the end value "
        "is the quarter's true last minute.",
        f"- {optimism_sentence(s)}",
        f"- Wall time of this run: {eu(_num(s.get('wall_s')) / 60.0, 1)} minutes "
        f"(evaluate {eu(_num(timing.get('evaluate_s')) / 60.0, 1)}, headroom "
        f"{eu(_num(timing.get('headroom_s', 0)) / 60.0, 1)}).",
    ]
    return lines


SOURCES_PARAGRAPH = (
    "Day-ahead prices come from energy-charts.info (CC BY 4.0, Bundesnetzagentur / SMARD.de). "
    "Settled imbalance prices come from TenneT publications. IDA prices come from the Nord Pool "
    "data portal. Live imbalance prices come from the TenEnergy live page, recorded by the "
    "owner's own recorder."
)


def render_report(s: dict) -> str:
    res, tiers, rules = s["results"], s["tiers"], s["rules"]
    f = s["facts"]
    gate = pct(rules["load_bearing_gain"], 0)
    title = f"# {_month(s)} price study" + (" (partial month)" if s.get("partial") else "")
    lines = [
        title,
        "",
        "Question (spec section 12): at the horizons a battery trader acts on, does each fresh "
        "information source lower the error of the settled Dutch imbalance price prediction over "
        "the day-ahead plus calendar baseline? Horizon h counts quarter-hours (ISPs) ahead.",
        "",
        "## Windows and data",
        "",
        *window_lines(s),
        "",
        "Folds (local dates):",
        "",
    ]
    lines += _table(
        ["fold", "train from", "train to", "test from", "test to", "test ISPs"],
        [
            [
                str(i),
                fo["train_first"],
                fo["train_last"],
                fo["test_first"],
                fo["test_last"],
                eu(fo["test_slots"], 0),
            ]
            for i, fo in enumerate(s["folds"])
        ],
    )
    lines += [
        "",
        "Feed coverage, share of quarter-hours with a value (IDA3: afternoon slots from 12:00 "
        "only). Train is the first fold's training span.",
        "",
    ]
    lines += _table(
        ["feed", "train", "test", "first day"],
        [[k, pct(v["train"]), pct(v["test"]), str(v["first"])] for k, v in s["coverage"].items()],
    )

    an = s["anchor"]
    lines += [
        "",
        "## Sanity anchor",
        "",
        f"Tier A buy MAE at h1: {eu(_num(an['tier_a_buy_mae_h1']), 2)} EUR/MWh. The "
        "owner's earlier unpublished baseline study (q50 pinball 27,4 EUR/MWh on 2023 to 2026) "
        "implies "
        f"about {eu(an['expected'], 0)} EUR/MWh. Ratio {eu(_num(an['ratio']), 2)}, "
        + ("in range (same order of magnitude)." if an["in_range"] else "OUT OF RANGE."),
        "",
        "## Reading rule",
        "",
        f"An information source is load-bearing if adding it lowers the buy q50 MAE by at least "
        f"{gate}, averaged over h1 to h4 (the mean of the per-horizon gains, on the common rows).",
        "",
    ]
    for tier in ("B", "C", "D"):
        if tier in s["reading"]:
            r = s["reading"][tier]
            lines.append(f"- {rule_sentence(tier, r)}")
            if r.get("over_A") and r["ref"] != "A":
                lines.append(
                    f"  Against tier A the gain is {pct(r['over_A']['mean'])} at h1 to h4."
                )
    if "D_h0_h2" in s["reading"]:
        lines.append(
            "- The spec expects the live feed to carry h0 to h2: tier D over tier C at h0 "
            f"(1 min) to h2 is {pct(s['reading']['D_h0_h2']['mean'])}."
        )
    if "D_over_B_h0_h2" in s["reading"]:
        lines.append(
            "- Counting tier C's live lags as live feed too, tier D over tier B at h0 (1 min) to "
            f"h2 is {pct(s['reading']['D_over_B_h0_h2']['mean'])}."
        )
    note = baseline_sentence(s)
    if note:
        lines += ["", note]

    hd = s["headroom"]
    lines += ["", "## Gradient-boosted headroom (tier D, buy)", ""]
    if hd is None:
        lines.append("Not run (tier D or the horizons h0 to h4 were not part of this run).")
    else:
        lines += [
            "Buy MAE in EUR/MWh of the linear tier D median against a "
            "HistGradientBoostingRegressor (absolute error, 300 iterations, learning rate 0,05) "
            "trained on the same tier D matrices per fold and scored on the same common rows. h0 "
            "uses 1 minute elapsed.",
            "",
        ]
        lines += _table(
            ["h", "linear", "boosted", "gain", "n"],
            [
                [
                    str(h),
                    eu(_num(hd["linear"][h])),
                    eu(_num(hd["boosted"][h])),
                    pct(hd["gain"][h]),
                    eu(hd["n"][h], 0),
                ]
                for h in hd["linear"]
            ],
        )
        verdict = "adopt boosting" if hd["adopt"] else "keep the linear model"
        lines += [
            "",
            f"Mean gain over h0 to h4: {pct(hd['mean_gain'])}. Adoption threshold "
            f"{pct(rules['adopt_min_gain'], 0)}. Verdict: {verdict}.",
        ]
        if not hd["complete"]:
            lines.append("The horizon set was incomplete, so the rule cannot adopt.")

    body = render_markdown(
        res,
        s["decay"],
        s["gap_test"],
        {"test window": f"{s['test_start']} to {s['test_last_day']}", "tiers": ", ".join(tiers)},
    )
    lines += ["", body.replace("# Price model evaluation", "## Error tables", 1).rstrip(), ""]
    ga = s["gap_all"]
    lines += [
        f"Over the whole live history ({f['live_first_day']} onward) the gap is: buy mean "
        f"{eu(_num(ga['buy']['mean']))}, median {eu(_num(ga['buy']['median']))}, "
        f"n {eu(ga['buy']['n'], 0)}; sell mean {eu(_num(ga['sell']['mean']))}, median "
        f"{eu(_num(ga['sell']['median']))}, n {eu(ga['sell']['n'], 0)} (EUR/MWh, live minus "
        "settled).",
        "",
        "## Pinball loss",
        "",
        "Pinball loss in EUR/MWh of the q10, q50 and q90 forecasts, averaged over the three "
        "quantiles, on the common rows. Lower is better. It scores the whole band, not the "
        "median alone.",
        "",
    ]
    groups = [*tiers, "baseline"]
    for target in _TARGETS:
        rows = []
        for label, h, e in _row_labels(s["horizons"]):
            r = [label]
            for g in groups:
                c = _cell_node(res, g, h, e).get(target, {})
                vals = [_num(c.get("pinball_" + q)) for q in _QNAMES]
                r.append(eu(float(np.mean(vals)) if all(map(math.isfinite, vals)) else math.nan))
            rows.append(r)
        lines += [f"{target.capitalize()}:", "", *_table(["h", *groups], rows), ""]

    lines += [
        "## Buy MAE per fold at h1",
        "",
        "Buy q50 MAE in EUR/MWh per weekly fold at h1, to show how much the weeks differ.",
        "",
    ]
    rows = []
    for g in (*tiers, "naive_da", "naive_persist"):
        c = _cell_node(res, g, 1, None).get("buy", {})
        rows.append([g, *(eu(_num(v)) for v in c.get("folds", []))])
    lines += _table(["column", *(f"fold {i}" for i in range(len(s["folds"])))], rows)

    lines += [
        "",
        "## Training rows per tier and fold",
        "",
        "Training rows (quarter-hours with every feature and label) per fold at h"
        f"{s['training'][tiers[0]]['ref_h']}, and the range over every horizon of the run.",
        "",
    ]
    rows = []
    for tier in tiers:
        t = s["training"][tier]
        rows.append(
            [tier, *(eu(n, 0) for n in t["per_fold"]), f"{eu(t['min'], 0)} to {eu(t['max'], 0)}"]
        )
    lines += _table(
        ["tier", *(f"fold {i}" for i in range(len(s["folds"]))), "range over horizons"], rows
    )
    missing = (
        f", for example from {f['label_missing_from']} onward" if f["label_missing_from"] else ""
    )
    lines += [
        "",
        "## Dropped rows",
        "",
        f"Rows of the whole master table ({f['master_first_day']} to {f['master_last_day']}, "
        f"{eu(s['master_rows'], 0)} quarter-hours) dropped per reason: a missing feature (tiers C "
        f"and D before the live history starts on {f['live_first_day']}; IDA gaps never drop a "
        f"row) or a missing label (no settled price{missing}). Test predicted is the number of "
        "test-window quarter-hours the tier predicts; the common row set of the error tables is "
        "smaller where tiers differ.",
        "",
    ]
    rows = []
    for key, d in s["dropped"].items():
        tier, h, e = key.split(":")
        if int(h) not in _DROP_HORIZONS or e not in ("-", str(ELAPSED_MIN[0])):
            continue
        hl = f"0 ({e} min)" if h == "0" else h
        rows.append(
            [
                tier,
                hl,
                eu(d["features"], 0),
                eu(d["labels"], 0),
                eu(d["kept"], 0),
                f"{eu(d['test_predicted'], 0)} of {eu(d['test_slots'], 0)}",
            ]
        )
    lines += _table(
        ["tier", "h", "dropped: feature", "dropped: label", "kept", "test predicted"], rows
    )
    lines += [
        "",
        f"Skipped cells (fewer than {rules['min_rows']} training rows in a fold): "
        f"{eu(res.get('skipped_cells', 0), 0)}.",
        "",
        "## What this means for a battery owner",
        "",
        " ".join(conclusion(s)),
        "",
        "## Sources and attribution",
        "",
        SOURCES_PARAGRAPH,
        "",
    ]
    return "\n".join(lines)
