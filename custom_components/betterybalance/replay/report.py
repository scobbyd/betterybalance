"""Replay evidence: the ladder, the selection rule, the greed check, the report and the files.

``ladder_rungs`` names the 20 knob sets, ``run_ladder`` replays each on one ``ReplayInputs``
(one band cache), ``select_default`` applies the selection rule on paired daily differences
against the default rung, ``greed_check`` counts the 18:00 greed rows,
``measure_hourly_bias`` re-ledgers the actual path on hourly means, ``render_report`` writes
the markdown and ``write_outputs`` the report plus the CSVs.

Rendered text prints every number through ``core.fmt.eu`` (European decimals). The CSVs
carry plain decimals (``repr`` of the float, so ``nan``, ``inf`` and ``-inf`` for non-finite
values) and no time of the run, so repeated calls on the same inputs give identical bytes.
Nothing here prints the site's coordinates. The ``meta`` dict is printed as given, so a
caller passes only labels it chose (``site``, ``window``).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import __version__
from ..core.fmt import eu
from ..core.ledger import LedgerRow, bootstrap_interval, rows_for_quarter
from ..core.results import export_results, results_bytes
from ..core.site.contract import Knobs, SiteConfig
from ..core.timegrid import ISP_S, Table, isp_start
from .bands import BandCache
from .disaggregate import flatten_to_hours, hourly_bias
from .engine import ReplayInputs, ReplayResult, decisions_table, ledger_table, replay
from .phases import GREED_COLUMNS, Prices, row_index

ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"
DATA_TERMS = (
    "Day-ahead prices come from energy-charts.info (CC BY 4.0, Bundesnetzagentur / SMARD.de).",
    "Settled imbalance prices come from TenneT publications.",
    "Live imbalance prices come from the TenEnergy live page.",
    "Weather data comes from the open-meteo archive (CC BY 4.0, non-commercial use, no key).",
)
IMPLS = ("ranking", "dp")
QUANTILE_PAIRS = ((0.1, 0.9), (0.25, 0.75), (0.5, 0.5))
KAPPAS = (0.25, 0.5, 1.0)
GREED_ROWS_SHOWN = 10
INDICATIVE_SENTENCE = "This comparison is indicative only."
# the ``bias`` argument for hourly input flows: no finer booking exists to compare against
HOURLY_INPUT = {"hourly_input": True}
HOURLY_INPUT_SENTENCE = (
    "The input flows are hourly, so the replay books disaggregated quarters and the hourly "
    "bias cannot be measured on this window."
)
# the quarter-versus-hour bias measured on the reference house (the owner's live window,
# Task 25, docs/reports/2026-10-31-plan2-replay.md "Hourly bias"): 26,71 EUR, 93,2 % of the
# live headline. Design 9.3: every hourly-input report prints a bias beside the headline and
# says "indicative only" above a tenth of it; a house with hourly flows only cannot measure
# its own, so the report prints this one with the indicative-only sentence.
REFERENCE_HOURLY_BIAS_SHARE = 0.932
NO_LIVE_DAY_SENTENCE = "The live store holds no day of this replay, so no difference is stated."


def hourly_input_lines() -> list[str]:
    """The sentences beside the headline of a report on hourly input flows: the reference
    house's measured bias as context and ``INDICATIVE_SENTENCE``."""
    share = eu(REFERENCE_HOURLY_BIAS_SHARE * 100, 1)
    return [
        "The quarter-versus-hour bias could not be measured for this house; on the reference "
        f"house it was {share} % of the live headline. {INDICATIVE_SENTENCE}"
    ]


def _minimal(x: float) -> str:
    """``x`` through ``eu`` with the fewest decimals that keep its value (0,25, 0,5, 1)."""
    for dec in range(7):
        if round(x, dec) == x:
            return eu(x, dec)
    return eu(x, 6)


def rung_name(impl: str, q_take: float, q_feed: float, kappa: float, penalty: str) -> str:
    """``"{impl} q{take}/q{feed} kappa {kappa} {penalty}"``, the quantiles in percent.

    ``penalty`` is the last part as printed: ``penalty on``, ``penalty off``,
    ``penalty none`` or ``expected-gain``.
    """
    take, feed = round(q_take * 100), round(q_feed * 100)
    return f"{impl} q{take}/q{feed} kappa {_minimal(kappa)} {penalty}"


def default_rung_name(knobs: Knobs) -> str:
    """The ladder name of a knob set: ``impl``, the quantiles, ``kappa`` and the last part
    (``expected-gain`` for that commit rule; else ``penalty on`` or ``penalty off`` for
    ``ranking`` and ``penalty none`` for ``dp``, which has no dual penalty)."""
    if knobs.commit_rule == "expected_gain":
        last = "expected-gain"
    elif knobs.impl == "ranking":
        last = "penalty on" if knobs.dual_penalty else "penalty off"
    else:
        last = "penalty none"
    return rung_name(knobs.impl, knobs.q_take, knobs.q_feed, knobs.kappa, last)


# the rung the selection measures against: the default knobs
DEFAULT_RUNG = default_rung_name(Knobs())


def check_ladder_default(knobs: Knobs) -> None:
    """Raise ``ValueError`` when ``knobs`` name no rung of ``ladder_rungs(knobs)``.

    The selection measures every rung against the rung of the site's knobs, so a ladder run
    on knobs off the grid would replay all 20 rungs and then fail. Call it before the first
    replay; the message lists the valid grid.
    """
    name = default_rung_name(knobs)
    names = [n for n, _ in ladder_rungs(knobs)]
    if name not in names:
        raise ValueError(
            f"the site's knobs ({name}) are not a ladder rung; --ladder needs knobs that name "
            "one of: " + "; ".join(names)
        )


def ladder_rungs(base: Knobs | None = None) -> list[tuple[str, Knobs]]:
    """The 20 rungs as ``(name, knobs)``, each the ``base`` knobs (default ``Knobs()``) with
    the rung's fields set.

    18 rungs: ``impl`` in (ranking, dp) by ``(q_take, q_feed)`` in ``QUANTILE_PAIRS`` by
    ``kappa`` in ``KAPPAS``, with the dual penalty on for ``ranking`` and off for ``dp``
    (named ``penalty none``: ``dp`` has no dual penalty). Then ``ranking q25/q75 kappa 0,5``
    with the penalty off and with ``commit_rule="expected_gain"``.
    """
    base = Knobs() if base is None else base
    rungs: list[tuple[str, Knobs]] = []
    for impl in IMPLS:
        for q_take, q_feed in QUANTILE_PAIRS:
            for kappa in KAPPAS:
                penalty = "penalty on" if impl == "ranking" else "penalty none"
                knobs = replace(
                    base,
                    impl=impl,
                    q_take=q_take,
                    q_feed=q_feed,
                    kappa=kappa,
                    commit_rule="margin",
                    dual_penalty=impl == "ranking",
                )
                rungs.append((rung_name(impl, q_take, q_feed, kappa, penalty), knobs))
    # the two variants of the plan-2 starting rung (ranking q25/q75 kappa 0,5)
    start = replace(
        base,
        impl="ranking",
        q_take=0.25,
        q_feed=0.75,
        kappa=0.5,
        commit_rule="margin",
        dual_penalty=True,
    )
    rungs.append(
        (rung_name("ranking", 0.25, 0.75, 0.5, "penalty off"), replace(start, dual_penalty=False))
    )
    rungs.append(
        (
            rung_name("ranking", 0.25, 0.75, 0.5, "expected-gain"),
            replace(start, commit_rule="expected_gain"),
        )
    )
    return rungs


def run_ladder(
    inputs: ReplayInputs,
    rungs: list[tuple[str, Knobs]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """One ``replay`` per rung on the same inputs (the same band cache).

    ``rungs`` defaults to ``ladder_rungs(inputs.cfg.knobs)``. ``progress(done, total)`` is
    called after every rung. Each row: ``name``, ``knobs``, ``diff_eur`` (the sum of the
    daily differences), ``wrong_side``, ``cycles_per_day``, ``daily`` (the daily
    differences) and ``days`` (their local days).
    """
    rungs = ladder_rungs(inputs.cfg.knobs) if rungs is None else rungs
    rows = []
    for i, (name, knobs) in enumerate(rungs):
        r = replay(inputs, knobs)
        daily = np.asarray(r.pairs["diff_eur"], dtype=float)
        rows.append(
            {
                "name": name,
                "knobs": knobs,
                "diff_eur": float(daily.sum()),
                "wrong_side": int(r.wrong_side),
                "cycles_per_day": float(r.cycles_per_day),
                "daily": daily,
                "days": np.asarray(r.pairs["day"], dtype=object),
            }
        )
        if progress is not None:
            progress(i + 1, len(rungs))
    return rows


def _paired(row: dict, default: dict, days: set[str]) -> np.ndarray:
    """``row - default`` daily differences on the days both have, in ``days``, finite only."""
    ref = {str(d): float(v) for d, v in zip(default["days"], default["daily"], strict=True)}
    out = []
    for d, v in zip(row["days"], row["daily"], strict=True):
        d = str(d)
        if d in days and d in ref:
            diff = float(v) - ref[d]
            if np.isfinite(diff):
                out.append(diff)
    return np.asarray(out, dtype=float)


def select_default(
    rows: list[dict], default_name: str, live_days: set[str], ab_days: set[str], seed: int = 0
) -> dict:
    """The selection rule on paired daily differences against the default rung.

    For every rung, the gain per day over the default on ``live_days`` (the rung's daily
    difference minus the default's, positive means the rung earned more) and its 90 %
    ``bootstrap_interval`` (seeded). A rung is a candidate when its mean gain is strictly
    positive, the interval's low end is above zero, and its mean gain on ``ab_days`` is at
    least zero (not worse than the default). The A/B condition is vacuously true only when
    ``ab_days`` is empty; with A/B days given, a rung without a paired A/B day fails it.
    The candidate with the largest live gain wins, the first in row order on equal gains.
    Without a candidate the default stays.

    Returns ``{"chosen", "table", "reason"}``; ``table`` holds one dict per rung with
    ``name``, ``mean_gain``, ``lo``, ``hi``, ``ab_mean`` (the paired A/B mean gain, NaN
    without A/B days), ``candidate``.
    """
    by_name = {r["name"]: r for r in rows}
    if default_name not in by_name:
        raise ValueError(f"default rung {default_name!r} is not in the ladder rows")
    default = by_name[default_name]
    live, ab = {str(d) for d in live_days}, {str(d) for d in ab_days}
    table = []
    for r in rows:
        gain = _paired(r, default, live)
        mean, lo, hi = bootstrap_interval(gain, seed=seed)
        ab_gain = _paired(r, default, ab)
        ab_mean = float(ab_gain.mean()) if len(ab_gain) else float("nan")
        ab_ok = not ab or (len(ab_gain) > 0 and ab_mean >= 0.0)
        candidate = (
            r["name"] != default_name
            and bool(np.isfinite(mean) and mean > 0.0)
            and bool(np.isfinite(lo) and lo > 0.0)
            and ab_ok
        )
        table.append(
            {
                "name": r["name"],
                "mean_gain": float(mean),
                "lo": float(lo),
                "hi": float(hi),
                "ab_mean": ab_mean,
                "candidate": bool(candidate),
            }
        )
    best = None
    for t in table:
        if t["candidate"] and (best is None or t["mean_gain"] > best["mean_gain"]):
            best = t
    if best is None:
        reason = (
            f"The default {default_name} stays: no rung gains more than it per live day with "
            "the low end of the 90 % interval above zero"
            + (" and no loss on the A/B days." if ab else ".")
        )
        return {"chosen": default_name, "table": table, "reason": reason}
    reason = (
        f"{best['name']} gains {eu(best['mean_gain'])} EUR per live day over the default "
        f"{default_name} (90 % interval {eu(best['lo'])} to {eu(best['hi'])} EUR)"
        + (
            f" and {eu(best['ab_mean'])} EUR per A/B day."
            if np.isfinite(best["ab_mean"])
            else "; no A/B days."
        )
    )
    return {"chosen": best["name"], "table": table, "reason": reason}


def greed_check(greed: Table) -> dict:
    """Per check (``low``, ``high``): the days it applies, the days it holds, the first fail.

    A day counts as ``ok`` when the check applies and holds. ``first_fail`` is the first
    day (in table order) where it applies and fails, None without one.
    """
    out = {}
    days = np.asarray(greed["day"], dtype=object)
    for side in ("low", "high"):
        applies = np.asarray(greed[f"{side}_soc_applies"], dtype=bool)
        ok = np.asarray(greed[f"{side}_soc_ok"], dtype=bool)
        fails = np.flatnonzero(applies & ~ok)
        out[side] = {
            "applies": int(applies.sum()),
            "ok": int((applies & ok).sum()),
            "first_fail": str(days[fails[0]]) if len(fails) else None,
        }
    return out


def hourly_ledgers(
    inputs: ReplayInputs, result: ReplayResult
) -> tuple[list[LedgerRow], list[LedgerRow]]:
    """The booked actual rows and the same quarters re-ledgered on hourly means, paired.

    The second list books every quarter of ``result.actual`` on its hour's mean grid net
    (``flatten_to_hours`` of ``inputs.flows_quarter``) flat over the 15 minutes, with the
    same prices and learning flag. A quarter missing in either list is missing in both,
    with NaN money and tax (the ledger invariant of a missing row).
    """
    flat = flatten_to_hours(inputs.flows_quarter)
    ts = np.asarray(flat["ts"], dtype=np.int64)
    grid_kw = (
        np.asarray(flat["grid_import_kwh"], dtype=float)
        - np.asarray(flat["grid_export_kwh"], dtype=float)
    ) * (3600.0 / ISP_S)
    flat_missing = np.asarray(flat["missing"], dtype=bool)
    prices = Prices(inputs.master)
    terms = inputs.cfg.terms
    booked, hourly = [], []
    for row in result.actual:
        k = row_index(ts, row.ts)
        bad = row.missing or k is None or bool(flat_missing[k]) or not np.isfinite(grid_kw[k])
        net = np.full(ISP_S // 60, np.nan if k is None else float(grid_kw[k]))
        h = rows_for_quarter(
            row.ts, "actual", net, prices.at(row.ts), terms, row.learning, 0.0, bad
        )
        nan = float("nan")
        booked.append(replace(row, missing=True, money_eur=nan, tax_eur=nan) if bad else row)
        hourly.append(h)
    return booked, hourly


def measure_hourly_bias(inputs: ReplayInputs, result: ReplayResult) -> dict:
    """``hourly_bias`` of the booked actual path against its hourly re-ledger
    (``hourly_ledgers``), relative to the headline cumulative difference."""
    booked, hourly = hourly_ledgers(inputs, result)
    return hourly_bias(booked, hourly, headline(result)["diff_eur"])


# -- rendering --------------------------------------------------------------------------


def _missing_quarters(result: ReplayResult) -> int:
    return int(sum(v.missing or a.missing for v, a in zip(result.virtual, result.actual)))


def headline(result: ReplayResult) -> dict:
    """The headline numbers: cumulative difference, mean and 90 % interval per day,
    terminal valuation, days and missing quarters."""
    mean, lo, hi = result.interval
    return {
        "diff_eur": float(np.sum(np.asarray(result.pairs["diff_eur"], dtype=float))),
        "mean_eur_day": float(mean),
        "lo_eur_day": float(lo),
        "hi_eur_day": float(hi),
        "terminal_eur": float(result.terminal_eur),
        "days": int(len(result.pairs["day"])),
        "missing_quarters": _missing_quarters(result),
    }


def headline_lines(result: ReplayResult) -> list[str]:
    """The headline as text lines, every number through ``eu`` (also the CLI print)."""
    h = headline(result)
    return [
        f"Cumulative difference (actual minus virtual): {eu(h['diff_eur'])} EUR "
        f"over {eu(h['days'], 0)} days",
        f"Mean per day: {eu(h['mean_eur_day'])} EUR, 90 % interval "
        f"{eu(h['lo_eur_day'])} to {eu(h['hi_eur_day'])} EUR",
        f"Terminal valuation of the SoC difference: {eu(h['terminal_eur'])} EUR",
        f"Missing quarters: {eu(h['missing_quarters'], 0)}",
    ]


def backtest_headline_lines(h: dict) -> list[str]:
    """The backtest headline (``live.backtest.result_headline``) as text lines: the shadow's
    own result (house minus shadow, design 10.4), the interval, the terminal valuation beside
    it (never added), the missing quarters, the PV learning days and the irradiance source.
    Every number through ``eu``; a missing value prints as "n/a"."""

    def money(x) -> str:
        return "n/a" if x is None else eu(float(x))

    return [
        f"Shadow result (house minus shadow): {money(h['result_eur'])} EUR "
        f"over {eu(h['days'], 0)} days",
        f"Mean per day: {money(h['mean_eur_day'])} EUR, 90 % interval "
        f"{money(h['lo_eur_day'])} to {money(h['hi_eur_day'])} EUR",
        f"Terminal valuation of the shadow's SoC change (beside the result, not added): "
        f"{money(h['terminal_eur'])} EUR",
        f"Missing quarters: {eu(h['missing_quarters'], 0)}",
        f"PV learning days: {eu(h['learning_days'], 0)}",
        f"Irradiance: {h['irradiance']}",
    ]


def render_backtest_report(result: ReplayResult, h: dict, meta: dict) -> str:
    """The backtest's markdown report: the headline block (``backtest_headline_lines``), the
    indicative sentence ``h["indicative"]``, the decision summary, the data terms and
    ``ATTRIBUTION``. It shows the shadow's own result only: no line reads the recorded
    battery's path (design 10.4)."""
    lines = ["# Backtest report", ""]
    if meta:
        lines += [*_meta_lines(meta), ""]
    lines += ["## Headline", "", *(f"- {line}" for line in backtest_headline_lines(h)), ""]
    lines += [
        "A positive result means the shadow battery cost the house less than no battery.",
        "",
        str(h["indicative"]),
        "",
    ]
    lines += _decision_section(result)
    lines += _terms_section()
    return "\n".join(lines)


def _yes(flag) -> str:
    return "yes" if bool(flag) else "no"


def _row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)]


def _meta_lines(meta: dict) -> list[str]:
    out = []
    for key in sorted(meta):
        value = meta[key]
        if isinstance(value, bool):
            text = _yes(value)
        elif isinstance(value, int | float):
            text = _minimal(float(value))
        else:
            text = str(value)
        out.append(f"- {key}: {text}")
    return out


def _headline_section(result: ReplayResult, meta: dict, bias: dict | None = None) -> list[str]:
    lines = ["# Replay report", ""]
    if meta:
        lines += [*_meta_lines(meta), ""]
    lines += ["## Headline", ""]
    lines += [f"- {line}" for line in headline_lines(result)]
    lines += [
        "",
        "A positive difference means the virtual battery cost the house less than the "
        "recorded one.",
        "",
    ]
    if bias is not None and bias.get("hourly_input"):
        lines += [*hourly_input_lines(), ""]
    return lines


def live_difference_lines(result: ReplayResult, live_store) -> list[str]:
    """The walk-forward replay against the live day (design 9): the mean absolute
    difference of the daily result (``reexec.daily_result_mad``), else
    ``NO_LIVE_DAY_SENTENCE``."""
    from .reexec import daily_result_mad  # local: the walk-forward path never loads live/

    mad = daily_result_mad(live_store, result)
    if mad != mad:
        return [NO_LIVE_DAY_SENTENCE]
    return [f"Differs from the live day by a mean absolute {eu(mad)} EUR per day."]


def _tiers_section(result: ReplayResult) -> list[str]:
    lines = ["## Tiers per month", ""]
    if not result.tiers_by_month:
        return lines + ["No quarter had a band.", ""]
    rows = [[m, t] for m, t in sorted(result.tiers_by_month.items())]
    return lines + _table(["month", "modal slot-0 tier"], rows) + [""]


def _greed_section(result: ReplayResult) -> list[str]:
    g = result.greed
    lines = ["## Greed check at 18:00", ""]
    n = len(g["day"])
    if n == 0:
        return lines + ["No greed row (no 18:00 quarter with a band).", ""]
    rows = []
    for i in range(min(n, GREED_ROWS_SHOWN)):
        rows.append(
            [
                str(g["day"][i]),
                eu(float(g["soc_kwh"][i])),
                eu(float(g["charge_threshold"][i])),
                eu(float(g["sale_threshold"][i])),
                eu(float(g["night_median_take"][i])),
                eu(float(g["night_median_feed"][i])),
                eu(float(g["surplus_tomorrow_kwh"][i])),
                f"{_yes(g['low_soc_applies'][i])} / {_yes(g['low_soc_ok'][i])}",
                f"{_yes(g['high_soc_applies'][i])} / {_yes(g['high_soc_ok'][i])}",
            ]
        )
    header = [
        "day",
        "SoC kWh",
        "charge threshold EUR/MWh",
        "sale threshold EUR/MWh",
        "night take EUR/MWh",
        "night feed EUR/MWh",
        "surplus tomorrow kWh",
        "low SoC applies / ok",
        "high SoC applies / ok",
    ]
    lines += _table(header, rows)
    if n > GREED_ROWS_SHOWN:
        lines += ["", f"First {eu(GREED_ROWS_SHOWN, 0)} of {eu(n, 0)} days."]
    check = greed_check(g)
    lines.append("")
    for side, label in (("low", "Low SoC"), ("high", "High SoC")):
        c = check[side]
        fail = c["first_fail"] or "none"
        lines.append(
            f"- {label}: applies on {eu(c['applies'], 0)} days, holds on {eu(c['ok'], 0)}, "
            f"first failing day {fail}."
        )
    return lines + [""]


def _num(x) -> str:
    return eu(None if x is None else float(x))


def _shading_lines(s: dict) -> list[str]:
    """One line per string of the champion's shading profile (``shading_share``: the share of
    the modelled energy it removes, shading.share_by_string); none without a profile."""
    return [
        f"- Shading removes {eu(100.0 * float(v), 1)} % of the modelled energy of {k}"
        for k, v in (s.get("shading_share") or {}).items()
    ]


def _site_section(result: ReplayResult) -> list[str]:
    s = result.site_report
    lines = ["## Site models", ""]
    lines.append(f"- PV: {s['pv']}")
    planes = s.get("planes") or []
    scales = s.get("scales_kwp") or []
    for i, p in enumerate(planes):
        kwp = f", {eu(scales[i])} kWp" if i < len(scales) else ""
        lines.append(
            f"- Plane {eu(i + 1, 0)}: tilt {eu(p['tilt_deg'], 1)} deg, "
            f"azimuth {eu(p['azimuth_deg'], 1)} deg{kwp}"
        )
    lines += _shading_lines(s)
    lines.append(f"- Holdout error: {_num(s.get('holdout_error'))}")
    lines.append(f"- Null error: {_num(s.get('null_error'))}")
    stability = s.get("stability")
    if stability is not None:
        lines.append("- Stability: " + ", ".join(eu(float(x)) for x in stability))
    lines.append(f"- Learning days: {eu(s['learning_days'], 0)}")
    lines.append(f"- Learned days: {eu(s['learned_days'], 0)}")
    lines.append(f"- Drift alarm seen: {_yes(s['drift'])}")
    lines.append(f"- EV sessions detected: {eu(s['ev']['sessions'], 0)}")
    lines.append(f"- Irradiance: {s['irradiance']}")
    lines.append(f"- GEN-port production: {s['gen_handling']}")
    lines.append(f"- Quarters without a band: {eu(s['no_band_quarters'], 0)}")
    lines.append(f"- Gross-split bias: {eu(s['gross_split_bias_eur'])} EUR")
    anchor = s.get("soc_end_anchor") or {}
    if anchor.get("minutes_before_end", 0) > 0:
        lines.append(
            "- SoC end anchor: no recorded SoC at the end; the terminal valuation reads the "
            f"last finite SoC, {eu(anchor['minutes_before_end'], 0)} min before the end"
        )
    return lines + [""]


def _ladder_section(ladder: list[dict] | None, selection: dict | None) -> list[str]:
    if not ladder and not selection:
        return []
    lines = ["## Ladder", ""]
    sel = {t["name"]: t for t in (selection or {}).get("table", [])}
    if ladder:
        header = ["rung", "days", "difference EUR", "wrong side", "cycles per day"]
        if sel:
            header += ["gain EUR/day", "90 % interval", "A/B gain EUR/day", "candidate"]
        rows = []
        for r in ladder:
            cells = [
                r["name"],
                eu(len(r["days"]), 0),
                eu(r["diff_eur"]),
                eu(r["wrong_side"], 0),
                eu(r["cycles_per_day"]),
            ]
            t = sel.get(r["name"])
            if sel:
                cells += (
                    [
                        eu(t["mean_gain"]),
                        f"{eu(t['lo'])} to {eu(t['hi'])}",
                        eu(t["ab_mean"]),
                        _yes(t["candidate"]),
                    ]
                    if t is not None
                    else [eu(None)] * 4
                )
            rows.append(cells)
        lines += _table(header, rows) + [""]
    if selection:
        lines += [f"Chosen: {selection['chosen']}.", "", selection["reason"], ""]
    return lines


def bias_paragraph(bias: dict | None) -> list[str]:
    """The hourly-bias section: the measure (``hourly_bias``), ``HOURLY_INPUT`` (the input
    was hourly, so nothing was measured) or nothing for None."""
    if bias is None:
        return []
    if bias.get("hourly_input"):
        return ["## Hourly bias", "", HOURLY_INPUT_SENTENCE, ""]
    text = (
        f"Booked on hourly means, the recorded flows cost {eu(bias['hourly_eur'])} EUR "
        f"against {eu(bias['quarter_eur'])} EUR on quarters: a bias of "
        f"{eu(bias['bias_eur'])} EUR, {eu(bias['bias_share'] * 100, 1)} % of the headline."
    )
    if bias.get("indicative_only"):
        text += " " + INDICATIVE_SENTENCE
    return ["## Hourly bias", "", text, ""]


def _quarters_with(decisions, reason: str) -> int:
    return len({int(isp_start(d.ts)) for d in decisions if d.reason == reason})


def _decision_section(result: ReplayResult) -> list[str]:
    d = result.decisions
    rows = [
        ("Commits (quarters)", eu(_quarters_with(d, "commit"), 0)),
        ("Pre-commits (quarters)", eu(_quarters_with(d, "precommit"), 0)),
        ("Revokes (quarters)", eu(_quarters_with(d, "revoked"), 0)),
        ("Negative-price charges (quarters)", eu(_quarters_with(d, "negative_take"), 0)),
        ("Wrong-side quarters", eu(result.wrong_side, 0)),
        ("Cycles per day", eu(result.cycles_per_day)),
    ]
    return ["## Decisions", "", *(f"- {k}: {v}" for k, v in rows), ""]


def _terms_section() -> list[str]:
    return ["## Data terms", "", *(f"- {t}" for t in DATA_TERMS), "", ATTRIBUTION, ""]


def render_report(
    result: ReplayResult,
    meta: dict,
    ladder: list[dict] | None = None,
    selection: dict | None = None,
    bias: dict | None = None,
    live_store=None,
) -> str:
    """The markdown report. Every number goes through ``eu``.

    With ``live_store`` (a ``core.store.Store`` of the live site) the headline also states
    the mean absolute difference from the live days (``live_difference_lines``): a
    walk-forward replay reads the band cache and the archive, not the live inputs, so it
    never claims identity with the live day.

    Sections: headline, tiers per month, greed check (first ``GREED_ROWS_SHOWN`` rows and
    the counts), site models, the ladder and the selection (when given), the hourly bias
    (when given, with ``INDICATIVE_SENTENCE`` when ``bias["indicative_only"]``, or
    ``HOURLY_INPUT_SENTENCE`` for ``HOURLY_INPUT``), the
    decision summary, the data terms and ``ATTRIBUTION``. For ``HOURLY_INPUT`` the headline
    always carries ``hourly_input_lines``: the reference house's bias and the indicative-only
    sentence (design 9.3).
    """
    lines = _headline_section(result, meta, bias)
    if live_store is not None:
        lines += [*live_difference_lines(result, live_store), ""]
    lines += _tiers_section(result)
    lines += _greed_section(result)
    lines += _site_section(result)
    lines += _ladder_section(ladder, selection)
    lines += bias_paragraph(bias)
    lines += _decision_section(result)
    lines += _terms_section()
    return "\n".join(lines)


# -- files ------------------------------------------------------------------------------


def _cell(v) -> str:
    """A CSV cell: ints as ints, floats by ``repr`` (``nan``, ``inf``, ``-inf``),
    booleans ``true``/``false``, the rest as text."""
    if isinstance(v, bool | np.bool_):
        return "true" if bool(v) else "false"
    if isinstance(v, int | np.integer):
        return str(int(v))
    if isinstance(v, float | np.floating):
        return repr(float(v))
    return str(v)


def _csv_bytes(columns: list[str], table: dict) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    n = len(table[columns[0]]) if columns else 0
    for i in range(n):
        writer.writerow([_cell(table[c][i]) for c in columns])
    return buf.getvalue().encode("utf-8")


def _ledger_csv(rows: list[LedgerRow]) -> bytes:
    t = ledger_table(rows)
    return _csv_bytes(list(t), t)


def _ladder_table(ladder: list[dict], selection: dict | None) -> tuple[list[str], dict]:
    sel = {t["name"]: t for t in (selection or {}).get("table", [])}
    cols: dict[str, list] = {
        "name": [],
        "impl": [],
        "q_take": [],
        "q_feed": [],
        "kappa": [],
        "commit_rule": [],
        "dual_penalty": [],
        "days": [],
        "diff_eur": [],
        "wrong_side": [],
        "cycles_per_day": [],
    }
    extra = ("mean_gain", "lo", "hi", "ab_mean", "candidate", "chosen") if selection else ()
    for c in extra:
        cols[c] = []
    for r in ladder:
        k = r["knobs"]
        cols["name"].append(r["name"])
        cols["impl"].append(k.impl)
        cols["q_take"].append(float(k.q_take))
        cols["q_feed"].append(float(k.q_feed))
        cols["kappa"].append(float(k.kappa))
        cols["commit_rule"].append(k.commit_rule)
        cols["dual_penalty"].append(bool(k.dual_penalty))
        cols["days"].append(int(len(r["days"])))
        cols["diff_eur"].append(float(r["diff_eur"]))
        cols["wrong_side"].append(int(r["wrong_side"]))
        cols["cycles_per_day"].append(float(r["cycles_per_day"]))
        if selection:
            t = sel.get(r["name"], {})
            nan = float("nan")
            for c in ("mean_gain", "lo", "hi", "ab_mean"):
                cols[c].append(float(t.get(c, nan)))
            cols["candidate"].append(bool(t.get("candidate", False)))
            cols["chosen"].append(r["name"] == selection["chosen"])
    return list(cols), cols


def write_outputs(
    result: ReplayResult,
    out_dir: Path,
    meta: dict,
    ladder: list[dict] | None = None,
    selection: dict | None = None,
    bias: dict | None = None,
    cfg: SiteConfig | None = None,
    version: str = __version__,
    live_store=None,
) -> list[Path]:
    """Write the report and the CSVs into ``out_dir``; return the paths in write order.

    ``report.md``, ``ledger_virtual.csv``, ``ledger_actual.csv``, ``decisions.csv``,
    ``daily_pairs.csv``, ``greed.csv``, ``ladder.csv`` (only with ``ladder``) and
    ``results.json`` (only with ``cfg``: the anonymised export, ``export_results`` with
    ``version``). ``live_store`` goes to ``render_report``. Byte-identical on repeated calls
    with the same inputs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    decisions = decisions_table(result.decisions)
    pairs = result.pairs
    files: list[tuple[str, bytes]] = [
        (
            "report.md",
            render_report(result, meta, ladder, selection, bias, live_store).encode("utf-8"),
        ),
        ("ledger_virtual.csv", _ledger_csv(result.virtual)),
        ("ledger_actual.csv", _ledger_csv(result.actual)),
        ("decisions.csv", _csv_bytes(list(decisions), decisions)),
        ("daily_pairs.csv", _csv_bytes(list(pairs), pairs)),
        ("greed.csv", _csv_bytes(list(GREED_COLUMNS), result.greed)),
    ]
    if ladder is not None:
        columns, table = _ladder_table(ladder, selection)
        files.append(("ladder.csv", _csv_bytes(columns, table)))
    if cfg is not None:
        files.append(("results.json", results_bytes(export_results(result, cfg, version))))
    paths = []
    for name, data in files:
        path = out / name
        path.write_bytes(data)
        paths.append(path)
    return paths


def earliest_start(cache: BandCache) -> int | None:
    """The first quarter of a band cache that a replay may start at (``fold_of >= 0``)."""
    k = np.flatnonzero(np.asarray(cache.fold_of, dtype=np.int64) >= 0)
    return None if not len(k) else int(np.asarray(cache.ts, dtype=np.int64)[k[0]])
