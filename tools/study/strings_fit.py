"""Supervised per-string PV fit against plan 2's blind fit on one house (plan 3, Task 21 Step 5).

    python -m tools.study.strings_fit --flows FLOWS.csv --site SITE.json --irradiance IRR.csv \
        [--planes PLANES.json [--assign pv2,pv1]] [--calls DIR] [--until YYYY-MM-DD] \
        [--shading] --out LOG

Reads an hourly flows export with string columns (tools.export_site --solarman), splits it to
quarters the way the integration does (flat, hourly origin), fits strings.fit_strings and
separate.fit_pv on the same window, and scores both against the measured sum of the strings
on the holdout days of the supervised fit, with the same Huber delta. The measured sum is AC:
the DC strings pass the loss profile's DC-to-AC stage (battery.pv_dc_to_ac_stage), as in
strings.fit_strings. The supervised fits run without the shading profile; with --shading
each supervised fit runs a second time with it (plan 3, Task 22) and the log adds the share it
removes per string. The profile reads the fit rows only, which lie before the holdout, so
the holdout scores it as a forecast. Every number goes
through core.fmt.eu. Weather data by Open-Meteo.com (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from betterybalance.core.battery import pv_dc_to_ac_stage
from betterybalance.core.feeds.openmeteo import read_irradiance_csv
from betterybalance.core.fmt import eu
from betterybalance.core.site import contract as C
from betterybalance.core.site import separate as SEP
from betterybalance.core.site import shading as SH
from betterybalance.core.site import strings as ST
from betterybalance.core.site.flows import read_flows
from betterybalance.core.site.load import local_fields
from betterybalance.core.site.solar import daylight, solar_position
from betterybalance.core.timegrid import align
from betterybalance.replay.disaggregate import hourly_to_quarters

ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"


def _holdout_rows(q, cfg, now, holdout_days=14):
    """The supervised fit's holdout rows (its mask rules, repeated here for scoring) and the
    measured AC total: the DC strings through the loss profile's DC-to-AC stage, plus GEN."""
    ts = np.asarray(q["ts"], dtype=np.int64)
    present = ST.strings_present(q)
    stage = pv_dc_to_ac_stage(cfg.loss_profile)
    dc = sum(
        (np.asarray(q[c], dtype=float) * 4.0 for c in present if c != ST.GEN_COLUMN),
        np.zeros(len(ts)),
    )
    gen = np.asarray(q[ST.GEN_COLUMN], dtype=float) * 4.0 if ST.GEN_COLUMN in present else 0.0
    total = np.where(np.isfinite(dc), SEP.dc_to_ac_kw(dc, stage), np.nan) + gen
    zenith, _ = solar_position(ts, cfg.latitude, cfg.longitude)
    lit = daylight(zenith)
    ok = (
        lit
        & np.isfinite(total)
        & ~np.asarray(q["missing"], dtype=bool)
        & ~SEP.curtailment_mask(q)
        & (ts < now)
    )
    _, _, day = local_fields(ts)
    days = np.unique(day[ok])
    n_hold = min(holdout_days, len(days) // 2)
    hold = ok & np.isin(day, days[len(days) - n_hold :])
    return hold, np.nan_to_num(total), present, days[len(days) - n_hold :]


def _score(model, irr_q, q, cfg, hold, total, delta):
    pred = SEP.predict_pv_kw(
        model, irr_q, cfg.latitude, cfg.longitude, hour_mean_rows=SEP.hourly_origin_rows(q)
    )
    err = total[hold] - pred[hold]
    return {
        "huber": SEP.huber_loss(err, delta),
        "mae_kw": float(np.abs(err).mean()),
        "bias_share": float(pred[hold].sum() / total[hold].sum() - 1.0),
    }


def _model_lines(name, model):
    out = [f"{name}:"]
    for p, k in zip(model.planes, model.scales_kwp, strict=True):
        label = p.string or "plane"
        out.append(
            f"  {label}: tilt {eu(p.tilt_deg, 0)} deg, azimuth {eu(p.azimuth_deg, 0)} deg, "
            f"{eu(float(k), 2)} kWp"
        )
    he = "none" if model.holdout_error is None else eu(model.holdout_error, 4)
    ne = "none" if model.null_error is None else eu(model.null_error, 4)
    mf = ", ".join(eu(float(x), 3) for x in model.month_factor[8:10])
    out.append(f"  own holdout error {he}, null {ne}; month factor Sep, Oct {mf}")
    return out


def _shading_lines(model, irr_q, cfg):
    shares = SH.share_by_string(model, irr_q, cfg.latitude, cfg.longitude)
    return [
        "  shading removes " + ", ".join(f"{k} {eu(100 * v, 1)} %" for k, v in shares.items())
        + " of the modelled energy (whole window)"
    ]


def run(
    flows, site, irradiance, planes_file, calls, log, until=None, assign=None, shading=False
) -> list[str]:
    h = read_flows(Path(flows))
    if until is not None:  # end the window before this local day
        import datetime as dt

        from betterybalance.core.timegrid import ZONE

        end = int(dt.datetime.fromisoformat(until).replace(tzinfo=ZONE).timestamp())
        keep = np.asarray(h["ts"]) < end
        h = {k: np.asarray(v)[keep] for k, v in h.items()}
    q = hourly_to_quarters(h)
    d = json.loads(Path(site).read_text())
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    d["pv"] = {"none": False, "planes": []}
    cfg = C.from_dict(d)
    irr = read_irradiance_csv(Path(irradiance))
    now = int(q["ts"][-1]) + 900
    irr_q = align(irr, q["ts"], ["ghi_w_m2", "dni_w_m2", "dhi_w_m2", "temp_c"])
    irr_q["ts"] = q["ts"]
    lines = [
        f"Window: {eu(len(h['ts']), 0)} hours, last quarter before {now}",
        f"Strings present: {', '.join(ST.strings_present(q)) or 'none'}",
        f"Loss profile {cfg.loss_profile}: DC-to-AC stage "
        f"{eu(pv_dc_to_ac_stage(cfg.loss_profile)[0], 1)} W/kW + "
        f"{eu(pv_dc_to_ac_stage(cfg.loss_profile)[1], 1)} W/kW2",
    ]
    hold, total, present, hold_days = _holdout_rows(q, cfg, now)
    lines.append(f"Holdout: {eu(len(hold_days), 0)} days, {eu(int(hold.sum()), 0)} quarters")
    fit_rows = ~hold & (total > 0)
    delta = max(float(np.percentile(total[fit_rows], 90)) / 3.0, 1e-3)
    lines.append(f"Scoring delta {eu(delta, 3)} kW (a third of the P90 measured sum)")

    sup = ST.fit_strings(q, irr, cfg, now, shading=False)
    lines += _model_lines("Supervised per string", sup)
    shaded = []
    if shading:
        sup_sh = ST.fit_strings(q, irr, cfg, now)
        lines += _model_lines("Supervised per string, shading", sup_sh)
        lines += _shading_lines(sup_sh, irr_q, cfg)
        shaded.append(("supervised, shading", sup_sh))
    blind, _, _ = SEP.fit_pv(dict(q, gen_kwh=np.zeros(len(q["ts"]))), irr, cfg, now)
    lines += _model_lines("Blind (plan 2)", blind)

    scored = [("supervised", sup), ("blind", blind)]
    if planes_file:
        ref = json.loads(Path(planes_file).read_text()).get("reference", {})
        given = tuple(
            C.Plane(p["tilt_deg"], p["azimuth_deg"], p.get("kwp")) for p in ref.get("planes", ())
        )
        if given:
            fixed = ST.fit_strings(q, irr, cfg, now, planes=given, shading=False)
            lines += _model_lines("Reference geometry fixed, unnamed planes on the sum", fixed)
            off = ST.scale_check(fixed, given)
            lines.append(
                f"  scale check against the reference kWp: {ST.string_labels(given, off) or 'none'}"
            )
            scored.append(("reference fixed", fixed))
            if assign:  # the fill-in case: each reference plane names its string
                named = tuple(
                    C.Plane(p.tilt_deg, p.azimuth_deg, p.kwp, s)
                    for p, s in zip(given, assign.split(","), strict=True)
                )
                fill = ST.fit_strings(q, irr, cfg, now, planes=named, shading=False)
                lines += _model_lines("Reference geometry, each plane on its string", fill)
                off = ST.scale_check(fill, named)
                lines.append(
                    "  scale check against the reference kWp: "
                    + (ST.string_labels(named, off) or "none")
                    + "; fitted over entered "
                    + ", ".join(
                        f"{p.string} {eu(float(fill.scales_kwp[i]) / p.kwp, 2)}"
                        for i, p in enumerate(named)
                        if p.kwp
                    )
                )
                scored.append(("reference named", fill))
                if shading:
                    fill_sh = ST.fit_strings(q, irr, cfg, now, planes=named)
                    label = "Reference geometry, each plane on its string, shading"
                    lines += _model_lines(label, fill_sh)
                    lines += _shading_lines(fill_sh, irr_q, cfg)
                    lines.append(
                        "  scale check (scales before the shading round): "
                        + (ST.string_labels(named, ST.scale_check(fill_sh, named)) or "none")
                    )
                    shaded.append(("reference named, shading", fill_sh))

    lines.append("Against the measured sum on the supervised holdout (same rows, same delta):")
    for name, m in scored + shaded:
        s = _score(m, irr_q, q, cfg, hold, total, delta)
        lines.append(
            f"  {name}: Huber {eu(s['huber'], 4)}, MAE {eu(s['mae_kw'], 3)} kW, "
            f"energy bias {eu(100 * s['bias_share'], 1)} %"
        )
    call_dir = Path(calls) if calls else None
    extracts = sorted(call_dir.glob("*/flows.csv")) if call_dir and call_dir.is_dir() else []
    if not extracts:
        lines.append("Data calls: no extract has arrived (no folder under data/calls/).")
    lines.append(ATTRIBUTION)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    Path(log).write_text("\n".join(lines) + "\n")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flows", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--irradiance", required=True)
    ap.add_argument("--planes")
    ap.add_argument("--calls", default="data/calls")
    ap.add_argument("--until", help="end the window before this local day, YYYY-MM-DD")
    ap.add_argument("--assign", help="strings of the reference planes in file order, e.g. pv2,pv1")
    ap.add_argument("--shading", action="store_true", help="add the fits with the shading profile")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = run(
        a.flows, a.site, a.irradiance, a.planes, a.calls, a.out, a.until, a.assign, a.shading
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
