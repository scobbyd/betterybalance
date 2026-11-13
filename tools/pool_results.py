"""Pool anonymised results files (``results.json``, format 1) from several houses.

``python -m tools.pool_results DIR`` reads every ``*.json`` in DIR with ``format == 1`` and
prints the pooled markdown. Numbers in the markdown go through ``core.fmt.eu``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from betterybalance.core.fmt import eu
from betterybalance.core.ledger import bootstrap_interval
from betterybalance.core.results import RESULTS_FORMAT

BINS = ("<10", "10-20", "20-40", ">=40")


def pool(paths: list[Path], seed: int = 0) -> dict:
    """Pool the daily differences of all houses, each day one sample."""
    all_days: list[float] = []
    by_class = {b: 0 for b in BINS}
    per_house = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        diffs = [float(r["diff_eur"]) for r in d["daily"] if r["diff_eur"] is not None]
        all_days += diffs
        cap = d["site_class"]["capacity_bin"]
        by_class[cap] = by_class.get(cap, 0) + 1
        mean = float(np.mean(diffs)) if diffs else float("nan")
        per_house.append({"name": Path(p).stem, "days": len(diffs), "mean": mean})
    mean, lo, hi = bootstrap_interval(np.asarray(all_days, dtype=float), seed=seed)
    return {
        "houses": len(paths),
        "days": len(all_days),
        "mean_eur_day": mean,
        "interval": (lo, hi),
        "by_class": by_class,
        "per_house": per_house,
    }


def render(pooled: dict) -> str:
    """The pooled result as markdown, every number through ``eu``."""
    lo, hi = pooled["interval"]
    lines = [
        "# Pooled results",
        "",
        f"- Houses: {eu(pooled['houses'], 0)}",
        f"- Days: {eu(pooled['days'], 0)}",
        f"- Mean per day: {eu(pooled['mean_eur_day'])} EUR, 90 % interval "
        f"{eu(lo)} to {eu(hi)} EUR",
        "",
        "## Houses per capacity class (kWh)",
        "",
        "| class | houses |",
        "| --- | --- |",
        *(f"| {b} | {eu(n, 0)} |" for b, n in pooled["by_class"].items()),
        "",
        "## Per house",
        "",
        "| house | days | mean EUR/day |",
        "| --- | --- | --- |",
        *(f"| {h['name']} | {eu(h['days'], 0)} | {eu(h['mean'])} |" for h in pooled["per_house"]),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path, help="directory with results files")
    args = ap.parse_args(argv)
    paths = []
    for p in sorted(args.dir.glob("*.json")):
        try:
            if json.loads(p.read_text()).get("format") == RESULTS_FORMAT:
                paths.append(p)
        except (ValueError, AttributeError):
            continue
    print(render(pool(paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
