import numpy as np

from tools import train_champion as TC
from tools.study.october_2026 import local_midnight


def test_windows_are_local_calendar_days():
    w = TC.windows("2026-10-28", holdout_days=14, train_days=365)
    # the holdout spans the 2026-10-25 DST change, so bounds are local midnights, not 86400 s steps
    assert w["end"] == local_midnight("2026-10-29")
    assert w["holdout_start"] == local_midnight("2026-10-15")
    assert w["start"] == local_midnight("2025-10-29")
    assert w["holdout_fit_start"] == local_midnight("2025-10-15")


def test_mask_is_half_open():
    ts = np.arange(0, 100, dtype=np.int64)
    m = TC.mask(ts, 10, 20)
    assert m.sum() == 10 and m[10] and not m[20]


def test_train_end_to_end_on_a_synthetic_master():
    from betterybalance.core import features as F

    from tests.test_features import _synthetic_master

    m = _synthetic_master(days=20)  # 2026-10-14 to 2026-11-02 local, across the DST change
    fits: dict = {}
    model = TC.train(
        m, "2026-10-31", holdout_days=3, train_days=12,
        tiers=("A", "B"), horizons=(1, 2), holdout_fits=fits,
    )
    meta = model.meta
    for key in (
        "train_start",
        "train_end",
        "train_start_by_tier",
        "holdout_start",
        "holdout_end",
        "holdout_mae",
        "holdout_protocol",
        "fitted_at",
        "version",
        "source_report",
        "cells",
    ):
        assert key in meta, key
    assert meta["holdout_start"] == "2026-10-29" and meta["holdout_end"] == "2026-10-31"
    assert meta["train_start"] == "2026-10-20" and meta["train_end"] == "2026-10-31"
    assert set(meta["train_start_by_tier"]) == {"A", "B"} <= set(F.TIERS)
    assert meta["cells"] == len(model.models) == 4

    # the holdout fit that train() made never sees a holdout label, the shipped fit does
    assert set(fits) == {"A", "B"}
    for tier, trial in fits.items():
        assert trial.meta["train_end"] < meta["holdout_start"] <= meta["train_end"], tier
        assert len(trial.models) == 2
    assert any(v["n"] > 0 for v in meta["holdout_mae"].values())
    assert all(v["mae"] is None or v["mae"] >= 0 for v in meta["holdout_mae"].values())
