import numpy as np
import pytest
from betterybalance.core import ledger as LG
from betterybalance.core.site.contract import Terms

T = Terms()


def test_balans_formula_with_fees_and_btw():
    imp, tax = LG.money_eur(1.0, 0.0, 100.0, 100.0, 1, T, "net_exporter")
    assert imp == pytest.approx((0.1 + 0.0165) * 1.21) and tax == 0.0
    exp, _ = LG.money_eur(0.0, 1.0, 50.0, 50.0, 1, T, "net_exporter")
    assert exp == pytest.approx(-(0.05 - 0.0165) * 1.21)
    dual, _ = LG.money_eur(
        1.0, 1.0, 300.0, -50.0, 2, T, "net_exporter"
    )  # dual: import at buy, export at sell
    assert dual == pytest.approx(((0.3 + 0.0165) + (0.05 + 0.0165)) * 1.21)
    inverted, _ = LG.money_eur(
        1.0, 0.0, 40.0, 60.0, 2, T, "net_exporter"
    )  # buy < sell on a state-2 row books as published
    assert inverted == pytest.approx((0.04 + 0.0165) * 1.21)


def test_single_pricing_uses_the_mean_on_both_legs():
    t = Terms(dual_pricing=False)
    total, _ = LG.money_eur(1.0, 1.0, 300.0, -50.0, 2, t, "net_exporter")
    assert total == pytest.approx(((0.125 + 0.0165) - (0.125 - 0.0165)) * 1.21)


def test_tax_modes():
    t = Terms(tax_eur_kwh=0.10)
    assert LG.money_eur(2.0, 1.0, 0.0, 0.0, 1, t, "none")[1] == pytest.approx(0.2 * 1.21)
    assert LG.money_eur(2.0, 1.0, 0.0, 0.0, 1, t, "net_importer")[1] == pytest.approx(0.1 * 1.21)
    assert LG.money_eur(2.0, 1.0, 0.0, 0.0, 1, t, "net_exporter")[1] == 0.0
    with pytest.raises(ValueError):
        LG.money_eur(2.0, 1.0, 0.0, 0.0, 1, t, "bogus")


def test_gross_split_books_both_directions_inside_a_quarter():
    net = np.array([6.0] * 10 + [-6.0] * 5)  # 10 min import at 6 kW, 5 min export at 6 kW
    imp, exp = LG.gross_split(net)
    assert imp == pytest.approx(1.0) and exp == pytest.approx(0.5)
    row = LG.rows_for_quarter(
        0, "virtual", net, {"buy": 100.0, "sell": 100.0, "state": 1, "settled": True}, T
    )
    assert (
        row.import_kwh == pytest.approx(1.0)
        and row.export_kwh == pytest.approx(0.5)
        and row.settled
    )
    assert row.money_eur == pytest.approx(((0.1 + 0.0165) * 1.0 - (0.1 - 0.0165) * 0.5) * 1.21)


def test_missing_quarter_has_nan_money_and_virtual_net():
    net = np.array([6.0] * 15)
    row = LG.rows_for_quarter(
        0, "actual", net, {"buy": 1.0, "sell": 1.0, "state": 1, "settled": False}, T, missing=True
    )
    assert (
        row.missing
        and np.isnan(row.money_eur)
        and np.isnan(row.tax_eur)
        and row.import_kwh == pytest.approx(1.5)
    )
    assert list(LG.virtual_net_kw(np.array([1.0, 2.0]), np.array([-3.0, 1.0]))) == [-2.0, 3.0]


def test_daily_pairs_exclude_missing_and_credit_balans_revenue():
    def rows(path, money, missing_at=()):
        return [
            LG.LedgerRow(
                1793487600 + 900 * i,
                path,
                100.0,
                100.0,
                1,
                True,
                0.5,
                0.0,
                money,
                0.0,
                False,
                0.0,
                i in missing_at,
            )
            for i in range(192)
        ]  # two local days from 2026-11-01 00:00

    v = rows("virtual", 0.10)
    a = rows("actual", 0.12, missing_at=(5,))
    pairs = LG.daily_pairs(v, a, Terms(balans_revenue_eur_month=30.0))
    assert (
        list(pairs["day"]) == ["2026-11-01", "2026-11-02"]
        and pairs["n_missing"][0] == 1
        and pairs["n_missing"][1] == 0
    )
    assert pairs["diff_eur"][1] == pytest.approx(
        96 * 0.02 - 1.0
    )  # actual minus virtual, minus the 1 EUR/day Balans credit
    assert pairs["diff_eur"][0] == pytest.approx(95 * 0.02 - 1.0)


def test_terminal_valuation_and_bootstrap_are_deterministic():
    assert LG.terminal_valuation(30.0, 20.0, 120.0, 0.9) == pytest.approx(10 * 0.12 * 0.9)
    diff = np.random.default_rng(1).normal(0.5, 1.0, 60)
    m, lo, hi = LG.bootstrap_interval(diff, seed=0)
    assert (
        lo < m < hi
        and m == pytest.approx(diff.mean())
        and LG.bootstrap_interval(diff, seed=0) == (m, lo, hi)
    )
    assert LG.bootstrap_interval(diff, seed=1) != (m, lo, hi)
    one = LG.bootstrap_interval(np.array([1.0]))
    assert one[0] == 1.0 and np.isnan(one[1]) and np.isnan(one[2])


def test_trailing_mean_buy_and_drift():
    ts = np.arange(0, 10 * 86400, 86400)
    buy = np.array([np.nan] + [100.0] * 9)
    assert LG.trailing_mean_buy(buy, ts, 10 * 86400, days=7) == pytest.approx(100.0)
    assert np.isnan(LG.trailing_mean_buy(buy[:1], ts[:1], 86400))

    def row(ts_, b):
        return LG.LedgerRow(ts_, "actual", b, b, 1, True, 0.0, 0.0, 0.0, 0.0, False, 0.0, False)

    assert LG.drift_eur_mwh(
        [row(0, 110.0), row(900, 90.0), row(1800, 5.0)], [row(0, 100.0), row(900, 100.0)]
    ) == pytest.approx(0.0)
    assert np.isnan(LG.drift_eur_mwh([row(0, 1.0)], [row(900, 1.0)]))
