"""The trader: one ranking per quarter, one decision per minute.

The trader ranks the window once per quarter (``rank``, at minute 1 or later) and keeps
the thresholds. Every minute (``minute``) it compares the nowcast against the charge and
discharge thresholds of the running quarter's slot, in band half-widths, and commits a
grid trade when the margin clears the commit rule. A committed action stays until the
quarter ends. At the quarter boundary it may pre-commit from the previous quarter's
ranking. Without a commitment the pack runs the value-tested default policy.

Prices are EUR/MWh, values EUR per cells kWh, powers AC kW with charge positive. Dwell
and latency are the executor's job, not the trader's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..battery import LossProfile
from ..pricemodel import QUANTILES
from ..site.contract import SiteConfig
from ..site.flows import ac_legs
from ..site.socpath import SocPath
from ..timegrid import ISP_S, isp_start
from .thresholds import Thresholds, thresholds
from .window import Nowcast, Window, band_for_slot

STALE_S = 3600
NORMAL_Q75_Z = 0.6745  # the q25 to q75 half-width of a normal band in standard deviations
WRONG_SIDE_LOSS_W = 1.0  # expected-gain rule: loss on the wrong side, in half-widths
COMMITTED_REASONS = ("commit", "precommit", "negative_take")


@dataclass
class Decision:
    ts: int
    action: str
    kw: float
    committed: bool
    reason: str
    margin_w: float
    value_stored: float
    charge_threshold: float
    discharge_threshold: float
    learning: bool


@dataclass
class _Commitment:
    quarter: int
    action: str
    reason: str


def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _in_widths(diff: float, w: float) -> float:
    """``diff / w``; with ``w <= 0`` the sign of ``diff`` as an infinity (0 stays 0)."""
    if w > 0:
        return diff / w
    if diff == 0:
        return 0.0
    return math.copysign(math.inf, diff)


def margins(
    take_point: float,
    feed_point: float,
    w_take: float,
    w_feed: float,
    charge_th: float,
    discharge_th: float,
) -> tuple[float, float]:
    """Charge and discharge margins in half-widths.

    ``margin_take = (charge_th - take_point) / w_take`` and
    ``margin_feed = (feed_point - discharge_th) / w_feed``.
    """
    return (
        _in_widths(charge_th - take_point, w_take),
        _in_widths(feed_point - discharge_th, w_feed),
    )


def kappa_test(margin_w: float, kappa: float) -> bool:
    """The margin rule: commit when ``margin_w >= kappa``."""
    return margin_w >= kappa


def expected_gain_test(margin_w: float) -> bool:
    """The expected-gain rule: commit when ``P * margin_w - (1 - P) * 1.0 > 0``.

    ``P = phi(0.6745 * margin_w)`` is the chance that the price lands on the right side
    of the threshold for a normal band; the wrong-side loss is taken as one half-width.
    """
    if math.isnan(margin_w):
        return False
    if math.isinf(margin_w):
        return margin_w > 0
    p = phi(NORMAL_Q75_Z * margin_w)
    return p * margin_w - (1.0 - p) * WRONG_SIDE_LOSS_W > 0


def _band_at(band: dict, q: float) -> float:
    """Band value at quantile ``q``, linear between the fitted quantiles."""
    return float(np.interp(q, QUANTILES, [band[x] for x in QUANTILES]))


def _half_width(band: dict) -> float:
    return (band[0.75] - band[0.25]) / 2.0


class Trader:
    """Minute decisions of one site.

    Precedence inside ``minute`` (``e`` = whole minutes into the quarter):

    1. ``stale``: no ranking in the last ``STALE_S`` seconds. Default policy.
    2. A pre-commit of this quarter: its revoke test runs first. A margin under ``-kappa``
       revokes it; that minute is ``revoked`` (default policy) and tests no new commit.
    3. A commitment of this quarter (``commit``, ``precommit``, ``negative_take``) stays.
    4. ``e == 0``: the boundary pre-commit (``precommit``), else ``default``.
    5. ``1 <= e <= latest_commit_min``: ``negative_take``, then ``commit``, else ``wait``.
    6. ``e > latest_commit_min``: ``late``.

    Every non-committed decision runs the value-tested default policy for ``kw``, and its
    ``action`` follows the sign of ``kw`` (charge above 0, discharge below 0, hold at 0).
    """

    def __init__(self, cfg: SiteConfig, loss: LossProfile):
        self.cfg = cfg
        self.loss = loss
        self._window: Window | None = None
        self._th: Thresholds | None = None
        self._ranked_at: int | None = None
        self._commitment: _Commitment | None = None

    def rank(self, window: Window, soc_path: SocPath, need: float, now: int) -> Thresholds:
        """Rank the window and store the window, the thresholds and the time of the ranking."""
        cfg = self.cfg
        th = thresholds(window, soc_path, cfg.pack, cfg.terms, cfg.knobs, need, now, loss=self.loss)
        self._window, self._th, self._ranked_at = window, th, int(now)
        return th

    # -- the stored ranking -------------------------------------------------------------

    def _slot(self, quarter: int) -> int | None:
        """Index of ``quarter`` in the stored window, or None without one that holds it."""
        if self._window is None:
            return None
        j = (quarter - int(self._window.ts[0])) // ISP_S
        return j if 0 <= j < len(self._window.ts) else None

    def _levels(self, j: int | None) -> tuple[float, float, float]:
        """``value_stored``, ``charge`` and ``discharge`` of slot ``j`` (NaN without one)."""
        if j is None or self._th is None:
            return math.nan, math.nan, math.nan
        th = self._th
        return float(th.value_stored[j]), float(th.charge[j]), float(th.discharge[j])

    # -- the default policy -------------------------------------------------------------

    def _prices_now(self, nowcast: Nowcast | None, j: int | None) -> tuple[float, float]:
        """Take and feed points for the default policy: the nowcast, else slot q50."""
        if nowcast is not None:
            return nowcast.take_point, nowcast.feed_point
        if j is None:
            return math.nan, math.nan
        band = band_for_slot(self._window, j)
        return band["take"][0.5], band["feed"][0.5]

    def default_kw(
        self,
        residual_kw: float,
        take_point: float,
        feed_point: float,
        value_stored: float,
        ev_plan_slot: bool,
    ) -> float:
        """Absorb surplus and cover load, each unless the grid beats the stored value.

        Surplus: ``min(-residual, charge_kw)`` unless ``feed / 1000 - fee_export >
        value_stored * ec``. Load: ``-min(load, discharge_kw)`` unless ``take / 1000 +
        fee_import < value_stored / ed``. In an EV plan slot the load is
        ``max(residual - charger_kw, 0)``. A NaN price or value skips its value test.
        """
        pack, terms = self.cfg.pack, self.cfg.terms
        ec, ed = ac_legs(pack)
        if residual_kw < 0:
            if feed_point / 1000.0 - terms.fee_export_eur_kwh > value_stored * ec:
                return 0.0
            return min(-residual_kw, pack.charge_kw)
        load = max(residual_kw - self.cfg.ev.charger_kw, 0.0) if ev_plan_slot else residual_kw
        if load <= 0:
            return 0.0
        if take_point / 1000.0 + terms.fee_import_eur_kwh < value_stored / ed:
            return 0.0
        return -min(load, pack.discharge_kw)

    # -- the commit tests ---------------------------------------------------------------

    def commit_test(self, margin_w: float) -> bool:
        """``knobs.commit_rule``: ``margin`` (kappa) or ``expected_gain``."""
        knobs = self.cfg.knobs
        if knobs.commit_rule == "expected_gain":
            return expected_gain_test(margin_w)
        return kappa_test(margin_w, knobs.kappa)

    def _precommit(self, j: int) -> tuple[str | None, float]:
        """Boundary pre-commit on the stored ranking's h1 band of slot ``j`` (= 1).

        Points at ``q_take`` and ``q_feed``, half-widths from q25 and q75. Only a charge
        slot tests the charge side and only a sale slot the sale side. Returns the action
        (None without a commit) and the margin: the acting side's, else the larger one.
        """
        th, knobs = self._th, self.cfg.knobs
        band = band_for_slot(self._window, j)
        m_take, m_feed = margins(
            _band_at(band["take"], knobs.q_take),
            _band_at(band["feed"], knobs.q_feed),
            _half_width(band["take"]),
            _half_width(band["feed"]),
            float(th.charge[j]),
            float(th.discharge[j]),
        )
        m_take = m_take if th.charge_slots[j] else -math.inf
        m_feed = m_feed if th.sale_slots[j] else -math.inf
        action, best = ("charge", m_take) if m_take >= m_feed else ("discharge", m_feed)
        return (action if self.commit_test(best) else None), best

    def _trade_slot(self, j: int) -> bool:
        """Slot ``j`` is a charge or a sale slot of the stored ranking."""
        return bool(self._th.charge_slots[j] or self._th.sale_slots[j])

    def _revoked(self, action: str, m_take: float, m_feed: float) -> bool:
        """A pre-commit falls when its own side's margin is under ``-kappa``."""
        m = m_take if action == "charge" else m_feed
        return m < -self.cfg.knobs.kappa

    # -- the minute ---------------------------------------------------------------------

    def _kw_for(self, action: str) -> float:
        pack = self.cfg.pack
        return pack.charge_kw if action == "charge" else -pack.discharge_kw

    def minute(
        self,
        ts: int,
        nowcast: Nowcast | None,
        residual_kw: float,
        soc_kwh: float,
        ev_plan_slot: bool,
        learning: bool,
    ) -> Decision:
        """One decision for the minute at ``ts`` (precedence in the class docstring)."""
        ts = int(ts)
        quarter = int(isp_start(ts))
        e = (ts - quarter) // 60
        knobs = self.cfg.knobs
        stale = self._ranked_at is None or ts - self._ranked_at > STALE_S
        j = self._slot(quarter)
        value, charge_th, discharge_th = self._levels(j)
        take_now, feed_now = self._prices_now(nowcast, j)
        if nowcast is not None:
            m_take, m_feed = margins(
                nowcast.take_point,
                nowcast.feed_point,
                nowcast.w_take,
                nowcast.w_feed,
                charge_th,
                discharge_th,
            )
        else:
            m_take, m_feed = math.nan, math.nan
        best = max(m_take, m_feed) if nowcast is not None else math.nan

        def decide(action: str, kw: float, reason: str, margin_w: float) -> Decision:
            return Decision(
                ts,
                action,
                float(kw),
                reason in COMMITTED_REASONS,
                reason,
                float(margin_w),
                value,
                charge_th,
                discharge_th,
                learning,
            )

        def default(reason: str, margin_w: float = best) -> Decision:
            kw = self.default_kw(residual_kw, take_now, feed_now, value, ev_plan_slot)
            if kw == 0:
                action = "hold"
            else:
                action = "charge" if kw > 0 else "discharge"
            return decide(action, kw, reason, margin_w)

        def commit(action: str, reason: str, margin_w: float) -> Decision:
            self._commitment = _Commitment(quarter, action, reason)
            return decide(action, self._kw_for(action), reason, margin_w)

        # 1. stale ranking
        if stale:
            return default("stale")
        held = self._commitment
        if held is not None and held.quarter != quarter:
            held = self._commitment = None
        # 2. revoke test of a pre-commit
        if held is not None and held.reason == "precommit" and nowcast is not None:
            if self._revoked(held.action, m_take, m_feed):
                self._commitment = None
                return default("revoked")
        # 3. a commitment stays until the quarter ends
        if held is not None:
            m = m_take if held.action == "charge" else m_feed
            return decide(held.action, self._kw_for(held.action), held.reason, m)
        # 4. quarter boundary
        if e == 0:
            if knobs.boundary_precommit and j == 1 and self._trade_slot(j):
                action, m = self._precommit(j)
                if action is not None:
                    return commit(action, "precommit", m)
                return default("default", margin_w=m)
            return default("default")
        # 5. commit window
        if e <= knobs.latest_commit_min:
            if nowcast is None:
                return default("wait")
            if nowcast.take_q50 / 1000.0 + self.cfg.terms.fee_import_eur_kwh < 0:
                return commit("charge", "negative_take", m_take)
            action, m = ("charge", m_take) if m_take >= m_feed else ("discharge", m_feed)
            if self.commit_test(m):
                return commit(action, "commit", m)
            return default("wait")
        # 6. too late to commit
        return default("late")
