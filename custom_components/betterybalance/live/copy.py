"""The verdict and status sentences of the state sensor, in English and Dutch (design 14.3).

``verdict`` renders one trader decision to one line: the plan-2 reason, the action and its
power, the threshold it cleared and, for a decision without a grid trade, the next trade
window (with its day when it does not start today). ``status_line`` renders one status flag of
``LiveSite.status()``. Numbers go through ``core.fmt.eu``; times are Europe/Amsterdam ``HH:MM``.
An unknown language reads English.
"""

from __future__ import annotations

import datetime as dt

from ..core.fmt import eu
from ..core.timegrid import ZONE

REASONS = (
    "commit", "precommit", "negative_take", "default", "wait", "late", "stale", "revoked",
    "missing",
)
LANGS = ("en", "nl")
# the status flags of ``LiveSite.status()`` in display order; ``live.site`` imports this one
FLAGS = (
    "learning",
    "live_stale",
    "irradiance_climatology",
    "pv_drift",
    "no_pack",
    "day_ahead_missing",
)
TRADE_REASONS = ("commit", "precommit", "negative_take")

# (reason, action) -> line of a grid trade; {kw} and {th} are formatted numbers
_TRADE = {
    "en": {
        ("commit", "charge"): "Buying {kw} kW this quarter: the price is below {th} ct/kWh.",
        ("commit", "discharge"): "Selling {kw} kW this quarter: the price clears {th} ct/kWh.",
        ("precommit", "charge"): (
            "Buying {kw} kW from the start of this quarter: the forecast is below {th} ct/kWh."
        ),
        ("precommit", "discharge"): (
            "Selling {kw} kW from the start of this quarter: the forecast clears {th} ct/kWh."
        ),
        ("negative_take", "charge"): "Buying {kw} kW this quarter: the take price is below zero.",
    },
    "nl": {
        ("commit", "charge"): "Koopt {kw} kW dit kwartier: de prijs ligt onder {th} ct/kWh.",
        ("commit", "discharge"): "Verkoopt {kw} kW dit kwartier: de prijs ligt boven {th} ct/kWh.",
        ("precommit", "charge"): (
            "Koopt {kw} kW vanaf het begin van dit kwartier: de verwachting ligt onder {th} ct/kWh."
        ),
        ("precommit", "discharge"): (
            "Verkoopt {kw} kW vanaf het begin van dit kwartier: de verwachting ligt boven "
            "{th} ct/kWh."
        ),
        ("negative_take", "charge"): "Koopt {kw} kW dit kwartier: de afnameprijs is negatief.",
    },
}
# the reason before the default policy's clause; "" for none
_LEAD = {
    "en": {
        "commit": "", "precommit": "", "negative_take": "", "default": "",
        "wait": "Waiting for a clear price", "late": "Too late to trade this quarter",
        "stale": "No fresh plan", "revoked": "Pre-commit revoked",
        "missing": "No house data this minute",
    },
    "nl": {
        "commit": "", "precommit": "", "negative_take": "", "default": "",
        "wait": "Wacht op een duidelijke prijs", "late": "Te laat om dit kwartier te handelen",
        "stale": "Geen vers plan", "revoked": "Voorafgaande keuze ingetrokken",
        "missing": "Geen huisdata deze minuut",
    },
}
# the default policy's clause per action
_BODY = {
    "en": {
        "charge": "the battery stores {kw} kW of solar surplus",
        "discharge": "the battery covers {kw} kW of house load",
        "hold": "the battery holds",
    },
    "nl": {
        "charge": "de batterij slaat {kw} kW zonne-overschot op",
        "discharge": "de batterij dekt {kw} kW huisverbruik",
        "hold": "de batterij staat stil",
    },
}
_WINDOW = {
    "en": {
        "charge": " Next buy window {start} to {end}.",
        "discharge": " Next sale window {start} to {end}.",
    },
    "nl": {
        "charge": " Volgend koopvenster {start} tot {end}.",
        "discharge": " Volgend verkoopvenster {start} tot {end}.",
    },
}
# the day of a window that does not start today (Amsterdam): tomorrow, else the weekday
_DAY = {
    "en": ("tomorrow", ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                        "Sunday")),
    "nl": ("morgen", ("maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag",
                      "zondag")),
}
_LAST = {
    "en": {"charge": "Bought {kw} kW at {at}.", "discharge": "Sold {kw} kW at {at}."},
    "nl": {"charge": "{kw} kW gekocht om {at}.", "discharge": "{kw} kW verkocht om {at}."},
}
_STATUS = {
    "en": {
        "learning": "Learning the site models from your data",
        "live_stale": "Live imbalance price older than 3 minutes",
        "irradiance_climatology": "No fresh weather forecast: PV from climatology",
        "pv_drift": "PV forecast off by more than 15 % this week",
        "no_pack": "No data pack installed: the shipped price model serves",
        "day_ahead_missing": "Day-ahead prices missing for part of the window",
    },
    "nl": {
        "learning": "Leert de sitemodellen uit je data",
        "live_stale": "Live onbalansprijs ouder dan 3 minuten",
        "irradiance_climatology": "Geen verse weersverwachting: PV uit klimatologie",
        "pv_drift": "PV-verwachting wijkt deze week meer dan 15 % af",
        "no_pack": "Geen datapakket geïnstalleerd: het meegeleverde prijsmodel rekent",
        "day_ahead_missing": "Day-ahead-prijzen ontbreken voor een deel van het venster",
    },
}


def _lang(lang: str | None) -> str:
    """``lang`` cut to its language part (``nl-BE`` reads ``nl``); English when unknown."""
    base = (lang or "").split("-")[0].split("_")[0].lower()
    return base if base in LANGS else "en"


def clock(ts: int) -> str:
    """``HH:MM`` of epoch ``ts`` in Europe/Amsterdam."""
    return dt.datetime.fromtimestamp(int(ts), ZONE).strftime("%H:%M")


def _day(start: int, now: int | None, lang: str) -> str:
    """The day word and a space before a window start that is not on ``now``'s Amsterdam day
    ("tomorrow", else the weekday); empty for today or without ``now``."""
    if now is None:
        return ""
    day = dt.datetime.fromtimestamp(int(start), ZONE).date()
    today = dt.datetime.fromtimestamp(int(now), ZONE).date()
    if day == today:
        return ""
    tomorrow, weekdays = _DAY[lang]
    return f"{tomorrow if day == today + dt.timedelta(days=1) else weekdays[day.weekday()]} "


def verdict(
    reason: str,
    action: str,
    kw: float,
    threshold_ct: float,
    window: dict | None,
    lang: str,
    now: int | None = None,
) -> str:
    """One line for a decision (module docstring).

    A trade reason with a charge or discharge reads the trade line; every other decision
    reads the reason and the default policy's clause, and the next trade window when
    ``window`` (``{"start", "end", "side"}``, epoch s) is given. With ``now`` (epoch s) a
    window that does not start on the Amsterdam day of ``now`` names its day: "tomorrow",
    else the weekday.
    """
    lang = _lang(lang)
    power = eu(abs(float(kw)), 1)
    line = _TRADE[lang].get((reason, action))
    if line is not None:
        return line.format(kw=power, th=eu(float(threshold_ct), 2))
    body = _BODY[lang].get(action, _BODY[lang]["hold"]).format(kw=power)
    lead = _LEAD[lang].get(reason, "")
    text = f"{lead}: {body}." if lead else f"{body[0].upper()}{body[1:]}."
    if window is not None and window.get("side") in _WINDOW[lang]:
        text += _WINDOW[lang][window["side"]].format(
            start=_day(window["start"], now, lang) + clock(window["start"]),
            end=clock(window["end"]),
        )
    return text


def trade_line(action: str, kw: float, ts: int, lang: str) -> str:
    """The last grid trade as one line: side, power and the Amsterdam time of its minute."""
    lang = _lang(lang)
    line = _LAST[lang].get(action)
    if line is None:
        return ""
    return line.format(kw=eu(abs(float(kw)), 1), at=clock(ts))


def status_line(flag: str, lang: str) -> str:
    """The sentence of one status flag of ``FLAGS``."""
    return _STATUS[_lang(lang)][flag]
