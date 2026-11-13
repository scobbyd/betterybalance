"""European number formatting for reports."""

from __future__ import annotations

import math

NA = "n.v.t."


def eu(x: float, dec: int = 2) -> str:
    """Format x with a dot as thousands separator and a comma as decimal mark.

    A value that rounds to zero at ``dec`` decimals prints without a sign ("0,00", never
    "-0,00").
    """
    if x is None or not math.isfinite(x):
        return NA
    s = f"{x:,.{dec}f}"
    if s.startswith("-") and not any(c in "123456789" for c in s):
        s = s[1:]
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
