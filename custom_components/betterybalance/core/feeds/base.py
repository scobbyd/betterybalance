"""Shared feed plumbing: HTTP seam, per-day JSON-lines cache and health."""

from __future__ import annotations

import datetime as dt
import http.cookiejar
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ... import __version__

USER_AGENT = f"betterybalance/{__version__} (+https://github.com/scobbyd/betterybalance)"


class FeedError(Exception):
    """Raised when a feed request fails."""


def json_object(payload: bytes, source: str) -> dict:
    """Decode a JSON body that must be an object. Raises FeedError with a body excerpt."""
    excerpt = payload[:80].decode("utf-8", "replace")
    try:
        doc = json.loads(payload)
    except ValueError as exc:
        raise FeedError(f"{source}: body is not JSON: {excerpt!r}") from exc
    if not isinstance(doc, dict):
        raise FeedError(f"{source}: JSON body is not an object: {excerpt!r}")
    return doc


def _open(
    request: urllib.request.Request,
    url: str,
    timeout: float,
    jar: http.cookiejar.CookieJar | None = None,
) -> bytes:
    try:
        if jar is None:
            opened = urllib.request.urlopen(request, timeout=timeout)
        else:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
            opened = opener.open(request, timeout=timeout)
        with opened as response:
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, socket.timeout) as exc:
        raise FeedError(f"request failed: {url}: {exc}") from exc


def _headers(extra: dict | None) -> dict:
    return {"User-Agent": USER_AGENT, **(extra or {})}


def http_get(
    url: str,
    params: dict | None = None,
    timeout: float = 20.0,
    headers: dict | None = None,
    jar: http.cookiejar.CookieJar | None = None,
) -> bytes:
    """GET a URL, optionally with query parameters and a cookie jar. Raises FeedError on failure."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=_headers(headers), method="GET")
    return _open(request, url, timeout, jar)


def http_post(
    url: str,
    data: bytes,
    timeout: float = 20.0,
    headers: dict | None = None,
    jar: http.cookiejar.CookieJar | None = None,
) -> bytes:
    """POST raw bytes to a URL, optionally with a cookie jar. Raises FeedError on failure."""
    request = urllib.request.Request(url, data=data, headers=_headers(headers), method="POST")
    return _open(request, url, timeout, jar)


class DayCache:
    """Per-day cache stored as JSON lines; the last line for a day wins."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _entries(self) -> list[dict]:
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def get(self, day: dt.date) -> list[dict] | None:
        key = day.isoformat()
        rows = None
        for entry in self._entries():
            if entry["day"] == key:
                rows = entry["rows"]
        return rows

    def put(self, day: dt.date, rows: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"day": day.isoformat(), "rows": rows})
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def days(self) -> list[dt.date]:
        return sorted({dt.date.fromisoformat(entry["day"]) for entry in self._entries()})


@dataclass(frozen=True)
class FeedHealth:
    ok: bool
    age_s: float | None
    note: str = ""


def local_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Return every date from start to end, both inclusive."""
    count = (end - start).days + 1
    return [start + dt.timedelta(days=offset) for offset in range(count)]
