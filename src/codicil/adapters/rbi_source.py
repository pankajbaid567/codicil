"""Fetching RBI Master Direction pages. Polite by design: honest User-Agent, sequential requests, a pause between them, no retries,
and it stops (rather than tries to get around it) when the site answers with a bot-challenge page."""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from codicil.application.refresh import SourceUnavailable

BASE = "https://rbi.org.in/Scripts/"
LISTING = BASE + "BS_ViewMasDirections.aspx"
UA = "codicil-research/0.1 (student project; +https://github.com/pankajbaid567/codicil)"
PAUSE_S = 2.0


def listing_index(client: httpx.Client) -> dict[str, str]:
    """Map PDF file stem -> detail page id, pairing the two links that share a table row of the listing page."""
    page = client.get(LISTING)
    page.raise_for_status()
    pairs: dict[str, str] = {}
    for row in re.findall(r"<tr>.*?</tr>", page.text, flags=re.S):
        detail = re.search(r"BS_ViewMasDirections\.aspx\?id=(\d+)", row)
        pdf = re.search(r"/PDFs/([^'\"/]+)\.PDF", row, flags=re.I)
        if detail and pdf:
            pairs[pdf.group(1).upper()] = detail.group(1)
    return pairs


def pdf_stem(url: str) -> str:
    return url.rsplit("/", 1)[-1].rsplit(".", 1)[0].upper()


class RbiFetcher:
    """`Fetcher` over the live site. Use as a context manager."""

    def __init__(self, pause_s: float = PAUSE_S, timeout: float = 90.0, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=timeout)
        self._pause, self._first = pause_s, True
        self._index: dict[str, str] | None = None

    def __enter__(self) -> RbiFetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def fetch(self, entry: dict[str, Any]) -> bytes:
        try:
            if self._index is None:
                self._index = listing_index(self._client)
            pdf = str(entry.get("pdf_url") or entry["url"])
            page_id = self._index.get(pdf_stem(pdf))
            if page_id is None:
                raise SourceUnavailable(f"no page for {entry['doc_id']} in the listing (pdf stem {pdf_stem(pdf)})")
            if not self._first:
                time.sleep(self._pause)
            self._first = False
            r = self._client.get(f"{BASE}BS_ViewMasDirections.aspx?id={page_id}")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise SourceUnavailable(f"{type(e).__name__}: {e}") from e
        if "TSPD" in r.text[:4000]:
            raise SourceUnavailable("bot challenge page returned; not attempting to bypass")
        return r.content
