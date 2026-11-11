"""A tiny RBI-shaped HTML document and a helper that builds an index from it with the hashing embedder."""

from __future__ import annotations

from pathlib import Path

from codicil.adapters.hashing_embedder import HashingEmbedder
from codicil.application.index import Index, build_index
from codicil.domain.models import DocumentMeta

HTML = """<html><body><table><tr><td>
<p class="head" align="center">Chapter I – Customer Protection</p>
<p class="head">A. Unauthorised Transactions</p>
<p>1. A bank shall credit the amount of an unauthorised electronic transaction to the customer's account within 10 working days from the date of notification.</p>
<p>(1) The customer shall report the transaction within three working days.</p>
<p>Provided that the bank may extend this period for reasons recorded in writing.</p>
<p>2. The liability of a customer is zero where the fraud is due to a deficiency on the part of the bank.</p>
<p class="head">B. Savings Deposits</p>
<p>3. Interest on savings deposits shall be calculated on a daily product basis and credited quarterly.</p>
<p>4. A bank shall issue a passbook to every savings account holder on request without charge.</p>
</td></tr></table></body></html>"""

META = DocumentMeta(
    "mini", "Reserve Bank of India (Mini) Directions, 2026", "(Mini) Directions, 2026", "RBI", "Commercial Banks", "https://example.invalid/mini"
)


def mini_index(out: Path) -> Index:
    return build_index([(META, HTML)], HashingEmbedder(), out)
