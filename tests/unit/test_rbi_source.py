import httpx
import pytest

from codicil.adapters.rbi_source import RbiFetcher, pdf_stem
from codicil.application.refresh import SourceUnavailable

LISTING = '<table><tr><td><a href="BS_ViewMasDirections.aspx?id=123">x</a></td><td><a href="https://rbidocs.rbi.org.in/rdocs/PDFs/ABC123.PDF">pdf</a></td></tr></table>'


def fetcher(handler):
    return RbiFetcher(pause_s=0, client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))


def test_pdf_stem_and_listing_pairing():
    assert pdf_stem("https://x/PDFs/abc123.pdf") == "ABC123"


def test_fetches_the_detail_page_of_the_matching_listing_row():
    seen = []

    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(200, text=LISTING if "id=" not in str(req.url) else "<html>the page</html>")

    with fetcher(handler) as f:
        assert f.fetch({"doc_id": "d", "pdf_url": "https://rbidocs.rbi.org.in/rdocs/PDFs/ABC123.PDF"}) == b"<html>the page</html>"
    assert seen[0].endswith("BS_ViewMasDirections.aspx") and seen[1].endswith("id=123")


def test_bot_challenge_and_missing_documents_and_http_errors_are_unavailable_not_content():
    def challenge(req):
        return httpx.Response(200, text=LISTING if "id=" not in str(req.url) else "<html>TSPD challenge</html>")

    with fetcher(challenge) as f, pytest.raises(SourceUnavailable, match="bot challenge"):
        f.fetch({"doc_id": "d", "pdf_url": "https://x/PDFs/ABC123.PDF"})
    with fetcher(lambda req: httpx.Response(200, text=LISTING)) as f, pytest.raises(SourceUnavailable, match="no page"):
        f.fetch({"doc_id": "d", "pdf_url": "https://x/PDFs/NOPE.PDF"})
    with fetcher(lambda req: httpx.Response(503)) as f, pytest.raises(SourceUnavailable):
        f.fetch({"doc_id": "d", "pdf_url": "https://x/PDFs/ABC123.PDF"})
