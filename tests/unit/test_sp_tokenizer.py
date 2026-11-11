from pathlib import Path

import pytest

from codicil.adapters.sp_tokenizer import SentencePieceTokenizer

MODEL = Path("data/models/multilingual-e5-small/sentencepiece.bpe.model")

# ids produced by the Hugging Face tokenizer.json of the same model; this tokenizer must reproduce them exactly
GOLDEN = {
    "query: What is the daily limit?": [0, 41, 1294, 12, 4865, 83, 70, 31815, 17475, 32, 2],
    "passage: बैंक को KYC अपडेट कितनी बार करना चाहिए?": [0, 46692, 12, 16945, 629, 55771, 441, 60578, 183344, 12797, 16835, 23508, 32, 2],
    "query: auto debit se pehle notification kitne ghante pehle?": [
        0,
        41,
        1294,
        12,
        1809,
        8,
        3137,
        40,
        280,
        7057,
        185325,
        24448,
        86,
        31380,
        2733,
        280,
        7057,
        32,
        2,
    ],
    "₹15,000/- per transaction  ’quoted’": [0, 87316, 418, 26738, 64, 9, 117, 124161, 6, 26, 68726, 71, 26, 2],
}

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_models.py first")


def test_matches_hugging_face_ids():
    tok = SentencePieceTokenizer(MODEL)
    for text, want in GOLDEN.items():
        assert tok.encode_batch([text])[0].ids == want


def test_batch_is_padded_and_masked():
    a, b = SentencePieceTokenizer(MODEL).encode_batch(["hello", "hello hello hello world"])
    assert len(a.ids) == len(b.ids) and a.ids[-1] == 1 and a.attention_mask[-1] == 0 and b.attention_mask[-1] == 1
    assert sum(a.attention_mask) < sum(b.attention_mask)


def test_truncation_keeps_start_and_end_tokens():
    ids = SentencePieceTokenizer(MODEL, max_length=8).encode_batch(["word " * 50])[0].ids
    assert len(ids) == 8 and ids[0] == 0 and ids[-1] == 2
