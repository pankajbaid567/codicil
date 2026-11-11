"""Hindi / Hinglish banking vocabulary -> canonical English, plus script and language detection. Curated by hand; deliberately small."""

from __future__ import annotations

import re

from codicil.domain.models import Query

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# multi-word keys first (matched on the lower-cased, space-normalised query)
GLOSSARY: dict[str, str] = {
    # devanagari
    "बैंक": "bank",
    "ब्याज दर": "interest rate",
    "ब्याज": "interest",
    "ऋण": "loan",
    "कर्ज": "loan",
    "कर्ज़": "loan",
    "खाता": "account",
    "जमा": "deposit",
    "धोखाधड़ी": "fraud",
    "शिकायत": "complaint",
    "ग्राहक": "customer",
    "क्रेडिट कार्ड": "credit card",
    "डेबिट कार्ड": "debit card",
    "शुल्क": "charges",
    "जुर्माना": "penalty",
    "देयता": "liability",
    "लेनदेन": "transaction",
    "अनधिकृत": "unauthorised",
    "वापसी": "refund",
    "रिफंड": "refund",
    "केवाईसी": "kyc",
    "पासवर्ड": "password",
    "समय सीमा": "timeline",
    "कितना": "how much",
    "कितने दिन": "how many days",
    "क्या": "what",
    "नियम": "rules",
    "आरबीआई": "rbi",
    "रिज़र्व बैंक": "reserve bank",
    # romanised
    "byaj dar": "interest rate",
    "byaj": "interest",
    "brij": "interest",
    "karz": "loan",
    "karza": "loan",
    "loan": "loan",
    "khata": "account",
    "khate": "account",
    "jama": "deposit",
    "dhokhadhadi": "fraud",
    "dhokha": "fraud",
    "fraud": "fraud",
    "shikayat": "complaint",
    "grahak": "customer",
    "jurmana": "penalty",
    "shulk": "charges",
    "paisa wapas": "refund",
    "paise wapas": "refund",
    "wapas": "refund",
    "chori": "theft",
    "galat": "unauthorised",
    "bina permission": "unauthorised",
    "bina pooche": "unauthorised",
    "kat gaye": "debited",
    "kat gaya": "debited",
    "kaat liye": "debited",
    "kaat liya": "debited",
    "kitne din": "how many days",
    "kitna": "how much",
    "kitni": "how much",
    "kab tak": "timeline",
    "samay seema": "timeline",
    "niyam": "rules",
    "kya": "what",
    "bank ne": "bank",
    "mera": "my",
    "meri": "my",
    "mere": "my",
    "nahi": "not",
    "nahin": "not",
    "kar sakta": "can",
    "sakta hai": "can",
    "lena": "charge",
    "lagta hai": "charged",
    "lagega": "charged",
    "late payment": "late payment",
    "emi": "emi",
    "band": "close",
    "khaata": "account",
    "bachat khata": "savings account",
    "chalu khata": "current account",
    "nominee": "nominee",
    "adhar": "aadhaar",
    "aadhar": "aadhaar",
    "pan card": "pan",
    "credit card": "credit card",
    "debit card": "debit card",
    "upi": "upi",
    "atm": "atm",
    "otp": "otp",
    "cheque": "cheque",
    "kyc": "kyc",
    "wallet": "wallet",
}
_HINGLISH_MARKERS = frozenset(
    [
        "kya",
        "hai",
        "hain",
        "nahi",
        "nahin",
        "mera",
        "meri",
        "mere",
        "kitna",
        "kitni",
        "kitne",
        "kab",
        "kaise",
        "kyun",
        "bank",
        "ne",
        "ko",
        "se",
        "mai",
        "main",
        "hum",
        "aap",
        "tak",
        "par",
        "pe",
        "liye",
        "karna",
        "karne",
        "kar",
        "sakta",
        "sakte",
        "lagta",
        "lagega",
        "wapas",
        "paisa",
        "paise",
        "khata",
        "karz",
        "byaj",
        "shikayat",
        "kat",
        "gaye",
        "gaya",
        "galat",
        "bina",
    ]
)
_ENTITY = [
    (re.compile(r"\bnbfcs?\b|non[- ]banking", re.I), "NBFCs"),
    (re.compile(r"payments? banks?", re.I), "Payments Banks"),
    (re.compile(r"\bppis?\b|prepaid|wallet", re.I), "Prepaid Payment Instrument issuers"),
    (re.compile(r"small finance bank", re.I), "Small Finance Banks"),
    (re.compile(r"commercial bank", re.I), "Commercial Banks"),
]


def detect_language(text: str) -> str:
    if _DEVANAGARI.search(text):
        return "hi"
    toks = re.findall(r"[a-z]+", text.lower())
    if not toks:
        return "en"
    hits = sum(t in _HINGLISH_MARKERS for t in toks if t not in {"bank", "loan", "kya"} or t == "kya")
    return "hinglish" if hits >= 2 and hits / len(toks) >= 0.2 else "en"


def normalise_query(raw: str) -> Query:
    text = " ".join(raw.strip().split())
    lang = detect_language(text)
    hints = tuple(dict.fromkeys(name for pat, name in _ENTITY if pat.search(text)))
    if lang == "en":
        return Query(raw=raw, normalised=text, language=lang, entity_hints=hints)
    low = text.lower()
    matched: list[tuple[str, str]] = []
    for key in sorted(GLOSSARY, key=len, reverse=True):
        pat = re.compile(rf"(?<![\wऀ-ॿ]){re.escape(key)}(?![\wऀ-ॿ])")
        if pat.search(low):
            matched.append((key, GLOSSARY[key]))
            low = pat.sub(f" {GLOSSARY[key]} ", low)
    canon = " ".join(low.split())
    # keep the original words too: product names and English terms inside a Hinglish sentence still match lexically
    return Query(raw=raw, normalised=f"{canon} {text}" if canon != text.lower() else text, language=lang, entity_hints=hints, matched_terms=tuple(matched))
