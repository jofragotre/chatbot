# scripts/export_transformed_features.py
# Export the final, model-ready feature matrix (TF-IDF + scaled numerics) to CSV.

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import dateparser.search as dp_search
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# -------------------------
# Config (edit paths/params)
# -------------------------

INPUT_JSONL = "data/provided/small_sample.jsonl"
OUTPUT_CSV = "artifacts/features.csv"
OUTPUT_FEATURE_NAMES = "artifacts/feature_names.txt"

# Keep this aligned with your training pipeline
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)

# -------------------------
# Lexicons/patterns (same set used in Phase 1)
# -------------------------

BOOKING_VERBS = [
    "book",
    "reserve",
    "proceed",
    "confirm",
    "hold a room",
    "hold the rate",
]
PAYMENT_WORDS = [
    "payment",
    "pay",
    "card",
    "credit card",
    "enter card",
    "guarantee",
    "pre-author",
    "preauthor",
    "secure link",
]
ABANDON_PHRASES = [
    "finish later",
    "i'll finish later",
    "i will finish later",
    "later",
]
AVAILABILITY_WORDS = ["availability", "available", "do you have", "have rooms"]
PRICE_WORDS = ["price", "cost", "rate", "€", "eur", "$", "usd", "per night"]
ROOM_WORDS = ["room", "deluxe", "single", "superior", "suite", "balcony", "view"]
OCCUPANCY_WORDS = ["adult", "adults", "child", "children", "kids", "guests"]
POLICY_OR_INFO = [
    "policy",
    "check-in",
    "check in",
    "check-out",
    "check out",
    "cancellation",
    "pets",
    "pet",
    "wifi",
    "wi-fi",
    "parking",
    "shuttle",
    "airport",
    "accessible",
    "step-free",
    "invoice",
]
SERVICE_WORDS = [
    "towel",
    "pillows",
    "housekeeping",
    "ac",
    "aircon",
    "maintenance",
    "router",
    "ticket",
    "lost",
    "found",
    "umbrella",
]

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE
)
MONEY_RE = re.compile(
    r"(?:€|\$)\s?\d+(?:[.,]\d{1,2})?|"
    r"\b\d+(?:[.,]\d{1,2})?\s?(?:eur|usd)\b",
    re.IGNORECASE,
)
DATE_RANGE_RE = re.compile(
    r"\b\d{1,2}\s*[–-]\s*\d{1,2}\s*[A-Za-z]{3,9}\b", re.IGNORECASE
)
DATE_LIKE_EXTRA = [
    "tonight",
    "today",
    "tomorrow",
    "this weekend",
    "weekend",
    "next week",
]

# -------------------------
# Helpers to load and featurize
# -------------------------


def load_sessions(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def concat_user_text(messages: List[Dict[str, Any]]) -> str:
    return " ".join([m.get("text", "") for m in messages if m.get("role") == "user"])


def contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def find_dates(text: str) -> List[str]:
    hits: List[str] = []
    try:
        found = dp_search.search_dates(text, languages=["en"])
        if found:
            hits.extend([span for span, _ in found])
    except Exception:
        pass
    for token in DATE_LIKE_EXTRA:
        if token in text.lower():
            hits.append(token)
    hits.extend(DATE_RANGE_RE.findall(text))
    # de-duplicate
    seen = set()
    out: List[str] = []
    for h in hits:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out


def find_money(text: str) -> List[str]:
    return MONEY_RE.findall(text)


def find_emails(text: str) -> List[str]:
    return EMAIL_RE.findall(text)


def session_numeric_features(sess: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    msgs = sess["messages"]
    text_user = concat_user_text(msgs)
    text_all = " ".join([m.get("text", "") for m in msgs])
    nl = text_all.lower()

    feats = {
        "n_user_turns": int(sum(1 for m in msgs if m.get("role") == "user")),
        "n_bot_turns": int(sum(1 for m in msgs if m.get("role") == "bot")),
        "has_dates": int(bool(find_dates(text_all))),
        "has_money": int(bool(find_money(text_all))),
        "has_email": int(bool(find_emails(text_all))),
        "mentions_booking_verb": int(contains_any(nl, BOOKING_VERBS)),
        "mentions_payment": int(contains_any(nl, PAYMENT_WORDS)),
        "mentions_abandon": int(contains_any(nl, ABANDON_PHRASES)),
        "mentions_room": int(contains_any(nl, ROOM_WORDS)),
        "mentions_occupancy": int(contains_any(nl, OCCUPANCY_WORDS)),
        "mentions_service": int(contains_any(nl, SERVICE_WORDS)),
        "mentions_policy_info": int(contains_any(nl, POLICY_OR_INFO)),
        "asked_name_email": int(
            any(
                (m.get("role") == "bot")
                and ("name and email" in m.get("text", "").lower())
                for m in msgs
            )
        ),
        "user_provided_email": int(EMAIL_RE.search(text_user or "") is not None),
        "has_availability_words": int(contains_any(nl, AVAILABILITY_WORDS)),
        "has_price_words": int(contains_any(nl, PRICE_WORDS)),
        "has_room_words": int(contains_any(nl, ROOM_WORDS)),
        "has_payment_words": int(contains_any(nl, PAYMENT_WORDS)),
    }
    return feats, text_user


NUMERIC_COLS = [
    "n_user_turns",
    "n_bot_turns",
    "has_dates",
    "has_money",
    "has_email",
    "mentions_booking_verb",
    "mentions_payment",
    "mentions_abandon",
    "mentions_room",
    "mentions_occupancy",
    "mentions_service",
    "mentions_policy_info",
    "asked_name_email",
    "user_provided_email",
    "has_availability_words",
    "has_price_words",
    "has_room_words",
    "has_payment_words",
]


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    sessions = load_sessions(INPUT_JSONL)
    if not sessions:
        raise SystemExit("No sessions found in input JSONL.")

    # Build numeric features and text field
    rows: List[Dict[str, Any]] = []
    texts: List[str] = []
    session_ids: List[str] = []
    for s in sessions:
        feats, text_user = session_numeric_features(s)
        rows.append(feats)
        texts.append(text_user or "")
        session_ids.append(s["session_id"])

    X_num = pd.DataFrame(rows, columns=NUMERIC_COLS).astype(float)
    # Fit TF-IDF on user text
    tfidf = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE, max_features=TFIDF_MAX_FEATURES
    )
    X_tfidf = tfidf.fit_transform(texts)  # sparse CSR

    # Scale numeric features (variance scaling only to keep it sparse-friendly)
    scaler = StandardScaler(with_mean=False)
    X_num_scaled = scaler.fit_transform(X_num.values)  # dense ndarray

    # Combine: [TF-IDF | scaled numerics]
    X_combined = hstack([X_tfidf, csr_matrix(X_num_scaled)], format="csr")

    # Build feature names
    tfidf_names = [f"tfidf:{t}" for t in tfidf.get_feature_names_out()]
    num_names = [f"num:{c}" for c in NUMERIC_COLS]
    feature_names = tfidf_names + num_names

    # Convert to DataFrame for CSV (dense; fine for small datasets)
    X_dense = X_combined.toarray()
    df_out = pd.DataFrame(X_dense, columns=feature_names)
    df_out.insert(0, "session_id", session_ids)

    # Write outputs
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    with open(OUTPUT_FEATURE_NAMES, "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    print(
        "Wrote matrix with shape "
        f"{X_combined.shape} to {OUTPUT_CSV} and names to {OUTPUT_FEATURE_NAMES}"
    )
    print(
        "Note: CSV is dense. For large datasets, prefer saving the sparse "
        "matrix via scipy.sparse.save_npz and feature names separately."
    )


if __name__ == "__main__":
    main()