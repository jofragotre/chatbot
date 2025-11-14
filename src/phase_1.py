# src/phase1_pipeline.py
# Prettier-style formatting, <=80 chars per line.

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import dateparser.search as dp_search
import joblib
import numpy as np
import pandas as pd
import spacy
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from schemas import Message, Session, Evidence
from utils import load_sessions, normalize, concat_user_text, contains_any, count_any
from lexicons import *


def find_dates(text: str) -> List[str]:
    hits = []
    try:
        found = dp_search.search_dates(text, languages=["en"])
        if found:
            hits.extend([span for span, _ in found])
    except Exception:
        pass
    for token in DATE_LIKE_EXTRA:
        if token in text.lower():
            hits.append(token)
    # Simple range like "12–14 Oct" or "12-14 Oct"
    range_pat = re.compile(
        r"\b\d{1,2}\s*[–-]\s*\d{1,2}\s*[A-Za-z]{3,9}\b", re.IGNORECASE
    )
    hits.extend(range_pat.findall(text))
    return list(set(hits))


def find_money(text: str) -> List[str]:
    return MONEY_RE.findall(text)


def find_emails(text: str) -> List[str]:
    return EMAIL_RE.findall(text)


# -------------------------
# Evidence extraction
# -------------------------




def collect_evidence(sess: Session) -> List[Evidence]:
    ev: List[Evidence] = []
    for i, m in enumerate(sess.messages):
        t = m.text
        if find_emails(t):
            ev.append(Evidence("email", i, t))
        d = find_dates(t)
        if d:
            ev.append(Evidence("date", i, ", ".join(d)))
        money = find_money(t)
        if money:
            ev.append(Evidence("money", i, ", ".join(money)))
        if contains_any(t, BOOKING_VERBS):
            ev.append(Evidence("booking_verb", i, t))
        if contains_any(t, PAYMENT_WORDS):
            ev.append(Evidence("payment", i, t))
        if contains_any(t, ABANDON_PHRASES):
            ev.append(Evidence("abandon", i, t))
        if contains_any(t, ROOM_WORDS):
            ev.append(Evidence("room", i, t))
        if contains_any(t, OCCUPANCY_WORDS):
            ev.append(Evidence("occupancy", i, t))
        if contains_any(t, SERVICE_WORDS):
            ev.append(Evidence("service", i, t))
        if contains_any(t, POLICY_OR_INFO):
            ev.append(Evidence("policy_info", i, t))

    # Booking meta flags from bot
    for i, m in enumerate(sess.messages):
        if m.role == "bot" and ("booking" in m.meta):
            if m.meta["booking"] == "completed":
                ev.append(Evidence("booking_completed", i, "meta:completed"))
    for i, m in enumerate(sess.messages):
        if m.role == "bot":
            if "payment processed" in m.text.lower():
                ev.append(Evidence("booking_completed", i, m.text))
            if "payment complete" in m.text.lower():
                ev.append(Evidence("booking_completed", i, m.text))
            if "booking confirmed" in m.text.lower():
                ev.append(Evidence("booking_completed", i, m.text))
    return ev


# -------------------------
# Rule-based labeling
# -------------------------


def rule_label(sess: Session) -> Tuple[str, List[Evidence]]:
    ev = collect_evidence(sess)

    # Quick helpers
    def has(kind: str) -> bool:
        return any(e.kind == kind for e in ev)

    def after(index_a: int, cond_b) -> bool:
        # Is there an evidence matching cond_b after index_a?
        for e in ev:
            if e.message_idx > index_a and cond_b(e):
                return True
        return False

    # 1) Completed
    if has("booking_completed"):
        return "high_completed", ev

    # 2) Abandoned: payment asked, then user says finish later OR ends w/o complete
    pay_req_idx = None
    for e in ev:
        if e.kind == "payment":
            pay_req_idx = e.message_idx
    user_abandon = any(e.kind == "abandon" for e in ev)
    if pay_req_idx is not None and user_abandon:
        return "high_abandoned", ev
    # Also: payment/hold mentioned near the end with no completion
    if pay_req_idx is not None and not has("booking_completed"):
        # If last bot mentions hold rate or enter card and convo ends
        last_bot_text = ""
        for m in reversed(sess.messages):
            if m.role == "bot":
                last_bot_text = m.text.lower()
                break
        if "hold the rate" in last_bot_text or "enter card" in last_bot_text:
            return "high_abandoned", ev

    # 3) High actioning: book/proceed or PII after ask
    user_text = concat_user_text(sess).lower()
    has_book_intent = any(
        kw in user_text for kw in ["book", "reserve", "proceed", "i'll take"]
    )
    provided_email = EMAIL_RE.search(user_text) is not None
    asked_name_email = any(
        "name and email" in m.text.lower() and m.role == "bot"
        for m in sess.messages
    )
    if (has_book_intent and (find_dates(user_text) or contains_any(
        user_text, ROOM_WORDS
    ))) or (asked_name_email and provided_email):
        return "high_actioning", ev

    # 4) Medium: dates + (price or room/cancellation/availability)
    has_dates_flag = any(e.kind == "date" for e in ev)
    has_price_flag = any(e.kind == "money" for e in ev) or contains_any(
        user_text, PRICE_WORDS
    )
    discussing_rooms = contains_any(user_text, ROOM_WORDS)
    discussing_avail = contains_any(user_text, AVAILABILITY_WORDS)
    discussing_cancel = "cancellation" in user_text
    if has_dates_flag and (has_price_flag or discussing_rooms or discussing_avail
                           or discussing_cancel):
        return "medium_evaluating", ev

    # 5) None (service)
    discussing_service = any(e.kind == "service" for e in ev)
    if discussing_service:
        return "none", ev

    # 6) Low (exploring/policy/info)
    if any(e.kind == "policy_info" for e in ev):
        return "low_exploring", ev

    # 7) Fallback
    return "low_exploring", ev


# -------------------------
# Feature extraction for ML
# -------------------------


def session_features(sess: Session) -> Dict[str, Any]:
    text_user = concat_user_text(sess)
    text_all = " ".join([m.text for m in sess.messages])
    nl = normalize(text_all)
    ev = collect_evidence(sess)
    kinds = [e.kind for e in ev]

    feats = {
        "session_id": sess.session_id,
        "text_user": text_user,
        "n_user_turns": sum(1 for m in sess.messages if m.role == "user"),
        "n_bot_turns": sum(1 for m in sess.messages if m.role == "bot"),
        "has_dates": int(any(k == "date" for k in kinds)),
        "has_money": int(any(k == "money" for k in kinds)),
        "has_email": int(any(k == "email" for k in kinds)),
        "mentions_booking_verb": int(any(k == "booking_verb" for k in kinds)),
        "mentions_payment": int(any(k == "payment" for k in kinds)),
        "mentions_abandon": int(any(k == "abandon" for k in kinds)),
        "mentions_room": int(any(k == "room" for k in kinds)),
        "mentions_occupancy": int(any(k == "occupancy" for k in kinds)),
        "mentions_service": int(any(k == "service" for k in kinds)),
        "mentions_policy_info": int(any(k == "policy_info" for k in kinds)),
        "asked_name_email": int(
            any(
                m.role == "bot" and "name and email" in m.text.lower()
                for m in sess.messages
            )
        ),
        "user_provided_email": int(EMAIL_RE.search(text_user) is not None),
        "has_availability_words": int(contains_any(nl, AVAILABILITY_WORDS)),
        "has_price_words": int(contains_any(nl, PRICE_WORDS)),
        "has_room_words": int(contains_any(nl, ROOM_WORDS)),
        "has_payment_words": int(contains_any(nl, PAYMENT_WORDS)),
    }
    return feats


def build_dataframe(sessions: List[Session]) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    labels = []
    for s in sessions:
        label, _ = rule_label(s)
        labels.append(label)
        feats = session_features(s)
        rows.append(feats)
    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y


# -------------------------
# ML pipeline (optional)
# -------------------------


def build_model() -> Pipeline:
    numeric_cols = [
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
    pre = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000),
             "text_user"),
            ("num", StandardScaler(with_mean=False), numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    clf = LogisticRegression(
        max_iter=1000, solver="liblinear", multi_class="ovr"
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe


def crossval_report(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> None:
    skf = StratifiedKFold(n_splits=min(5, len(y)))
    y_pred = cross_val_predict(model, X, y, cv=skf)
    print(classification_report(y, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))


# -------------------------
# Inference with explanations
# -------------------------


def predict_with_rules_and_model(
    sess: Session, model: Pipeline | None
) -> Dict[str, Any]:
    # 1) Rule overrides for completed/abandoned
    label, ev = rule_label(sess)
    if label in ["high_completed", "high_abandoned"]:
        return {
            "session_id": sess.session_id,
            "label": label,
            "source": "rules",
            "evidence": [e.__dict__ for e in ev][:5],
            "proba": 1.0,
        }

    # 2) If model is provided, use it; else fall back to rules
    if model is not None:
        feats = session_features(sess)
        X = pd.DataFrame([feats])
        # add required text column
        X["text_user"] = concat_user_text(sess)
        proba = model.predict_proba(X)[0]
        classes = model.named_steps["clf"].classes_
        idx = int(np.argmax(proba))
        return {
            "session_id": sess.session_id,
            "label": str(classes[idx]),
            "source": "model",
            "proba": float(proba[idx]),
            "evidence": [e.__dict__ for e in ev][:5],
        }
    else:
        return {
            "session_id": sess.session_id,
            "label": label,
            "source": "rules",
            "evidence": [e.__dict__ for e in ev][:5],
            "proba": None,
        }



def main():
    
    data_path = "data/provided/small_sample.jsonl"
    mode = "train"
    model_out = "artifacts/phase1_model.joblib"
    model_in = "artifacts/phase1_model.joblib"

    os.makedirs("artifacts", exist_ok=True)
    sessions = load_sessions(data_path)

    if mode == "label":
        # Rules-only labeling preview
        rows = []
        for s in sessions:
            label, ev = rule_label(s)
            rows.append(
                {
                    "session_id": s.session_id,
                    "label": label,
                    "evidence": "; ".join(
                        f"{e.kind}@{e.message_idx}" for e in ev[:5]
                    ),
                }
            )
        df = pd.DataFrame(rows)
        print(df)
        return

    if mode == "train":
        X, y = build_dataframe(sessions)
        model = build_model()
        print("Cross-validation report (weakly supervised labels):")
        X.to_csv("./artifacts/train_data.csv")
        crossval_report(model, X, y)
        model.fit(X, y)
        joblib.dump(model, model_out)
        print(f"Saved model to {model_out}")
        return

    if mode == "predict":
        model = joblib.load(model_in)
        outputs = []
        for s in sessions:
            out = predict_with_rules_and_model(s, model)
            outputs.append(out)
        df = pd.DataFrame(outputs)
        df.to_csv("./artifacts/outputs.csv")
        return

if __name__ == "__main__":
    main()