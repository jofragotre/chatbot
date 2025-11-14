import re

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
DATE_LIKE_EXTRA = [
    "tonight",
    "today",
    "tomorrow",
    "this weekend",
    "weekend",
    "next week",
]