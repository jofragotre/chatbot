import json
from typing import List
from schemas import Session, Message


def load_sessions(path: str) -> List[Session]:
    sessions: List[Session] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = []
            for m in obj["messages"]:
                meta = {k: v for k, v in m.items() if k not in ["role", "text"]}
                msgs.append(Message(m["role"], m["text"], meta))
            sessions.append(Session(obj["session_id"], msgs))
    return sessions


def normalize(text: str) -> str:
    return text.lower().strip()


def concat_user_text(sess: Session) -> str:
    return " ".join([m.text for m in sess.messages if m.role == "user"])


def contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def count_any(text: str, terms: List[str]) -> int:
    t = text.lower()
    return sum(1 for term in terms if term in t)
