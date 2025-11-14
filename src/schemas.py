from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class Message:
    role: str
    text: str
    meta: Dict[str, Any]


@dataclass
class Session:
    session_id: str
    messages: List[Message]

@dataclass
class Evidence:
    kind: str
    message_idx: int
    text_snippet: str
