"""Conversation state and lightweight NLP helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional

try:
    from textblob import TextBlob  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TextBlob = None  # type: ignore


@dataclass
class ConversationTurn:
    user: str
    assistant: str
    intent: Optional[str] = None
    sentiment: str = "neutral"
    dialog_act: str = "statement"


@dataclass
class PersonalityProfile:
    name: str = "default"
    tone: str = "professional"
    traits: Dict[str, str] = field(default_factory=lambda: {
        "professional": "Keep responses concise, factual, and project-oriented.",
        "empathetic": "Acknowledge user concerns with care before providing facts.",
        "casual": "Keep the tone light while staying ready to pivot back to project details.",
    })


class ConversationManager:
    def __init__(self, max_turns: int = 6, personality: Optional[PersonalityProfile] = None) -> None:
        self._history: Deque[ConversationTurn] = deque(maxlen=max_turns)
        self.personality = personality or PersonalityProfile()

    # ------------------------------------------------------------------
    def record_turn(self, turn: ConversationTurn) -> None:
        self._history.appendleft(turn)

    def history(self) -> List[ConversationTurn]:
        return list(self._history)

    def classify_dialog_act(self, text: str) -> str:
        stripped = text.strip().lower()
        if not stripped:
            return "silence"
        if stripped.endswith("?"):
            return "question"
        if stripped.startswith(("please", "could you", "can you")):
            return "request"
        if any(token in stripped for token in ("thanks", "thank you", "appreciate")):
            return "gratitude"
        return "statement"

    def analyse_sentiment(self, text: str) -> str:
        if not text.strip():
            return "neutral"
        if TextBlob is None:
            return "neutral"
        try:
            polarity = TextBlob(text).sentiment.polarity
        except Exception:
            return "neutral"
        if polarity > 0.2:
            return "positive"
        if polarity < -0.2:
            return "negative"
        return "neutral"

    def stage(self) -> str:
        turns = len(self._history)
        if turns == 0:
            return "early"
        if turns < 4:
            return "mid"
        return "late"

    def engagement_level(self) -> str:
        if not self._history:
            return "normal"
        questions = sum(1 for turn in self._history if turn.dialog_act == "question")
        if questions >= max(1, len(self._history) // 2):
            return "high"
        return "normal"

    def get_context_blob(self, limit: int = 4) -> str:
        lines: List[str] = []
        for idx, turn in enumerate(list(self._history)[:limit][::-1], start=1):
            lines.append(f"User {idx}: {turn.user}")
            lines.append(f"Assistant {idx}: {turn.assistant}")
        return "\n".join(lines)


__all__ = ["ConversationManager", "ConversationTurn", "PersonalityProfile"]
