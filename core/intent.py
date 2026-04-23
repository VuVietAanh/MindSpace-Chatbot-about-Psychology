# ============================================================
# core/intent.py
# Phát hiện ý định của user trước khi generate response
# Rule-based nhanh + LLM fallback cho trường hợp mơ hồ
# ============================================================

import os
import re
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Intent(str, Enum):
    DEFER    = "defer"     # Để sau, không muốn nói ngay
    REJECT   = "reject"    # Từ chối hẳn, muốn dừng
    VENTING  = "venting"   # Xả cảm xúc, không cần hỏi lại
    CONTINUE = "continue"  # Muốn nói tiếp, kể thêm
    QUESTION = "question"  # Đang hỏi chatbot điều gì đó
    GREETING = "greeting"  # Chào hỏi đơn giản
    NEUTRAL  = "neutral"   # Không rõ ý định


@dataclass
class IntentResult:
    intent:     Intent
    confidence: str        # "high" | "medium" | "low"
    raw_text:   str
    reason:     str        # Giải thích tại sao detect intent này


# ── Rule-based patterns ───────────────────────────────────────
DEFER_PATTERNS = [
    r"\bmaybe (tomorrow|later|next time|another time|soon)\b",
    r"\bnot (now|today|right now|yet)\b",
    r"\blater\b",
    r"\bi('ll| will) think (about it|about this)\b",
    r"\bsome other time\b",
    r"\bi('m| am) not ready\b",
    r"\blet me (think|process|sit with this)\b",
    r"\bneed (some time|a moment|space)\b",
    r"\bmaybe\b$",           # Câu chỉ có "maybe"
    r"\bnot sure yet\b",
]

REJECT_PATTERNS = [
    r"\bstop (asking|it|this)\b",
    r"\bleave me alone\b",
    r"\bi don't want to (talk|discuss|share)\b",
    r"\bnone of your business\b",
    r"\bforget it\b",
    r"\bdrop it\b",
    r"\bi('m| am) (done|finished) talking\b",
    r"\bstop\b$",
    r"\bno (more questions|thanks|thank you)\b",
]

VENTING_PATTERNS = [
    # Câu dài (>8 từ) kể lể cảm xúc thường là venting
    # Detect thêm qua keyword
    r"\beverything (is|was|feels?) (wrong|bad|terrible|awful|falling apart)\b",
    r"\bi (just|really) (needed|wanted) to (say|vent|let it out)\b",
    r"\bnobody (understands?|cares?|listens?)\b",
    r"\bi('ve| have) been (holding|keeping) (this|it) (in|back)\b",
    r"\bi don't know (what to do|anymore|how to)\b",
    r"\bi('m| am) (so|really|just) (tired|exhausted|done|over it)\b",
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|sup|yo|hiya|howdy)[.!?]?$",
    r"^good (morning|afternoon|evening|night)[.!?]?$",
    r"^(what's up|wassup|how are you|how's it going)[.!?]?$",
]

QUESTION_PATTERNS = [
    r"^(what|how|why|when|where|who|can you|could you|do you|does|is there|are there)",
    r"\?$",
]


class IntentDetector:

    def __init__(self):
        self._llm = OpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def detect(self, text: str, use_llm: bool = False) -> IntentResult:
        """
        Detect intent từ input của user.
        Rule-based trước, LLM nếu không rõ ràng.
        """
        text_stripped = text.strip()
        text_lower    = text_stripped.lower()
        word_count    = len(text_stripped.split())

        # ── Rule-based detection ─────────────────────────────

        # DEFER
        for pattern in DEFER_PATTERNS:
            if re.search(pattern, text_lower):
                return IntentResult(
                    intent=Intent.DEFER, confidence="high",
                    raw_text=text,
                    reason=f"Matched defer pattern: '{pattern}'"
                )

        # REJECT
        for pattern in REJECT_PATTERNS:
            if re.search(pattern, text_lower):
                return IntentResult(
                    intent=Intent.REJECT, confidence="high",
                    raw_text=text,
                    reason=f"Matched reject pattern: '{pattern}'"
                )

        # GREETING
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, text_lower):
                return IntentResult(
                    intent=Intent.GREETING, confidence="high",
                    raw_text=text, reason="Greeting detected"
                )

        # VENTING — câu dài + keyword pattern
        for pattern in VENTING_PATTERNS:
            if re.search(pattern, text_lower):
                return IntentResult(
                    intent=Intent.VENTING, confidence="high",
                    raw_text=text, reason="Venting pattern detected"
                )

        # Câu rất dài (> 20 từ) thường là venting/kể chuyện
        if word_count > 20:
            return IntentResult(
                intent=Intent.VENTING, confidence="medium",
                raw_text=text, reason="Long message — likely venting or storytelling"
            )

        # QUESTION
        for pattern in QUESTION_PATTERNS:
            if re.search(pattern, text_lower):
                return IntentResult(
                    intent=Intent.QUESTION, confidence="high",
                    raw_text=text, reason="Question detected"
                )

        # ── LLM fallback cho câu ngắn mơ hồ ─────────────────
        if use_llm and word_count <= 10:
            return self._llm_detect(text)

        # Default
        return IntentResult(
            intent=Intent.CONTINUE, confidence="low",
            raw_text=text, reason="No pattern matched — default to continue"
        )

    def _llm_detect(self, text: str) -> IntentResult:
        """LLM classify cho câu ngắn/mơ hồ."""
        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the user's intent in ONE word only.\n"
                            "Options: defer | reject | venting | continue | neutral\n"
                            "defer = wants to postpone or not talk right now\n"
                            "reject = refuses to engage\n"
                            "venting = expressing emotion without wanting advice\n"
                            "continue = wants to keep talking or share more\n"
                            "neutral = unclear\n"
                            "Reply with ONLY the one word. Nothing else."
                        )
                    },
                    {"role": "user", "content": f'Classify: "{text}"'}
                ],
                temperature=0.0,
                max_tokens=5,
            )
            label = response.choices[0].message.content.strip().lower()
            intent_map = {
                "defer":    Intent.DEFER,
                "reject":   Intent.REJECT,
                "venting":  Intent.VENTING,
                "continue": Intent.CONTINUE,
                "neutral":  Intent.NEUTRAL,
            }
            intent = intent_map.get(label, Intent.NEUTRAL)
            return IntentResult(
                intent=intent, confidence="medium",
                raw_text=text, reason=f"LLM classified as: {label}"
            )
        except Exception:
            return IntentResult(
                intent=Intent.NEUTRAL, confidence="low",
                raw_text=text, reason="LLM detection failed"
            )


# ── Response Strategy mapping ─────────────────────────────────
RESPONSE_STRATEGY = {
    Intent.DEFER: {
        "instruction": (
            "The user is deferring — they don't want to talk about this right now. "
            "Respect that completely. Respond with 1-2 warm sentences acknowledging their pace. "
            "Do NOT ask any follow-up question. Do NOT push them. "
            "Something like: 'Of course, whenever you're ready.' and leave space."
        ),
        "max_tokens": 80,
        "temperature": 0.6,
    },
    Intent.REJECT: {
        "instruction": (
            "The user is rejecting the conversation or shutting down. "
            "Respect this immediately. Give a very brief, warm response — 1 sentence max. "
            "Do NOT ask anything. Do NOT explain yourself. Just acknowledge and step back."
        ),
        "max_tokens": 50,
        "temperature": 0.5,
    },
    Intent.VENTING: {
        "instruction": (
            "The user is venting — they need to be heard, NOT questioned or advised. "
            "Your ONLY job right now: reflect what they said, validate the emotion, show you're present. "
            "Do NOT ask any question. Do NOT offer solutions. Do NOT give advice. "
            "Just make them feel heard. 2-3 sentences max. "
            "Example tone: 'That sounds incredibly heavy.' / 'It makes complete sense you'd feel that way.'"
        ),
        "max_tokens": 150,
        "temperature": 0.7,
    },
    Intent.CONTINUE: {
        "instruction": (
            "The user wants to engage. Follow normal conversational flow. "
            "Acknowledge first, then ask ONE thoughtful question to go deeper. "
            "Be warm and curious, not clinical."
        ),
        "max_tokens": 250,
        "temperature": 0.7,
    },
    Intent.QUESTION: {
        "instruction": (
            "The user is asking something. Answer it thoughtfully and warmly. "
            "Keep it conversational — not a lecture. "
            "After answering, you may gently check in on how they're doing, but only if it feels natural."
        ),
        "max_tokens": 300,
        "temperature": 0.7,
    },
    Intent.GREETING: {
        "instruction": (
            "The user is greeting you. Respond warmly and briefly. "
            "Gently invite them to share what's on their mind. 1-2 sentences only."
        ),
        "max_tokens": 80,
        "temperature": 0.8,
    },
    Intent.NEUTRAL: {
        "instruction": (
            "Respond warmly and gently. Match their energy. "
            "If unclear what they need, reflect what you heard and leave space for them to share more. "
            "At most one very gentle question."
        ),
        "max_tokens": 200,
        "temperature": 0.7,
    },
}


# ── Singleton ─────────────────────────────────────────────────
_detector_instance = None

def get_intent_detector() -> IntentDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = IntentDetector()
    return _detector_instance