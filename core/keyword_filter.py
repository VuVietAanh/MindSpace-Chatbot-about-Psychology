# ============================================================
# core/keyword_filter.py
# Lọc từ khóa nguy hiểm — có xử lý phủ định (negation)
# ============================================================

import re
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    SAFE     = "safe"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class FilterResult:
    is_flagged:       bool
    risk_level:       RiskLevel
    matched_keywords: list[str]
    category:         str | None
    message:          str | None


# ── Negation words — nếu có trước keyword thì KHÔNG flag ────
NEGATION_WORDS = [
    # basic
    "not", "no", "never", "none", "nothing", "nobody", "nowhere",

    # verbs
    "don't", "doesn't", "didn't", "won't", "wouldn't",
    "cannot", "can't", "couldn't", "shouldn't", "mustn't",
    "haven't", "hasn't", "hadn't",
    "aren't", "isn't", "wasn't", "weren't", "ain't",

    # connectors
    "neither", "nor",

    # phrases
    "no longer", "not anymore", "not any more",
    "not at all", "by no means", "in no way",

    # soft negation
    "hardly", "barely", "scarcely", "rarely", "seldom",

    # lack / absence
    "without", "lack", "lacking", "missing",

    # intention-related
    "don't want", "do not want", "not want",
    "don't feel like", "do not feel like",
    "don't wanna", "not interested", "no interest",
    "don't care", "can't be bothered",

    # personal forms
    "i don't", "i won't", "i wouldn't", "i never",
    "i am not", "i'm not", "i can't", "i cannot",
    "i haven't", "i hadn't"
]

KEYWORD_GROUPS = {
    RiskLevel.CRITICAL: {
        "self_harm": [
            # explicit intent
            "kill myself", "end my life", "take my life",
            "i want to die", "wanna die", "i wanna die",
            "i don't want to live", "i dont want to live",
            "i can't live anymore", "i cant live anymore",

            # passive death wish
            "wish i was dead", "better off dead",
            "rather be dead", "should be dead",

            # imminent signals
            "i'm going to end it", "i am going to end it",
            "this is my last day", "goodbye forever",
            "no reason to live",

            # vague but critical patterns
            "i'm done with life", "im done with life",
        ],

        "harm_others": [
            "kill someone", "murder", "going to kill",
            "hurt someone", "want to hurt someone",
            "i'll kill", "i will kill",
        ],
    },

    RiskLevel.HIGH: {
        "self_harm_ideation": [
            "hurt myself", "self harm", "self-harm",
            "cut myself", "cutting myself",

            # emotional escalation
            "i can't go on", "i cant go on",
            "can't take it anymore", "cant take it anymore",
            "too much to handle", "overwhelmed",

            # existential
            "no point in living", "life is pointless",
            "why am i alive", "what's the point of living",
        ],

        "substance_crisis": [
            "took too many pills", "overdosed",
            "took all my medication",
            "drank too much", "too much alcohol",
        ],
    },

    RiskLevel.MEDIUM: {
        "emotional_distress": [
            "hopeless", "worthless", "useless",
            "nothing matters",

            # collapse signals
            "falling apart", "breaking down",
            "losing control",

            # self-worth
            "i hate myself", "hate myself",
            "i'm a failure", "im a failure",

            # isolation
            "no one cares", "all alone",
            "completely alone", "no one understands",
        ],
    },

    RiskLevel.LOW: {
        "general_sadness": [
            "feeling down", "very sad",
            "depressed", "miserable",
            "unhappy", "struggling",

            # softer signals
            "not okay", "not doing well",
            "feeling off", "feeling low",
            "bad day", "rough day",
        ],
    },
}

SAFE_RESPONSES = {
    RiskLevel.CRITICAL: (
        "I'm really sorry you're going through this. "
        "I'm really concerned about your safety right now. "
        "You don’t have to go through this alone — reaching out to someone you trust or a local crisis hotline can really help. "
        "If you're able, please consider contacting your local emergency number or a suicide prevention hotline in your country. "
        "I'm here with you. Can you tell me what’s been happening?"
    ),

    RiskLevel.HIGH: (
        "That sounds really overwhelming, and I'm really glad you shared this with me. "
        "I'm here to listen and support you. "
        "Are you safe right now? "
        "If things feel like too much, reaching out to someone you trust could really help."
    ),

    RiskLevel.MEDIUM: (
        "It sounds like you're dealing with a lot right now. "
        "I'm here to listen — you don't have to go through this alone. "
        "Do you want to share more about what's been weighing on you?"
    ),

    RiskLevel.LOW: (
        "I'm here with you. If something’s been bothering you, feel free to share — I'm listening."
    ),

    RiskLevel.SAFE: None,
}


def _has_negation_before(text: str, keyword: str, window: int = 6) -> bool:
    """
    Kiểm tra xem có từ phủ định nào trong `window` từ trước keyword không.
    Ví dụ: "not want to die" → True (có "not" trước "die")
    """
    text_lower  = text.lower()
    kw_lower    = keyword.lower()
    kw_pos      = text_lower.find(kw_lower)
    if kw_pos == -1:
        return False

    # Lấy đoạn text trước keyword
    before_text = text_lower[:kw_pos]
    before_words = before_text.split()

    # Kiểm tra trong `window` từ gần nhất
    check_words = before_words[-window:] if len(before_words) >= window else before_words
    check_str   = " ".join(check_words)

    return any(neg in check_str for neg in NEGATION_WORDS)


class KeywordFilter:

    def __init__(self):
        self._lookup: dict[str, tuple[RiskLevel, str]] = {}
        for risk_level, categories in KEYWORD_GROUPS.items():
            for category, keywords in categories.items():
                for kw in keywords:
                    self._lookup[kw.lower()] = (risk_level, category)

    def check(self, text: str) -> FilterResult:
        text_lower = text.lower()
        text_clean = re.sub(r"[^\w\s]", " ", text_lower)

        matched       = []
        highest_level = RiskLevel.SAFE
        matched_cat   = None

        LEVEL_ORDER = [
            RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM,
            RiskLevel.HIGH, RiskLevel.CRITICAL,
        ]

        for keyword, (risk_level, category) in self._lookup.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_clean):
                # ── Negation check ────────────────────────────
                # Nếu có từ phủ định trước keyword → bỏ qua
                if _has_negation_before(text_clean, keyword, window=4):
                    continue

                matched.append(keyword)
                if LEVEL_ORDER.index(risk_level) > LEVEL_ORDER.index(highest_level):
                    highest_level = risk_level
                    matched_cat   = category

        is_flagged = highest_level != RiskLevel.SAFE

        return FilterResult(
            is_flagged=is_flagged,
            risk_level=highest_level,
            matched_keywords=matched,
            category=matched_cat,
            message=SAFE_RESPONSES.get(highest_level),
        )

    def add_keyword(self, keyword: str, risk_level: RiskLevel, category: str):
        self._lookup[keyword.lower()] = (risk_level, category)

    def remove_keyword(self, keyword: str):
        self._lookup.pop(keyword.lower(), None)


keyword_filter = KeywordFilter()


if __name__ == "__main__":
    tests = [
        ("I want to die",               True,  "CRITICAL"),
        ("I don't want to die",         False, "SAFE"),
        ("No no no, i still feel bad, not want to die", False, "SAFE"),
        ("I never want to kill myself", False, "SAFE"),
        ("I feel hopeless",             True,  "MEDIUM"),
        ("I hate myself",               True,  "MEDIUM"),
        ("The weather is nice",         False, "SAFE"),
        ("I want to hurt myself",       True,  "HIGH"),
        ("I don't want to hurt anyone", False, "SAFE"),
    ]

    print("="*60)
    for text, expect_flag, expect_level in tests:
        result = keyword_filter.check(text)
        status = "✅" if result.is_flagged == expect_flag else "❌"
        print(f"{status} [{result.risk_level}] {text}")
        if result.matched_keywords:
            print(f"   Matched: {result.matched_keywords}")
    print("="*60)