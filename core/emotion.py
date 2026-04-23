# ============================================================
# core/emotion.py
# Phân tích cảm xúc từ input của user
# FIX: Không suy ra trust/anticipation từ joy → tránh joy bị sai
# C1: Expand input ngắn/mơ hồ | C3: Ghép history
# ============================================================

import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline

load_dotenv()

MODEL_TO_PLUTCHIK = {
    "anger":   "anger",
    "disgust": "disgust",
    "fear":    "fear",
    "joy":     "joy",
    "sadness": "sadness",
    "surprise":"surprise",
    "neutral": None,
}

PLUTCHIK_EMOTIONS    = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "trust", "anticipation"]
NEGATIVE_EMOTIONS    = ["anger", "disgust", "fear", "sadness"]
MODEL_EMOTIONS       = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
SHORT_INPUT_THRESHOLD = 8


@dataclass
class EmotionResult:
    scores:           dict[str, float]
    dominant_emotion: str
    is_negative:      bool
    confidence:       float
    raw_text:         str
    analyzed_text:    str
    method:           str   # "direct" | "expanded" | "with_history"


class EmotionAnalyzer:

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        print(f"⏳ Loading emotion model: {model_name}")
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            task="text-classification",
            model=model_name,
            top_k=None,
            device=device,
            truncation=True,
            max_length=512,
        )
        self._llm = OpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
        self._llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        print("✅ Emotion model loaded!")

    def analyze(self, text: str, recent_history: list[str] = None) -> EmotionResult:
        if not text or not text.strip():
            return self._empty_result(text)

        analyzed_text = text
        method        = "direct"

        # C3: Ghép history nếu có
        if recent_history and len(recent_history) > 0:
            analyzed_text = self._combine_with_history(text, recent_history)
            method        = "with_history"
        # C1: Expand nếu input ngắn/mơ hồ
        elif self._is_short_or_vague(text):
            analyzed_text = self._expand_input(text)
            method        = "expanded"

        return self._run_model(raw_text=text, analyzed_text=analyzed_text, method=method)

    def _combine_with_history(self, text: str, recent_history: list[str], max_history: int = 3) -> str:
        recent   = recent_history[-max_history:]
        combined = ". ".join(recent) + ". " + text
        return combined.strip()

    def _is_short_or_vague(self, text: str) -> bool:
        return len(text.split()) <= SHORT_INPUT_THRESHOLD

    def _expand_input(self, text: str) -> str:
        try:
            response = self._llm.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Expand short emotional statements into 1-2 fuller sentences for emotion analysis. "
                            "CRITICAL: Preserve the EXACT sentiment — never add positive spin to negative statements. "
                            "Sad stays sad. Angry stays angry. Output ONLY the expanded sentence."
                        )
                    },
                    {"role": "user", "content": f'Expand: "{text}"'}
                ],
                temperature=0.2,
                max_tokens=80,
            )
            expanded = response.choices[0].message.content.strip()
            return expanded if expanded and len(expanded) <= 300 else text
        except Exception:
            return text

    def _run_model(self, raw_text: str, analyzed_text: str, method: str) -> EmotionResult:
        raw_output = self._pipe(analyzed_text)[0]

        # Khởi tạo tất cả về 0.0
        scores = {emotion: 0.0 for emotion in PLUTCHIK_EMOTIONS}

        for item in raw_output:
            label = item["label"].lower()
            score = round(item["score"], 4)
            if label == "neutral":
                continue
            mapped = MODEL_TO_PLUTCHIK.get(label)
            if mapped:
                scores[mapped] = score

        # ── FIX JOY SCORE ──────────────────────────────────────
        # KHÔNG suy ra trust/anticipation từ joy/fear
        # trust & anticipation giữ nguyên 0.0
        # → Joy chỉ tăng khi model thực sự detect joy
        # ────────────────────────────────────────────────────────

        # Normalize chỉ trên 6 emotions model trả về thật
        model_total = sum(scores[e] for e in MODEL_EMOTIONS)
        if model_total > 0:
            for e in MODEL_EMOTIONS:
                scores[e] = round(scores[e] / model_total, 4)

        # trust & anticipation = 0 (không có model support, không suy ra)
        scores["trust"]        = 0.0
        scores["anticipation"] = 0.0

        # Dominant chỉ xét 6 emotions model thật
        model_scores = {e: scores[e] for e in MODEL_EMOTIONS}
        dominant     = max(model_scores, key=model_scores.get) if any(v > 0 for v in model_scores.values()) else "neutral"
        confidence   = scores.get(dominant, 0.0)
        is_negative  = dominant in NEGATIVE_EMOTIONS

        return EmotionResult(
            scores=scores,
            dominant_emotion=dominant,
            is_negative=is_negative,
            confidence=confidence,
            raw_text=raw_text,
            analyzed_text=analyzed_text,
            method=method,
        )

    def _empty_result(self, text: str) -> EmotionResult:
        return EmotionResult(
            scores={e: 0.0 for e in PLUTCHIK_EMOTIONS},
            dominant_emotion="neutral",
            is_negative=False,
            confidence=0.0,
            raw_text=text or "",
            analyzed_text=text or "",
            method="direct",
        )


_analyzer_instance = None

def get_emotion_analyzer() -> EmotionAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = EmotionAnalyzer()
    return _analyzer_instance