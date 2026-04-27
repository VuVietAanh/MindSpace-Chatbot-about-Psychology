# ============================================================
# core/emotion_remote.py
# Gọi HF Space thay vì chạy transformers local
# Drop-in replacement cho EmotionAnalyzer
# ============================================================

import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()

PLUTCHIK_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "trust", "anticipation"]
NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]


@dataclass
class EmotionResult:
    scores:           dict[str, float]
    dominant_emotion: str
    is_negative:      bool
    confidence:       float
    raw_text:         str
    analyzed_text:    str
    method:           str


class RemoteEmotionAnalyzer:
    """
    Gọi HF Space /emotion endpoint.
    Interface giống hệt EmotionAnalyzer — pipeline.py không cần đổi gì.
    """

    def __init__(self, base_url: str):
        # Bỏ trailing slash nếu có
        self._base_url = base_url.rstrip("/")
        self._timeout  = float(os.getenv("AI_SERVICE_TIMEOUT", "30"))
        print(f"✅ RemoteEmotionAnalyzer → {self._base_url}")

    def analyze(self, text: str, recent_history: list[str] = None) -> EmotionResult:
        if not text or not text.strip():
            return self._empty_result(text)

        payload = {
            "text": text,
            "recent_history": recent_history or [],
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/emotion", json=payload)
                resp.raise_for_status()
                data = resp.json()

            scores           = data["scores"]
            dominant_emotion = data["dominant_emotion"]
            method           = data.get("method", "remote")

            # Đảm bảo đủ 8 Plutchik keys
            for e in PLUTCHIK_EMOTIONS:
                scores.setdefault(e, 0.0)

            confidence  = scores.get(dominant_emotion, 0.0)
            is_negative = dominant_emotion in NEGATIVE_EMOTIONS

            return EmotionResult(
                scores=scores,
                dominant_emotion=dominant_emotion,
                is_negative=is_negative,
                confidence=confidence,
                raw_text=text,
                analyzed_text=text,
                method=method,
            )

        except httpx.TimeoutException:
            print(f"⚠️  Emotion service timeout — fallback to neutral")
            return self._empty_result(text)
        except Exception as e:
            print(f"⚠️  Emotion service error: {e} — fallback to neutral")
            return self._empty_result(text)

    def _empty_result(self, text: str) -> EmotionResult:
        return EmotionResult(
            scores={e: 0.0 for e in PLUTCHIK_EMOTIONS},
            dominant_emotion="neutral",
            is_negative=False,
            confidence=0.0,
            raw_text=text or "",
            analyzed_text=text or "",
            method="fallback",
        )
