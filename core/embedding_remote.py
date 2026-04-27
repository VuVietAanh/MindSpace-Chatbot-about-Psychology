# ============================================================
# core/embedding_remote.py
# Gọi HF Space thay vì chạy sentence-transformers local
# Drop-in replacement cho Embedder
# ============================================================

import os
from dataclasses import dataclass

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


@dataclass
class EmbeddingResult:
    text:   str
    vector: np.ndarray
    dim:    int


class RemoteEmbedder:
    """
    Gọi HF Space /embed endpoint.
    Interface giống hệt Embedder — pipeline.py không cần đổi gì.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._timeout  = float(os.getenv("AI_SERVICE_TIMEOUT", "30"))
        self._dim      = EMBEDDING_DIM
        print(f"✅ RemoteEmbedder → {self._base_url}")

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> EmbeddingResult:
        if not text or not text.strip():
            return EmbeddingResult(
                text=text,
                vector=np.zeros(self._dim, dtype=np.float32),
                dim=self._dim,
            )

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/embed/single",
                    json={"text": text},
                )
                resp.raise_for_status()
                data = resp.json()

            vector = np.array(data["embedding"], dtype=np.float32)
            return EmbeddingResult(text=text, vector=vector, dim=len(vector))

        except Exception as e:
            print(f"⚠️  Embedding service error: {e} — fallback to zeros")
            return EmbeddingResult(
                text=text,
                vector=np.zeros(self._dim, dtype=np.float32),
                dim=self._dim,
            )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/embed",
                    json={"texts": texts},
                )
                resp.raise_for_status()
                data = resp.json()

            return [
                EmbeddingResult(
                    text=text,
                    vector=np.array(vec, dtype=np.float32),
                    dim=len(vec),
                )
                for text, vec in zip(texts, data["embeddings"])
            ]

        except Exception as e:
            print(f"⚠️  Batch embedding error: {e} — fallback to zeros")
            return [
                EmbeddingResult(
                    text=t,
                    vector=np.zeros(self._dim, dtype=np.float32),
                    dim=self._dim,
                )
                for t in texts
            ]

    def embed_with_emotion(self, text: str, emotion_summary: str) -> EmbeddingResult:
        """Giống Embedder.embed_with_emotion — ghép emotion context trước khi embed"""
        combined = f"Emotion context: {emotion_summary}. User said: {text}"
        return self.embed(combined)

    def to_matrix(self, results: list[EmbeddingResult]) -> np.ndarray:
        return np.vstack([r.vector for r in results]).astype(np.float32)
