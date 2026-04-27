# ============================================================
# core/embedding.py
# Chuyển text thành vector để đưa vào FAISS
# Model: all-MiniLM-L6-v2 (nhẹ, nhanh, tốt cho tiếng Anh)
# ============================================================

import os
import pickle
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv

load_dotenv()
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "")


@dataclass
class EmbeddingResult:
    text:   str
    vector: np.ndarray
    dim:    int


class Embedder:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # ── sentence_transformers chỉ import khi dùng local model ──
        from sentence_transformers import SentenceTransformer

        print(f"⏳ Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._dim   = self._model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded! Dimension: {self._dim}")

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
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return EmbeddingResult(text=text, vector=vector, dim=self._dim)

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        return [
            EmbeddingResult(text=text, vector=vector, dim=self._dim)
            for text, vector in zip(texts, vectors)
        ]

    def embed_with_emotion(self, text: str, emotion_summary: str) -> EmbeddingResult:
        combined = f"Emotion context: {emotion_summary}. User said: {text}"
        return self.embed(combined)

    def save_embeddings(self, results: list[EmbeddingResult], save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"✅ Saved {len(results)} embeddings → {save_path}")

    def load_embeddings(self, load_path: str) -> list[EmbeddingResult]:
        with open(load_path, "rb") as f:
            results = pickle.load(f)
        print(f"✅ Loaded {len(results)} embeddings ← {load_path}")
        return results

    def to_matrix(self, results: list[EmbeddingResult]) -> np.ndarray:
        return np.vstack([r.vector for r in results]).astype(np.float32)


_embedder_instance = None

def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    global _embedder_instance
    if _embedder_instance is None:
        if AI_SERVICE_URL:
            print(f"🌐 Using remote embedder: {AI_SERVICE_URL}")
            from core.embedding_remote import RemoteEmbedder
            _embedder_instance = RemoteEmbedder(AI_SERVICE_URL)
        else:
            print("💻 Using local embedder")
            _embedder_instance = Embedder(model_name=model_name)
    return _embedder_instance