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
from sentence_transformers import SentenceTransformer

load_dotenv()
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "")

# ------------------------------------------------------------
# Kết quả embedding
# ------------------------------------------------------------
@dataclass
class EmbeddingResult:
    text:   str
    vector: np.ndarray    # Shape: (384,) với MiniLM
    dim:    int           # Số chiều vector


# ------------------------------------------------------------
# Embedder
# ------------------------------------------------------------
class Embedder:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"⏳ Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._dim   = self._model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded! Dimension: {self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    # ----------------------------------------------------------
    # Embed 1 câu
    # ----------------------------------------------------------
    def embed(self, text: str) -> EmbeddingResult:
        """Embed 1 đoạn text → vector"""
        if not text or not text.strip():
            return EmbeddingResult(
                text=text,
                vector=np.zeros(self._dim, dtype=np.float32),
                dim=self._dim,
            )

        vector = self._model.encode(
            text,
            normalize_embeddings=True,   # Normalize → cosine similarity = dot product
            show_progress_bar=False,
        )
        return EmbeddingResult(text=text, vector=vector, dim=self._dim)

    # ----------------------------------------------------------
    # Embed nhiều câu cùng lúc (hiệu quả hơn)
    # ----------------------------------------------------------
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed nhiều đoạn text cùng lúc — dùng khi index knowledge base"""
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

    # ----------------------------------------------------------
    # Embed query có kết hợp emotion context
    # ----------------------------------------------------------
    def embed_with_emotion(
        self,
        text: str,
        emotion_summary: str,
    ) -> EmbeddingResult:
        """
        Kết hợp text + emotion context trước khi embed
        Giúp RAG retrieval tìm được tài liệu phù hợp hơn với trạng thái cảm xúc

        Ví dụ:
        text            = "I don't know what to do"
        emotion_summary = "dominant: sadness (55%), fear (15%)"
        → combined      = "Emotion context: sadness, fear. User said: I don't know what to do"
        """
        combined = f"Emotion context: {emotion_summary}. User said: {text}"
        return self.embed(combined)

    # ----------------------------------------------------------
    # Lưu vectors ra file (dùng khi build knowledge base)
    # ----------------------------------------------------------
    def save_embeddings(
        self,
        results: list[EmbeddingResult],
        save_path: str,
    ):
        """Lưu danh sách EmbeddingResult ra file .pkl"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"✅ Saved {len(results)} embeddings → {save_path}")

    def load_embeddings(self, load_path: str) -> list[EmbeddingResult]:
        """Load embeddings đã lưu từ file"""
        with open(load_path, "rb") as f:
            results = pickle.load(f)
        print(f"✅ Loaded {len(results)} embeddings ← {load_path}")
        return results

    # ----------------------------------------------------------
    # Helper: lấy matrix numpy từ list results (để đưa vào FAISS)
    # ----------------------------------------------------------
    def to_matrix(self, results: list[EmbeddingResult]) -> np.ndarray:
        """
        Chuyển list EmbeddingResult → numpy matrix shape (N, dim)
        Dùng khi build FAISS index
        """
        return np.vstack([r.vector for r in results]).astype(np.float32)


# ------------------------------------------------------------
# Singleton
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Test thử
# ------------------------------------------------------------
if __name__ == "__main__":
    embedder = get_embedder()

    # Test embed đơn
    texts = [
        "I feel really sad and hopeless",
        "I'm excited about the future",
        "I don't know how to cope with anxiety",
        "Everything feels overwhelming right now",
    ]

    print("\n" + "="*60)
    print("📌 Single embed test:")
    for text in texts:
        result = embedder.embed(text)
        print(f"\n  Text   : {text}")
        print(f"  Shape  : {result.vector.shape}")
        print(f"  Norm   : {np.linalg.norm(result.vector):.4f}")  # Phải ≈ 1.0

    # Test embed với emotion context
    print("\n" + "="*60)
    print("📌 Embed with emotion context:")
    result = embedder.embed_with_emotion(
        text="I don't know what to do",
        emotion_summary="sadness (55%), fear (15%)",
    )
    print(f"  Combined text embedded successfully")
    print(f"  Shape: {result.vector.shape}")

    # Test similarity giữa 2 vector
    print("\n" + "="*60)
    print("📌 Similarity test:")
    r1 = embedder.embed("I feel hopeless and sad")
    r2 = embedder.embed("I am depressed and have no hope")
    r3 = embedder.embed("The weather is nice today")

    sim_12 = np.dot(r1.vector, r2.vector)
    sim_13 = np.dot(r1.vector, r3.vector)
    print(f"  'hopeless & sad' vs 'depressed & no hope' : {sim_12:.4f}  (should be HIGH)")
    print(f"  'hopeless & sad' vs 'weather is nice'     : {sim_13:.4f}  (should be LOW)")

    print("\n✅ Embedding test done!")