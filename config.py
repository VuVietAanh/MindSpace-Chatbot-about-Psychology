# ============================================================
# config.py
# Cấu hình trung tâm cho toàn bộ pipeline
# Đọc từ .env, có giá trị mặc định nếu không tìm thấy
# ============================================================

import os

from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------
# Database
# ------------------------------------------------------------
class DBConfig:
    URL: str = os.getenv("DATABASE_URL", "sqlite:///chatbot.db")


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
class ModelConfig:
    LANGUAGE: str = os.getenv("LANGUAGE", "en")  # "en" hoặc "vi"

    # Emotion Analysis
    EMOTION_MODEL = {
        "en": "j-hartmann/emotion-english-distilroberta-base",
        "vi": "vinai/phobert-base",
    }

    # Embedding
    EMBEDDING_MODEL = {
        "en": "all-MiniLM-L6-v2",
        "vi": "keepitreal/vietnamese-sbert",
    }

    @classmethod
    def get_emotion_model(cls) -> str:
        return cls.EMOTION_MODEL.get(cls.LANGUAGE, cls.EMOTION_MODEL["en"])

    @classmethod
    def get_embedding_model(cls) -> str:
        return cls.EMBEDDING_MODEL.get(cls.LANGUAGE, cls.EMBEDDING_MODEL["en"])


# ------------------------------------------------------------
# LLM (OpenAI)
# ------------------------------------------------------------
class LLMConfig:
    API_KEY:     str   = os.getenv("OPENAI_API_KEY", "")
    MODEL:       str   = os.getenv("LLM_MODEL", "gpt-4o-mini")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    MAX_TOKENS:  int   = int(os.getenv("LLM_MAX_TOKENS", 512))


# ------------------------------------------------------------
# Emotion & EMA
# ------------------------------------------------------------
class EmotionConfig:
    ALPHA: float = float(os.getenv("EMA_ALPHA", 0.3))  # EMA weight

    PLUTCHIK_EMOTIONS = [
        "anger", "disgust", "fear", "joy",
        "sadness", "surprise", "trust", "anticipation",
    ]

    NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]


# ------------------------------------------------------------
# Crisis Detection
# ------------------------------------------------------------
class CrisisConfig:
    # Số turn tiêu cực liên tiếp
    WATCH_CONSECUTIVE:    int = int(os.getenv("CRISIS_WATCH_TURNS",    2))
    ALERT_CONSECUTIVE:    int = int(os.getenv("CRISIS_ALERT_TURNS",    4))
    CRITICAL_CONSECUTIVE: int = int(os.getenv("CRISIS_CRITICAL_TURNS", 6))

    # Ngưỡng điểm cảm xúc
    HIGH_EMOTION_SCORE:      float = float(os.getenv("CRISIS_HIGH_SCORE",     0.55))
    COMBINED_NEGATIVE_SCORE: float = float(os.getenv("CRISIS_COMBINED_SCORE", 0.70))

    # Ngưỡng riêng cho từng cảm xúc nguy hiểm
    HIGH_RISK_THRESHOLDS: dict = {
        "sadness": 0.50,
        "fear":    0.55,
        "anger":   0.60,
    }


# ------------------------------------------------------------
# RAG & Retrieval
# ------------------------------------------------------------
class RAGConfig:
    # Chunking
    CHUNK_SIZE:    int = int(os.getenv("CHUNK_SIZE",    256))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 30))
    WINDOW_SIZE:   int = int(os.getenv("WINDOW_SIZE",   1))

    # Retrieval
    TOP_K:           int   = int(os.getenv("RETRIEVAL_TOP_K",         3))
    SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.3))

    # Paths
    KNOWLEDGE_BASE_DIR: str = os.getenv("KB_DIR",     "data/knowledge_base")
    FAISS_INDEX_PATH:   str = os.getenv("FAISS_PATH", "data/faiss.index")
    CHUNKS_PATH:        str = os.getenv("CHUNKS_PATH","data/chunks.json")


# ------------------------------------------------------------
# Keyword Filter
# ------------------------------------------------------------
class FilterConfig:
    # Mức nào trở lên thì skip RAG, trả safe response ngay
    SKIP_RAG_LEVELS = ["high", "critical"]


# ------------------------------------------------------------
# Context
# ------------------------------------------------------------
class ContextConfig:
    RECENT_MSG_LIMIT: int = int(os.getenv("RECENT_MSG_LIMIT", 6))


# ------------------------------------------------------------
# Export gọn — dùng trong các file khác
# ------------------------------------------------------------
class Config:
    db      = DBConfig
    model   = ModelConfig
    llm     = LLMConfig
    emotion = EmotionConfig
    crisis  = CrisisConfig
    rag     = RAGConfig
    filter  = FilterConfig
    context = ContextConfig

    @classmethod
    def validate(cls):
        """Kiểm tra config bắt buộc trước khi chạy pipeline"""
        errors = []

        if not cls.llm.API_KEY:
            errors.append("❌ OPENAI_API_KEY chưa được set trong .env")

        if not os.path.exists(cls.rag.KNOWLEDGE_BASE_DIR):
            errors.append(f"⚠️  Knowledge base dir không tồn tại: {cls.rag.KNOWLEDGE_BASE_DIR}")

        if errors:
            print("\n" + "="*60)
            print("CONFIG ERRORS:")
            for e in errors:
                print(f"  {e}")
            print("="*60 + "\n")
            return False

        return True


# ------------------------------------------------------------
# Test thử
# ------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("📋 Current Config:")
    print(f"  Language        : {Config.model.LANGUAGE}")
    print(f"  Emotion model   : {Config.model.get_emotion_model()}")
    print(f"  Embedding model : {Config.model.get_embedding_model()}")
    print(f"  LLM model       : {Config.llm.MODEL}")
    print(f"  LLM temperature : {Config.llm.TEMPERATURE}")
    print(f"  EMA alpha       : {Config.emotion.ALPHA}")
    print(f"  DB URL          : {Config.db.URL}")
    print(f"  Chunk size      : {Config.rag.CHUNK_SIZE}")
    print(f"  RAG top_k       : {Config.rag.TOP_K}")
    print(f"  Crisis turns    : watch={Config.crisis.WATCH_CONSECUTIVE} | alert={Config.crisis.ALERT_CONSECUTIVE} | critical={Config.crisis.CRITICAL_CONSECUTIVE}")
    print("="*60)

    is_valid = Config.validate()
    print(f"\n✅ Config valid: {is_valid}")