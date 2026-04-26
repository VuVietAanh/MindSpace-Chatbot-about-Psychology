# ============================================================
# hf_space/app.py — AI Microservice cho Hugging Face Space
# Chạy emotion analysis + embedding, được gọi từ main app
# ============================================================

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MindSpace AI Service")

# ── Load models once at startup ──────────────────────────────
print("⏳ Loading emotion model...")
from transformers import pipeline as hf_pipeline
emotion_pipe = hf_pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=-1,  # CPU
)
print("✅ Emotion model loaded")

print("⏳ Loading embedding model...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

PLUTCHIK = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "trust", "anticipation"]
MODEL_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


# ── Request / Response models ────────────────────────────────

class EmotionRequest(BaseModel):
    text:           str
    recent_history: list[str] | None = None

class EmotionResponse(BaseModel):
    scores:           dict[str, float]
    dominant_emotion: str
    raw_text:         str
    method:           str

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim:        int


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/emotion", response_model=EmotionResponse)
def analyze_emotion(req: EmotionRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # C1: Expand nếu text quá ngắn (< 4 từ)
    method = "direct"
    if len(text.split()) < 4 and req.recent_history:
        context = " ".join(req.recent_history[-2:])
        text    = f"{context} {text}"
        method  = "expanded"

    # Run emotion model
    results = emotion_pipe(text[:512])[0]

    # Map scores
    raw_scores = {r["label"].lower(): round(r["score"], 4) for r in results}

    # Build Plutchik 8 scores
    scores = {e: 0.0 for e in PLUTCHIK}
    for emotion in MODEL_EMOTIONS:
        if emotion in raw_scores and emotion in scores:
            scores[emotion] = raw_scores[emotion]

    # C3: Combine với history nếu có
    if req.recent_history and method == "direct":
        try:
            hist_text = " ".join(req.recent_history[-3:])
            hist_results = emotion_pipe(hist_text[:512])[0]
            hist_scores  = {r["label"].lower(): r["score"] for r in hist_results}
            alpha = 0.7  # Ưu tiên current input
            for e in PLUTCHIK:
                if e in hist_scores:
                    scores[e] = round(alpha * scores[e] + (1 - alpha) * hist_scores[e], 4)
            method = "combined"
        except Exception:
            pass

    # Normalize
    total = sum(scores.values())
    if total > 0:
        scores = {e: round(v / total, 4) for e, v in scores.items()}

    dominant = max(PLUTCHIK, key=lambda e: scores[e])

    return EmotionResponse(
        scores=scores,
        dominant_emotion=dominant,
        raw_text=req.text,
        method=method,
    )


@app.post("/embed", response_model=EmbedResponse)
def embed_texts(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Empty texts")

    embeddings = embed_model.encode(req.texts, normalize_embeddings=True)
    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dim=embeddings.shape[1],
    )


@app.post("/embed/single")
def embed_single(req: dict):
    text = req.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    vec = embed_model.encode([text], normalize_embeddings=True)[0]
    return {"embedding": vec.tolist(), "dim": len(vec)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
