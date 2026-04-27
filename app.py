# ============================================================
# app.py — Main FastAPI app (Render)
# KHÔNG có torch/transformers — AI chạy trên HF Space riêng
# ============================================================

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="MindSpace Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Khởi tạo pipeline 1 lần khi start ───────────────────────
from pipeline import ChatbotPipeline

_pipeline = ChatbotPipeline()


# ── Request / Response models ────────────────────────────────

class SetupRequest(BaseModel):
    user_id: str | None = None
    name:    str | None = None

class ChatRequest(BaseModel):
    user_input:      str
    user_id:         str
    conversation_id: str

class EndRequest(BaseModel):
    conversation_id: str


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/setup")
def setup(req: SetupRequest):
    from sqlalchemy.orm import sessionmaker

    from db.crud import create_conversation, create_user, get_user
    from db.models import init_db

    engine  = _pipeline._engine
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        user, conv = _pipeline.setup_user(
            session=session,
            user_id=req.user_id,
            name=req.name,
        )
        greeting = _pipeline.get_greeting_if_returning(
            session=session,
            user_id=user.user_id,
            name=req.name,
        )
        return {
            "user_id":         user.user_id,
            "conversation_id": conv.conversation_id,
            "greeting":        greeting,
        }
    finally:
        session.close()


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.user_input or not req.user_input.strip():
        raise HTTPException(status_code=400, detail="Empty input")

    result = _pipeline.process(
        user_input=req.user_input,
        user_id=req.user_id,
        conversation_id=req.conversation_id,
    )
    return {
        "response":         result.response,
        "crisis_level":     result.crisis_level,
        "dominant_emotion": result.dominant_emotion,
        "was_flagged":      result.was_flagged,
        "intent":           result.intent,
        "high_emotion":     result.high_emotion,
    }


@app.post("/end")
def end_session(req: EndRequest):
    _pipeline.end_session(req.conversation_id)
    return {"status": "ended"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)))