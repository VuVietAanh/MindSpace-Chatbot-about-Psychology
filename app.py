# ============================================================
# app.py — Main FastAPI app (Render)
# Full API matching frontend routes
# ============================================================

import os
import re
import secrets
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

app = FastAPI(title="MindSpace Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pipeline (khởi tạo 1 lần) ───────────────────────────────
from pipeline import ChatbotPipeline

_pipeline = ChatbotPipeline()
_Session  = sessionmaker(bind=_pipeline._engine)

def get_db():
    db = _Session()
    try:
        yield db
    finally:
        db.close()

# ── Simple token store (in-memory) ──────────────────────────
# Production nên dùng Redis hoặc JWT, nhưng đây đủ cho demo
_tokens: dict[str, str] = {}  # token → user_id

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    user_id = _tokens.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id

def get_optional_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    return _tokens.get(token)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class LoginRequest(BaseModel):
    email:    str
    password: str

class RegisterRequest(BaseModel):
    name:          str
    email:         str
    password:      str
    age:           int  | None = None
    gender:        str  | None = None
    date_of_birth: str  | None = None

class ProfileUpdateRequest(BaseModel):
    name:          str  | None = None
    age:           int  | None = None
    gender:        str  | None = None
    date_of_birth: str  | None = None

class StartRequest(BaseModel):
    name:    str | None = None
    user_id: str | None = None

class ConfirmIdentityRequest(BaseModel):
    user_id:         str
    conversation_id: str
    is_same_person:  bool
    new_name:        str | None = None

class ChatRequest(BaseModel):
    user_id:         str
    conversation_id: str
    message:         str

class NewConvRequest(BaseModel):
    user_id: str

class RenameRequest(BaseModel):
    name: str

class CarryOverRequest(BaseModel):
    user_id:                str
    source_conversation_id: str

class EndRequest(BaseModel):
    conversation_id: str


# ============================================================
# AUTH ROUTES
# ============================================================

def _validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def _validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least 1 uppercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least 1 number"
    return True, ""

@app.post("/api/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    session = Session()
    try:
        # Validate email format
        if not _validate_email(req.email):
            raise HTTPException(status_code=400, detail="Invalid email format")

        # Validate password strength
        ok, msg = _validate_password(req.password)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)

        # Check email exists
        if get_user_by_email(session, req.email):
            raise HTTPException(status_code=400, detail="Email already registered")

        user = create_user(
            session, name=req.name, email=req.email, password=req.password,
            age=req.age, gender=req.gender, date_of_birth=req.date_of_birth,
        )
        token = create_token(user.user_id, user.role)
        return AuthResponse(token=token, user_id=user.user_id,
                            name=user.name, role=user.role)
    finally:
        session.close()


@app.post("/api/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    from db.crud import authenticate_user
    user = authenticate_user(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    token = secrets.token_hex(32)
    _tokens[token] = user.user_id
    return {"token": token, "user_id": user.user_id, "name": user.name,
            "email": user.email, "role": user.role}


@app.get("/api/auth/me")
def me(user_id: str = Depends(get_current_user), db: Session = Depends(get_db)):
    from db.crud import get_user
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user.user_id, "name": user.name, "email": user.email,
            "role": user.role, "age": user.age, "gender": user.gender,
            "date_of_birth": str(user.date_of_birth) if user.date_of_birth else None}


@app.put("/api/auth/profile")
def update_profile(req: ProfileUpdateRequest,
                   user_id: str = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    from datetime import date

    from db.crud import update_user_profile
    dob = None
    if req.date_of_birth:
        try:
            dob = date.fromisoformat(req.date_of_birth)
        except ValueError:
            pass
    update_user_profile(db, user_id, name=req.name, age=req.age,
                        gender=req.gender, date_of_birth=dob)
    return {"status": "updated"}


# ============================================================
# BACKGROUNDS
# ============================================================

@app.get("/api/backgrounds")
def backgrounds():
    folder = "static/backgrounds"
    if not os.path.exists(folder):
        return {"images": []}
    images = [f for f in os.listdir(folder)
              if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    return {"images": images}


@app.get("/api/chat-backgrounds")
def chat_backgrounds():
    folder = "static/chat_backgrounds"
    if not os.path.exists(folder):
        return {"images": []}
    images = [f for f in os.listdir(folder)
              if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    return {"images": images}


# ============================================================
# SESSION START
# ============================================================

@app.post("/api/start")
def start(req: StartRequest, db: Session = Depends(get_db)):
    from db.crud import get_recent_conversations, get_user, get_user_by_name

    stored_uid  = req.user_id
    stored_user = get_user(db, stored_uid) if stored_uid else None

    # Identity check — nếu có user_id cũ nhưng gửi tên khác
    identity_check    = False
    identity_question = None

    if stored_user and req.name and stored_user.name:
        if req.name.strip().lower() != stored_user.name.strip().lower():
            identity_check    = True
            identity_question = (
                f"Welcome back! Are you **{stored_user.name}**, "
                f"or someone else using this device?"
            )

    # Setup user + conversation
    user, conv = _pipeline.setup_user(db, user_id=stored_uid, name=req.name)

    # Returning prompt
    returning_prompt   = None
    prev_conversation_id = None
    if not identity_check:
        recent = get_recent_conversations(db, user.user_id, limit=2)
        prev_convs = [c for c in recent if c.conversation_id != conv.conversation_id]
        if prev_convs:
            prev     = prev_convs[0]
            prev_name = prev.name_conversation or "your last conversation"
            returning_prompt     = (
                f"Welcome back! Last time we talked about **{prev_name}**. "
                f"Would you like to continue from where we left off?"
            )
            prev_conversation_id = prev.conversation_id

    greeting = None
    if not identity_check and not returning_prompt:
        greeting = _pipeline.get_greeting_if_returning(
            session=db, user_id=user.user_id, name=req.name
        )

    return {
        "user_id":              user.user_id,
        "conversation_id":      conv.conversation_id,
        "greeting":             greeting,
        "identity_check":       identity_check,
        "identity_question":    identity_question,
        "returning_prompt":     returning_prompt,
        "prev_conversation_id": prev_conversation_id,
    }


@app.post("/api/confirm-identity")
def confirm_identity(req: ConfirmIdentityRequest, db: Session = Depends(get_db)):
    from db.crud import create_conversation, create_user, get_user

    if req.is_same_person:
        user = get_user(db, req.user_id)
        greeting = _pipeline.get_greeting_if_returning(
            session=db, user_id=user.user_id, name=user.name
        ) or f"Good to see you again{', ' + user.name if user.name else ''}! How are you feeling today?"
        return {
            "user_id":         user.user_id,
            "conversation_id": req.conversation_id,
            "greeting":        greeting,
            "is_new_user":     False,
        }
    else:
        new_user = create_user(db, name=req.new_name)
        new_conv = create_conversation(db, user_id=new_user.user_id)
        greeting = f"Hi{', ' + req.new_name if req.new_name else ''}! I'm here to listen. How are you feeling today?"
        return {
            "user_id":         new_user.user_id,
            "conversation_id": new_conv.conversation_id,
            "greeting":        greeting,
            "is_new_user":     True,
        }


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    from db.crud import update_conversation_emotion
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    result = _pipeline.process(
        user_input=req.message,
        user_id=req.user_id,
        conversation_id=req.conversation_id,
    )

    # Update per-conversation emotion
    emotion_analyzer = _pipeline._emotion_analyzer
    try:
        emo = emotion_analyzer.analyze(req.message)
        update_conversation_emotion(db, req.conversation_id, emo.scores)
    except Exception:
        pass

    return {
        "response":         result.response,
        "crisis_level":     result.crisis_level,
        "dominant_emotion": result.dominant_emotion,
        "was_flagged":      result.was_flagged,
        "intent":           result.intent,
        "high_emotion":     result.high_emotion,
    }


# ============================================================
# CONVERSATIONS
# ============================================================

@app.post("/api/conversation/new")
def new_conversation(req: NewConvRequest, db: Session = Depends(get_db)):
    from db.crud import create_conversation, get_recent_conversations

    conv    = create_conversation(db, user_id=req.user_id)
    recent  = get_recent_conversations(db, req.user_id, limit=3)
    prev    = [c for c in recent if c.conversation_id != conv.conversation_id]

    returning_prompt     = None
    prev_conversation_id = None
    if prev:
        prev_name            = prev[0].name_conversation or "your last conversation"
        returning_prompt     = (
            f"Welcome back! Last time we talked about **{prev_name}**. "
            f"Would you like to continue, or start fresh?"
        )
        prev_conversation_id = prev[0].conversation_id

    greeting = _pipeline.get_greeting_if_returning(
        session=db, user_id=req.user_id, name=None
    ) if not returning_prompt else None

    return {
        "conversation_id":      conv.conversation_id,
        "greeting":             greeting,
        "returning_prompt":     returning_prompt,
        "prev_conversation_id": prev_conversation_id,
    }


@app.get("/api/conversations/{user_id}")
def get_conversations(user_id: str, db: Session = Depends(get_db)):
    from db.models import Conversation
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.started_at.desc())
        .all()
    )
    return [
        {
            "conversation_id":   c.conversation_id,
            "name_conversation": c.name_conversation or "New conversation",
            "started_at":        c.started_at.strftime("%b %d, %Y") if c.started_at else None,
            "ended_at":          c.ended_at.strftime("%b %d, %Y") if c.ended_at else None,
        }
        for c in convs
    ]


@app.get("/api/conversation/{conversation_id}/messages")
def get_messages(conversation_id: str, db: Session = Depends(get_db)):
    from db.crud import get_recent_messages
    msgs = list(reversed(get_recent_messages(db, conversation_id, limit=100)))
    result = []
    for m in msgs:
        if m.user_input:
            result.append({"role": "user", "content": m.user_input,
                           "timestamp": m.timestamp.strftime("%H:%M") if m.timestamp else ""})
        if m.bot_response:
            result.append({"role": "bot", "content": m.bot_response,
                           "timestamp": m.timestamp.strftime("%H:%M") if m.timestamp else ""})
    return result


@app.put("/api/conversation/{conversation_id}/rename")
def rename_conversation(conversation_id: str, req: RenameRequest,
                        db: Session = Depends(get_db)):
    from db.models import Conversation
    conv = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.name_conversation = req.name[:80]
    db.commit()
    return {"conversation_id": conversation_id, "name": conv.name_conversation}


@app.delete("/api/conversation/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    from db.crud import delete_conversation as _delete
    ok = _delete(db, conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}


@app.post("/api/conversation/carry-over")
def carry_over(req: CarryOverRequest, db: Session = Depends(get_db)):
    from db.crud import carry_over_emotion_from_conversation
    carry_over_emotion_from_conversation(db, req.user_id, req.source_conversation_id)
    return {"status": "carried_over"}


@app.post("/api/end/{conversation_id}")
def end_session(conversation_id: str, db: Session = Depends(get_db)):
    _pipeline.end_session(conversation_id)
    return {"status": "ended"}


# ============================================================
# EMOTION
# ============================================================

@app.get("/api/emotion/conversation/{conversation_id}")
def emotion_conversation(conversation_id: str, db: Session = Depends(get_db)):
    from db.crud import get_conversation_emotion
    ce = get_conversation_emotion(db, conversation_id)
    if not ce:
        return {"scores": {}, "dominant_emotion": None}
    EMOTIONS = ["anger","disgust","fear","joy","sadness","surprise","trust","anticipation"]
    return {
        "scores":           {e: getattr(ce, e, 0.0) for e in EMOTIONS},
        "dominant_emotion": ce.dominant_emotion,
        "turn_count":       ce.turn_count,
    }


# ============================================================
# ADMIN ROUTES
# ============================================================

def require_admin(user_id: str = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    from db.crud import get_user
    user = get_user(db, user_id)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user_id


@app.get("/api/admin/stats")
def admin_stats(admin: str = Depends(require_admin), db: Session = Depends(get_db)):
    from db.crud import (get_active_user_count, get_conversation_count,
                         get_user_count)
    from db.models import Message
    crisis_count = db.query(Message).filter(Message.crisis_flagged == True).count()
    return {
        "total_users":        get_user_count(db),
        "active_users":       get_active_user_count(db),
        "total_conversations": get_conversation_count(db),
        "crisis_messages":    crisis_count,
    }


@app.get("/api/admin/users")
def admin_users(limit: int = 200, admin: str = Depends(require_admin),
                db: Session = Depends(get_db)):
    from db.crud import get_all_users
    from db.models import Conversation
    users = get_all_users(db, limit=limit)
    result = []
    for u in users:
        conv_count = db.query(Conversation).filter(Conversation.user_id == u.user_id).count()
        result.append({
            "user_id":           u.user_id,
            "name":              u.name,
            "email":             u.email,
            "age":               u.age,
            "gender":            u.gender,
            "date_of_birth":     str(u.date_of_birth) if u.date_of_birth else None,
            "is_active":         u.is_active,
            "role":              u.role,
            "conversation_count": conv_count,
            "created_at":        u.created_at.isoformat() if u.created_at else None,
            "last_active":       u.last_active.isoformat() if u.last_active else None,
        })
    return result


@app.patch("/api/admin/users/{user_id}/toggle-active")
def toggle_user(user_id: str, admin: str = Depends(require_admin),
                db: Session = Depends(get_db)):
    from db.crud import get_user
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = not user.is_active
    db.commit()
    return {"user_id": user_id, "is_active": user.is_active}


@app.get("/api/admin/users/{user_id}/detail")
def admin_user_detail(user_id: str, admin: str = Depends(require_admin),
                      db: Session = Depends(get_db)):
    from db.crud import get_conversation_emotion, get_emotion_state, get_user
    from db.models import Conversation, Message

    user  = get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    state   = get_emotion_state(db, user_id)
    EMOTIONS = ["anger","disgust","fear","joy","sadness","surprise","trust","anticipation"]

    convs = (db.query(Conversation).filter(Conversation.user_id == user_id)
             .order_by(Conversation.started_at.desc()).all())
    convs_data = []
    for c in convs:
        msg_count = db.query(Message).filter(Message.conversation_id == c.conversation_id).count()
        crisis_c  = db.query(Message).filter(Message.conversation_id == c.conversation_id,
                                              Message.crisis_flagged == True).count()
        ce = get_conversation_emotion(db, c.conversation_id)
        convs_data.append({
            "conversation_id":   c.conversation_id,
            "name_conversation": c.name_conversation,
            "message_count":     msg_count,
            "crisis_count":      crisis_c,
            "dominant_emotion":  ce.dominant_emotion if ce else None,
            "emotion_scores":    {e: getattr(ce, e, 0.0) for e in EMOTIONS} if ce else {},
            "started_at":        c.started_at.isoformat() if c.started_at else None,
            "ended_at":          c.ended_at.isoformat() if c.ended_at else None,
        })

    return {
        "user_id":      user.user_id,
        "name":         user.name,
        "email":        user.email,
        "age":          user.age,
        "gender":       user.gender,
        "date_of_birth": str(user.date_of_birth) if user.date_of_birth else None,
        "is_active":    user.is_active,
        "created_at":   user.created_at.isoformat() if user.created_at else None,
        "last_active":  user.last_active.isoformat() if user.last_active else None,
        "global_emotion": {
            "scores":       {e: getattr(state, e, 0.0) for e in EMOTIONS} if state else {},
            "dominant":     state.dominant_emotion if state else None,
            "turn_count":   state.turn_count if state else 0,
            "crisis_flag":  state.crisis_flag if state else False,
        },
        "conversations": convs_data,
    }


@app.get("/api/admin/conversations")
def admin_conversations(limit: int = 200, admin: str = Depends(require_admin),
                        db: Session = Depends(get_db)):
    from db.models import Conversation, ConversationEmotion, Message, User

    convs = (db.query(Conversation)
             .order_by(Conversation.started_at.desc())
             .limit(limit).all())
    result = []
    for c in convs:
        user      = db.query(User).filter(User.user_id == c.user_id).first()
        msg_count = db.query(Message).filter(Message.conversation_id == c.conversation_id).count()
        crisis_c  = db.query(Message).filter(Message.conversation_id == c.conversation_id,
                                              Message.crisis_flagged == True).count()
        ce = db.query(ConversationEmotion).filter(
            ConversationEmotion.conversation_id == c.conversation_id
        ).first()
        result.append({
            "conversation_id":   c.conversation_id,
            "name_conversation": c.name_conversation,
            "user_name":         user.name if user else None,
            "user_email":        user.email if user else None,
            "message_count":     msg_count,
            "crisis_count":      crisis_c,
            "dominant_emotion":  ce.dominant_emotion if ce else None,
            "started_at":        c.started_at.isoformat() if c.started_at else None,
            "ended_at":          c.ended_at.isoformat() if c.ended_at else None,
        })
    return result


# ============================================================
# HEALTH + STATIC FILES
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/admin")
def admin_page():
    f = "static/admin.html"
    if os.path.exists(f):
        return FileResponse(f)
    raise HTTPException(status_code=404, detail="Admin page not found")


@app.get("/")
def root():
    f = "static/index.html"
    if os.path.exists(f):
        return FileResponse(f)
    return {"status": "ok", "message": "MindSpace API running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8001)))