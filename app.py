# ============================================================
# app.py — FastAPI with Auth + Admin
# ============================================================

import os
import secrets
from datetime import date, datetime, timedelta, timezone

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jose import jwt as jose_jwt
from jose.exceptions import ExpiredSignatureError, JWTError
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import sessionmaker

from db.crud import (authenticate_user, carry_over_emotion_from_conversation,
                     create_conversation, create_user, delete_conversation,
                     get_active_user_count, get_all_conversations_admin,
                     get_all_users, get_conversation_count,
                     get_conversation_emotion, get_emotion_state,
                     get_recent_conversations, get_recent_messages, get_user,
                     get_user_by_email, get_user_by_name, get_user_count,
                     reset_emotion_state, update_conversation_emotion,
                     update_user_name, update_user_profile)
from db.models import Conversation, ConversationEmotion, Message, User
from pipeline import ChatbotPipeline

load_dotenv()

app = FastAPI(title="MindSpace API")
pipeline = ChatbotPipeline()
Session  = sessionmaker(bind=pipeline._engine)

JWT_SECRET  = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_EXPIRE  = int(os.getenv("JWT_EXPIRE_HOURS", 24))

SETUP_BG_DIR = os.path.join("static", "backgrounds")
CHAT_BG_DIR  = os.path.join("static", "chat_backgrounds")
os.makedirs(SETUP_BG_DIR, exist_ok=True)
os.makedirs(CHAT_BG_DIR,  exist_ok=True)
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

PLUTCHIK_EMOTIONS = [
    "anger", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "anticipation"
]
HIGH_EMOTION_THRESH = 0.75
POSITIVE_EMOTIONS   = ["joy", "trust", "anticipation", "surprise"]
EMOTION_LABEL = {
    "anger": "anger", "disgust": "discomfort", "fear": "anxiety",
    "joy": "joy", "sadness": "sadness", "surprise": "surprise",
    "trust": "warmth", "anticipation": "anticipation",
}


# ============================================================
# JWT helpers
# ============================================================

def create_token(user_id: str, role: str) -> str:
    payload = {
        "user_id": user_id,
        "role":    role,
        "exp":     datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE),
    }
    return jose_jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jose_jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return decode_token(authorization.split(" ")[1])


def require_admin(current: dict = Depends(get_current_user)) -> dict:
    if current.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return current


# ============================================================
# Request / Response models
# ============================================================

class RegisterRequest(BaseModel):
    name:          str
    email:         str
    password:      str
    age:           int  | None = None
    gender:        str  | None = None
    date_of_birth: date | None = None

class LoginRequest(BaseModel):
    email:    str
    password: str

class AuthResponse(BaseModel):
    token:    str
    user_id:  str
    name:     str | None
    role:     str

class UpdateProfileRequest(BaseModel):
    name:          str  | None = None
    age:           int  | None = None
    gender:        str  | None = None
    date_of_birth: date | None = None

class StartRequest(BaseModel):
    name:    str | None = None
    user_id: str | None = None

class StartResponse(BaseModel):
    user_id:              str
    conversation_id:      str
    greeting:             str
    returning_prompt:     str | None = None
    prev_conversation_id: str | None = None
    identity_check:       bool = False
    identity_question:    str | None = None
    known_name:           str | None = None

class ConfirmIdentityRequest(BaseModel):
    user_id:         str
    conversation_id: str
    is_same_person:  bool
    new_name:        str | None = None

class ConfirmIdentityResponse(BaseModel):
    user_id:         str
    conversation_id: str
    greeting:        str
    is_new_user:     bool = False

class NewConvRequest(BaseModel):
    user_id: str

class NewConvResponse(BaseModel):
    conversation_id:      str
    greeting:             str
    returning_prompt:     str | None = None
    prev_conversation_id: str | None = None

class RenameConvRequest(BaseModel):
    name: str

class CarryOverRequest(BaseModel):
    user_id:                str
    source_conversation_id: str

class ChatRequest(BaseModel):
    user_id:         str
    conversation_id: str
    message:         str

class ChatResponse(BaseModel):
    response:         str
    dominant_emotion: str
    crisis_level:     str
    emotion_method:   str
    intent:           str = "neutral"
    high_emotion:     str | None = None

class EmotionResponse(BaseModel):
    dominant_emotion:           str
    turn_count:                 int
    consecutive_negative_turns: int
    crisis_flag:                bool
    scores:                     dict[str, float]

class ConversationItem(BaseModel):
    conversation_id:   str
    name_conversation: str
    started_at:        str
    ended_at:          str | None
    is_active:         bool

class MessageItem(BaseModel):
    role:      str
    content:   str
    timestamp: str


# ============================================================
# Helpers
# ============================================================

def _get_conv_emotion_scores(session, conversation_id: str) -> dict[str, float] | None:
    ce = get_conversation_emotion(session, conversation_id)
    if not ce or ce.turn_count == 0:
        return None
    return {e: round(getattr(ce, e, 0.0), 4) for e in PLUTCHIK_EMOTIONS}


def _get_returning_info(session, user_id: str, name: str | None):
    now     = datetime.now(timezone.utc)
    cutoff3 = now - timedelta(days=3)
    cutoff1 = now - timedelta(days=1)

    recent_convs = (
        session.query(Conversation)
        .filter(Conversation.user_id == user_id,
                Conversation.started_at >= cutoff3,
                Conversation.started_at <= cutoff1)
        .order_by(Conversation.started_at.desc())
        .all()
    )
    if not recent_convs:
        return None, None

    best_conv = None; best_scores = None; best_score = 0.0
    for conv in recent_convs:
        scores = _get_conv_emotion_scores(session, conv.conversation_id)
        if not scores:
            continue
        top = max(scores.values())
        if top > best_score and top >= HIGH_EMOTION_THRESH:
            best_score = top; best_scores = scores; best_conv = conv

    if not best_conv or not best_scores:
        return None, None

    model_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    top_emotions   = sorted(
        [(e, best_scores[e]) for e in model_emotions if best_scores[e] >= HIGH_EMOTION_THRESH],
        key=lambda x: -x[1]
    )[:2]
    if not top_emotions:
        return None, None

    conv_date = best_conv.started_at
    if conv_date.tzinfo is None:
        conv_date = conv_date.replace(tzinfo=timezone.utc)

    date_str      = conv_date.strftime("%B %d")
    name_part     = f" {name}" if name else ""
    emotion_names = [EMOTION_LABEL.get(e, e) for e, _ in top_emotions]
    emotion_str   = (f"**{emotion_names[0]}**" if len(emotion_names) == 1
                     else f"**{emotion_names[0]}** and **{emotion_names[1]}**")

    is_pos = top_emotions[0][0] in POSITIVE_EMOTIONS
    if is_pos:
        mood_ctx = f"your {emotion_str} was running really high"
        followup = "How are things going now — still in that good space?"
    else:
        mood_ctx = f"your {emotion_str} seemed quite elevated"
        followup = "How are you feeling about that now — has anything shifted?"

    prompt = (
        f"Hey{name_part}! On {date_str}, {mood_ctx}. "
        f"{followup} "
        f"Would you like to continue with that emotional thread, or start fresh today?"
    )
    return prompt, best_conv.conversation_id


def _check_identity(session, stored_uid, input_name):
    if not stored_uid or not input_name:
        return False, None, None
    user = get_user(session, stored_uid)
    if not user or not user.name:
        return False, None, None
    if user.name.strip().lower() == input_name.strip().lower():
        return False, user.name, None
    return True, user.name, f"Just to check — are you {user.name}? I noticed a different name."


# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.post("/api/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    session = Session()
    try:
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


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    session = Session()
    try:
        user = authenticate_user(session, req.email, req.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = create_token(user.user_id, user.role)
        return AuthResponse(token=token, user_id=user.user_id,
                            name=user.name, role=user.role)
    finally:
        session.close()


@app.get("/api/auth/me")
def get_me(current: dict = Depends(get_current_user)):
    session = Session()
    try:
        user = get_user(session, current["user_id"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "user_id":       user.user_id,
            "name":          user.name,
            "email":         user.email,
            "role":          user.role,
            "age":           user.age,
            "gender":        user.gender,
            "date_of_birth": str(user.date_of_birth) if user.date_of_birth else None,
            "created_at":    user.created_at.isoformat() if user.created_at else None,
            "last_active":   user.last_active.isoformat() if user.last_active else None,
        }
    finally:
        session.close()


@app.put("/api/auth/profile")
def update_profile(req: UpdateProfileRequest,
                   current: dict = Depends(get_current_user)):
    session = Session()
    try:
        update_user_profile(
            session, current["user_id"],
            name=req.name, age=req.age,
            gender=req.gender, date_of_birth=req.date_of_birth,
        )
        return {"status": "updated"}
    finally:
        session.close()


# ============================================================
# BACKGROUNDS
# ============================================================

@app.get("/api/backgrounds")
def list_backgrounds():
    try:
        images = [f for f in os.listdir(SETUP_BG_DIR)
                  if os.path.splitext(f)[1].lower() in ALLOWED_IMG_EXT]
    except FileNotFoundError:
        images = []
    return {"images": images}


@app.get("/api/chat-backgrounds")
def list_chat_backgrounds():
    try:
        images = [f for f in os.listdir(CHAT_BG_DIR)
                  if os.path.splitext(f)[1].lower() in ALLOWED_IMG_EXT]
    except FileNotFoundError:
        images = []
    return {"images": images}


# ============================================================
# CHAT SESSION
# ============================================================

@app.post("/api/start", response_model=StartResponse)
def start_session(req: StartRequest):
    session = Session()
    try:
        input_name = req.name.strip() if req.name else None
        needs_check, known_name, identity_q = _check_identity(session, req.user_id, input_name)

        if needs_check:
            user = get_user(session, req.user_id)
            conv = create_conversation(session, user_id=user.user_id)
            return StartResponse(
                user_id=user.user_id, conversation_id=conv.conversation_id,
                greeting="", identity_check=True,
                identity_question=identity_q, known_name=known_name,
            )

        if req.user_id:
            user = get_user(session, req.user_id)
            if not user:
                user = create_user(session, name=input_name)
            elif input_name and not user.name:
                update_user_name(session, user.user_id, input_name)
                user = get_user(session, req.user_id)
        else:
            user = create_user(session, name=input_name)

        conv            = create_conversation(session, user_id=user.user_id)
        user_id         = user.user_id
        conversation_id = conv.conversation_id

        returning_prompt, prev_conv_id = _get_returning_info(session, user_id, user.name)

        greeting = pipeline.get_greeting_if_returning(
            session=session, user_id=user_id, name=user.name
        )
        if not greeting:
            n = user.name
            greeting = (f"Hello {n}! I'm here to listen. How are you feeling today?"
                        if n else "Hello! I'm here to listen. How are you feeling today?")

        return StartResponse(
            user_id=user_id, conversation_id=conversation_id,
            greeting=greeting, returning_prompt=returning_prompt,
            prev_conversation_id=prev_conv_id,
        )
    finally:
        session.close()


@app.post("/api/confirm-identity", response_model=ConfirmIdentityResponse)
def confirm_identity(req: ConfirmIdentityRequest):
    session = Session()
    try:
        if req.is_same_person:
            user = get_user(session, req.user_id)
            if req.new_name and user:
                update_user_name(session, user.user_id, req.new_name)
                user = get_user(session, req.user_id)
            greeting = pipeline.get_greeting_if_returning(
                session=session, user_id=user.user_id, name=user.name
            ) or f"Welcome back{', ' + user.name if user.name else ''}! How are you today?"
            return ConfirmIdentityResponse(
                user_id=user.user_id, conversation_id=req.conversation_id,
                greeting=greeting, is_new_user=False,
            )
        else:
            new_user = create_user(session, name=req.new_name)
            new_conv = create_conversation(session, user_id=new_user.user_id)
            delete_conversation(session, req.conversation_id)
            greeting = (f"Hello {req.new_name}! How are you feeling today?"
                        if req.new_name else "Hello! How are you feeling today?")
            return ConfirmIdentityResponse(
                user_id=new_user.user_id, conversation_id=new_conv.conversation_id,
                greeting=greeting, is_new_user=True,
            )
    finally:
        session.close()


@app.post("/api/conversation/new", response_model=NewConvResponse)
def new_conversation(req: NewConvRequest):
    session = Session()
    try:
        user = get_user(session, req.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        conv                         = create_conversation(session, user_id=req.user_id)
        returning_prompt, prev_conv_id = _get_returning_info(session, req.user_id, user.name)
        greeting = pipeline.get_greeting_if_returning(
            session=session, user_id=req.user_id, name=user.name
        ) or f"Hello again{', ' + user.name if user.name else ''}! What's on your mind?"

        return NewConvResponse(
            conversation_id=conv.conversation_id, greeting=greeting,
            returning_prompt=returning_prompt, prev_conversation_id=prev_conv_id,
        )
    finally:
        session.close()


@app.put("/api/conversation/{conversation_id}/rename")
def rename_conversation(conversation_id: str, req: RenameConvRequest):
    session = Session()
    try:
        conv = session.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Not found")
        name = req.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        conv.name_conversation = name[:60]
        session.commit()
        return {"status": "renamed", "name": conv.name_conversation}
    finally:
        session.close()


@app.post("/api/conversation/carry-over")
def carry_over_emotion(req: CarryOverRequest):
    session = Session()
    try:
        success = carry_over_emotion_from_conversation(
            session=session,
            user_id=req.user_id,
            source_conversation_id=req.source_conversation_id,
        )
        return {"success": success}
    finally:
        session.close()


@app.get("/api/conversations/{user_id}", response_model=list[ConversationItem])
def get_conversations(user_id: str):
    session = Session()
    try:
        convs = (session.query(Conversation)
                 .filter(Conversation.user_id == user_id)
                 .order_by(Conversation.started_at.desc()).all())
        return [
            ConversationItem(
                conversation_id=c.conversation_id,
                name_conversation=c.name_conversation or "New conversation",
                started_at=c.started_at.strftime("%b %d, %Y") if c.started_at else "",
                ended_at=c.ended_at.strftime("%b %d, %Y %H:%M") if c.ended_at else None,
                is_active=c.ended_at is None,
            )
            for c in convs
        ]
    finally:
        session.close()


@app.delete("/api/conversation/{conversation_id}")
def delete_conv(conversation_id: str):
    session = Session()
    try:
        if not delete_conversation(session, conversation_id):
            raise HTTPException(status_code=404, detail="Not found")
        return {"status": "deleted"}
    finally:
        session.close()


@app.get("/api/conversation/{conversation_id}/messages", response_model=list[MessageItem])
def get_messages(conversation_id: str):
    session = Session()
    try:
        msgs = (session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp.asc()).all())
        result = []
        for msg in msgs:
            if msg.user_input:
                result.append(MessageItem(role="user", content=msg.user_input,
                    timestamp=msg.timestamp.strftime("%H:%M") if msg.timestamp else ""))
            if msg.bot_response:
                result.append(MessageItem(role="bot", content=msg.bot_response,
                    timestamp=msg.timestamp.strftime("%H:%M") if msg.timestamp else ""))
        return result
    finally:
        session.close()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    session = Session()
    try:
        result = pipeline.process(
            user_input=req.message,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
        )

        # Update per-conversation emotion
        emotion_state = get_emotion_state(session, req.user_id)
        if emotion_state:
            scores = {e: getattr(emotion_state, e, 0.0) for e in PLUTCHIK_EMOTIONS}
            update_conversation_emotion(session, req.conversation_id, scores)

        return ChatResponse(
            response=result.response,
            dominant_emotion=result.dominant_emotion,
            crisis_level=result.crisis_level,
            emotion_method=result.emotion_method,
            intent=result.intent,
            high_emotion=result.high_emotion,
        )
    finally:
        session.close()


@app.get("/api/emotion/{user_id}", response_model=EmotionResponse)
def get_emotion(user_id: str):
    session = Session()
    try:
        state = get_emotion_state(session, user_id)
        if not state:
            raise HTTPException(status_code=404, detail="Not found")
        scores = {e: round(getattr(state, e, 0.0), 4) for e in PLUTCHIK_EMOTIONS}
        return EmotionResponse(
            dominant_emotion=state.dominant_emotion or "none",
            turn_count=state.turn_count,
            consecutive_negative_turns=state.consecutive_negative_turns,
            crisis_flag=state.crisis_flag, scores=scores,
        )
    finally:
        session.close()


@app.get("/api/emotion/conversation/{conversation_id}")
def get_conv_emotion(conversation_id: str):
    session = Session()
    try:
        scores = _get_conv_emotion_scores(session, conversation_id)
        if not scores:
            return {"scores": {e: 0.0 for e in PLUTCHIK_EMOTIONS},
                    "dominant_emotion": "none", "turn_count": 0}
        model_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        dominant = max(model_emotions, key=lambda e: scores.get(e, 0))
        ce = get_conversation_emotion(session, conversation_id)
        return {"scores": scores, "dominant_emotion": dominant,
                "turn_count": ce.turn_count if ce else 0}
    finally:
        session.close()


@app.post("/api/end/{conversation_id}")
def end_session(conversation_id: str):
    pipeline.end_session(conversation_id)
    return {"status": "ok"}


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/api/admin/stats")
def admin_stats(current: dict = Depends(require_admin)):
    session = Session()
    try:
        total_users    = get_user_count(session)
        active_users   = get_active_user_count(session)
        total_convs    = get_conversation_count(session)
        crisis_msgs    = session.query(Message).filter(Message.crisis_flagged == True).count()
        return {
            "total_users":    total_users,
            "active_users":   active_users,
            "total_conversations": total_convs,
            "crisis_messages": crisis_msgs,
        }
    finally:
        session.close()


@app.get("/api/admin/users")
def admin_get_users(
    limit: int = 50, offset: int = 0,
    current: dict = Depends(require_admin),
):
    session = Session()
    try:
        users = get_all_users(session, limit=limit, offset=offset)
        return [
            {
                "user_id":       u.user_id,
                "name":          u.name,
                "email":         u.email,
                "age":           u.age,
                "gender":        u.gender,
                "date_of_birth": str(u.date_of_birth) if u.date_of_birth else None,
                "is_active":     u.is_active,
                "created_at":    u.created_at.isoformat() if u.created_at else None,
                "last_active":   u.last_active.isoformat() if u.last_active else None,
                "conversation_count": len(u.conversations),
            }
            for u in users
        ]
    finally:
        session.close()


@app.get("/api/admin/users/{user_id}/detail")
def admin_user_detail(user_id: str, current: dict = Depends(require_admin)):
    session = Session()
    try:
        user = get_user(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        emotion = get_emotion_state(session, user_id)
        convs   = (session.query(Conversation)
                   .filter(Conversation.user_id == user_id)
                   .order_by(Conversation.started_at.desc()).all())

        conv_list = []
        for c in convs:
            ce    = get_conversation_emotion(session, c.conversation_id)
            m_cnt = session.query(Message).filter(
                Message.conversation_id == c.conversation_id
            ).count()
            conv_list.append({
                "conversation_id":   c.conversation_id,
                "name_conversation": c.name_conversation or "New conversation",
                "started_at":        c.started_at.isoformat() if c.started_at else None,
                "ended_at":          c.ended_at.isoformat() if c.ended_at else None,
                "message_count":     m_cnt,
                "dominant_emotion":  ce.dominant_emotion if ce else None,
                "emotion_scores":    {e: round(getattr(ce, e, 0.0), 4) for e in PLUTCHIK_EMOTIONS} if ce else {},
            })

        return {
            "user_id":       user.user_id,
            "name":          user.name,
            "email":         user.email,
            "age":           user.age,
            "gender":        user.gender,
            "date_of_birth": str(user.date_of_birth) if user.date_of_birth else None,
            "is_active":     user.is_active,
            "created_at":    user.created_at.isoformat() if user.created_at else None,
            "last_active":   user.last_active.isoformat() if user.last_active else None,
            "global_emotion": {
                "dominant":    emotion.dominant_emotion if emotion else None,
                "turn_count":  emotion.turn_count if emotion else 0,
                "crisis_flag": emotion.crisis_flag if emotion else False,
                "scores":      {e: round(getattr(emotion, e, 0.0), 4)
                                for e in PLUTCHIK_EMOTIONS} if emotion else {},
            },
            "conversations": conv_list,
        }
    finally:
        session.close()


@app.patch("/api/admin/users/{user_id}/toggle-active")
def admin_toggle_user(user_id: str, current: dict = Depends(require_admin)):
    session = Session()
    try:
        user = get_user(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Not found")
        user.is_active = not user.is_active
        session.commit()
        return {"user_id": user_id, "is_active": user.is_active}
    finally:
        session.close()


@app.get("/api/admin/conversations")
def admin_get_conversations(
    limit: int = 50, offset: int = 0,
    current: dict = Depends(require_admin),
):
    session = Session()
    try:
        convs = get_all_conversations_admin(session, limit=limit, offset=offset)
        result = []
        for c in convs:
            user  = get_user(session, c.user_id)
            ce    = get_conversation_emotion(session, c.conversation_id)
            m_cnt = session.query(Message).filter(
                Message.conversation_id == c.conversation_id
            ).count()
            result.append({
                "conversation_id":   c.conversation_id,
                "name_conversation": c.name_conversation or "New conversation",
                "user_id":           c.user_id,
                "user_name":         user.name if user else None,
                "user_email":        user.email if user else None,
                "started_at":        c.started_at.isoformat() if c.started_at else None,
                "ended_at":          c.ended_at.isoformat() if c.ended_at else None,
                "message_count":     m_cnt,
                "dominant_emotion":  ce.dominant_emotion if ce else None,
                "crisis_count": session.query(Message).filter(
                    Message.conversation_id == c.conversation_id,
                    Message.crisis_flagged == True,
                ).count(),
            })
        return result
    finally:
        session.close()


# ============================================================
# Static + Admin pages
# ============================================================
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/admin")
@app.get("/admin/{path:path}")
def serve_admin(path: str = ""):
    return FileResponse("static/admin.html")


@app.get("/")
@app.get("/{path:path}")
def root(path: str = ""):
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)