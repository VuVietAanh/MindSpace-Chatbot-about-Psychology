# ============================================================
# db/crud.py
# ============================================================

import hashlib
import secrets
from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import (Conversation, ConversationEmotion, EmotionState,
                       Message, User, init_db)

PLUTCHIK_EMOTIONS = [
    "anger", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "anticipation"
]
NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]
ALPHA_CARRY_OVER  = 0.05


def get_session(db_url: str = "sqlite:///chatbot.db") -> Session:
    engine = create_engine(db_url, echo=False)
    return sessionmaker(bind=engine)()


# ============================================================
# AUTH HELPERS
# ============================================================

def hash_password(password: str) -> str:
    # SHA-256 không giới hạn độ dài — không cần bcrypt
    # Truncate ở 72 chars để safe nếu có bcrypt dependency ở đâu đó
    pwd    = password[:72]
    salt   = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + pwd).encode("utf-8")).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        pwd  = password[:72]
        salt, hashed = stored_hash.split(":", 1)
        return hashlib.sha256((salt + pwd).encode("utf-8")).hexdigest() == hashed
    except Exception:
        return False


# ============================================================
# USER
# ============================================================

def create_user(
    session: Session,
    name:          str  = None,
    email:         str  = None,
    password:      str  = None,
    role:          str  = "user",
    language:      str  = "en",
    age:           int  = None,
    gender:        str  = None,
    date_of_birth: date = None,
) -> User:
    user = User(
        name=name, email=email,
        password_hash=hash_password(password) if password else None,
        role=role, language=language,
        age=age, gender=gender, date_of_birth=date_of_birth,
    )
    session.add(user)
    session.flush()
    session.add(EmotionState(user_id=user.user_id))
    session.commit()
    session.refresh(user)
    return user


def get_user(session: Session, user_id: str) -> User:
    return session.query(User).filter(User.user_id == user_id).first()


def get_user_by_email(session: Session, email: str) -> User:
    if not email:
        return None
    return session.query(User).filter(User.email == email).first()


def get_user_by_name(session: Session, name: str) -> User:
    if not name:
        return None
    return (session.query(User)
            .filter(User.name.ilike(name.strip()))
            .order_by(User.created_at.desc())
            .first())


def update_user_name(session: Session, user_id: str, new_name: str):
    user = get_user(session, user_id)
    if user:
        user.name = new_name
        session.commit()


def update_user_profile(
    session: Session,
    user_id: str,
    name:          str  = None,
    age:           int  = None,
    gender:        str  = None,
    date_of_birth: date = None,
):
    user = get_user(session, user_id)
    if not user:
        return
    if name          is not None: user.name          = name
    if age           is not None: user.age            = age
    if gender        is not None: user.gender         = gender
    if date_of_birth is not None: user.date_of_birth  = date_of_birth
    session.commit()


def update_last_active(session: Session, user_id: str):
    user = get_user(session, user_id)
    if user:
        user.last_active = datetime.utcnow()
        session.commit()


def authenticate_user(session: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(session, email)
    if not user or not user.password_hash:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


# ============================================================
# CONVERSATION
# ============================================================

def create_conversation(session: Session, user_id: str,
                        name: str = "New conversation") -> Conversation:
    conv = Conversation(user_id=user_id, name_conversation=name)
    session.add(conv)
    session.flush()
    # Tạo ConversationEmotion snapshot đồng thời
    session.add(ConversationEmotion(conversation_id=conv.conversation_id))
    session.commit()
    session.refresh(conv)
    return conv


def update_conversation_name(session: Session, conversation_id: str, name: str):
    """
    Cập nhật tên conversation.
    Tự động gọi sau tin nhắn đầu tiên để lấy nội dung làm tên.
    """
    conv = session.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    if conv and (conv.name_conversation == "New conversation" or not conv.name_conversation):
        # Chỉ auto-update nếu vẫn là default name
        conv.name_conversation = name[:60] + ("..." if len(name) > 60 else "")
        session.commit()


def end_conversation(session: Session, conversation_id: str):
    conv = session.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    if conv:
        conv.ended_at = datetime.utcnow()
        session.commit()


def delete_conversation(session: Session, conversation_id: str) -> bool:
    conv = session.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    if not conv:
        return False
    session.query(ConversationEmotion).filter(
        ConversationEmotion.conversation_id == conversation_id
    ).delete()
    session.query(Message).filter(
        Message.conversation_id == conversation_id
    ).delete()
    session.delete(conv)
    session.commit()
    return True


def get_recent_conversations(session: Session, user_id: str, limit: int = 5):
    return (
        session.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.started_at.desc())
        .limit(limit)
        .all()
    )


# ============================================================
# MESSAGE
# ============================================================

def save_message(
    session: Session,
    conversation_id: str,
    user_input:      str,
    bot_response:    str  = None,
    intent:          str  = None,
    keyword_flagged: bool = False,
    crisis_flagged:  bool = False,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        user_input=user_input,
        bot_response=bot_response,
        intent=intent,
        keyword_flagged=keyword_flagged,
        crisis_flagged=crisis_flagged,
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)

    # Auto-update conversation name từ tin nhắn đầu tiên
    msg_count = session.query(Message).filter(
        Message.conversation_id == conversation_id
    ).count()
    if msg_count == 1 and user_input:
        update_conversation_name(session, conversation_id, user_input)

    return msg


def get_recent_messages(session: Session, conversation_id: str, limit: int = 10):
    return (
        session.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp.desc())
        .limit(limit)
        .all()
    )


# ============================================================
# EMOTION STATE (global)
# ============================================================

def get_emotion_state(session: Session, user_id: str) -> EmotionState:
    return session.query(EmotionState).filter(
        EmotionState.user_id == user_id
    ).first()


def update_emotion_state(
    session: Session,
    user_id: str,
    new_scores: dict,
    alpha: float = 0.3,
) -> EmotionState:
    state = get_emotion_state(session, user_id)
    if not state:
        return None

    effective_alpha = ALPHA_CARRY_OVER if state.just_carried_over else alpha

    for emotion in PLUTCHIK_EMOTIONS:
        old = getattr(state, emotion, 0.0)
        new = new_scores.get(emotion, 0.0)
        setattr(state, emotion, round((1 - effective_alpha) * old + effective_alpha * new, 4))

    state.dominant_emotion = max(PLUTCHIK_EMOTIONS, key=lambda e: getattr(state, e))

    is_negative = any(new_scores.get(e, 0) > 0.4 for e in NEGATIVE_EMOTIONS)
    state.consecutive_negative_turns = (
        state.consecutive_negative_turns + 1 if is_negative else 0
    )

    state.turn_count        += 1
    state.last_updated       = datetime.utcnow()
    state.just_carried_over  = False

    session.commit()
    session.refresh(state)
    return state


def carry_over_emotion_from_conversation(
    session: Session,
    user_id: str,
    source_conversation_id: str,
) -> bool:
    # Lấy từ ConversationEmotion thay vì tính lại từ messages
    conv_emotion = get_conversation_emotion(session, source_conversation_id)
    if not conv_emotion or conv_emotion.turn_count == 0:
        return False

    state = get_emotion_state(session, user_id)
    if not state:
        return False

    for emotion in PLUTCHIK_EMOTIONS:
        setattr(state, emotion, getattr(conv_emotion, emotion, 0.0))

    state.dominant_emotion  = conv_emotion.dominant_emotion
    state.just_carried_over = True
    state.turn_count        = max(state.turn_count, 1)
    state.last_updated      = datetime.utcnow()
    session.commit()
    return True


def reset_emotion_state(session: Session, user_id: str):
    state = get_emotion_state(session, user_id)
    if state:
        for emotion in PLUTCHIK_EMOTIONS:
            setattr(state, emotion, 0.0)
        state.dominant_emotion           = None
        state.consecutive_negative_turns = 0
        state.turn_count                 = 0
        state.crisis_flag                = False
        state.just_carried_over          = False
        session.commit()


def set_crisis_flag(session: Session, user_id: str, flag: bool = True):
    state = get_emotion_state(session, user_id)
    if state:
        state.crisis_flag = flag
        if flag:
            state.last_crisis_at = datetime.utcnow()
        session.commit()


# ============================================================
# CONVERSATION EMOTION (per-conversation)
# ============================================================

def get_conversation_emotion(session: Session, conversation_id: str) -> ConversationEmotion:
    return session.query(ConversationEmotion).filter(
        ConversationEmotion.conversation_id == conversation_id
    ).first()


def update_conversation_emotion(
    session: Session,
    conversation_id: str,
    new_scores: dict,
    alpha: float = 0.3,
) -> ConversationEmotion:
    """
    Cập nhật emotion score theo từng conversation riêng.
    Dùng EMA giống global nhưng reset mỗi conversation mới.
    """
    ce = get_conversation_emotion(session, conversation_id)
    if not ce:
        ce = ConversationEmotion(conversation_id=conversation_id)
        session.add(ce)
        session.flush()

    for emotion in PLUTCHIK_EMOTIONS:
        old = getattr(ce, emotion, 0.0)
        new = new_scores.get(emotion, 0.0)
        setattr(ce, emotion, round((1 - alpha) * old + alpha * new, 4))

    ce.dominant_emotion = max(PLUTCHIK_EMOTIONS, key=lambda e: getattr(ce, e))
    ce.turn_count      += 1
    ce.last_updated     = datetime.utcnow()

    session.commit()
    session.refresh(ce)
    return ce


# ============================================================
# ADMIN QUERIES
# ============================================================

def get_all_users(session: Session, limit: int = 100, offset: int = 0):
    return (
        session.query(User)
        .filter(User.role == "user")
        .order_by(User.created_at.desc())
        .limit(limit).offset(offset)
        .all()
    )


def get_user_count(session: Session) -> int:
    return session.query(User).filter(User.role == "user").count()


def get_active_user_count(session: Session) -> int:
    return session.query(User).filter(
        User.role == "user", User.is_active == True
    ).count()


def get_conversation_count(session: Session) -> int:
    return session.query(Conversation).count()


def get_all_conversations_admin(session: Session, limit: int = 100, offset: int = 0):
    return (
        session.query(Conversation)
        .order_by(Conversation.started_at.desc())
        .limit(limit).offset(offset)
        .all()
    )