# ============================================================
# db/models.py
# ============================================================

import uuid

from sqlalchemy import (Boolean, Column, Date, DateTime, Float, ForeignKey,
                        Integer, String, Text, create_engine)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


# ============================================================
# USERS
# ============================================================
class User(Base):
    __tablename__ = "users"

    user_id       = Column(String, primary_key=True, default=generate_uuid)

    # Auth (mới)
    email         = Column(String, unique=True, nullable=True)
    password_hash = Column(String, nullable=True)
    role          = Column(String, default="user")          # "user" | "admin"

    # Profile
    name          = Column(String, nullable=True)
    age           = Column(Integer, nullable=True)          # mới
    gender        = Column(String, nullable=True)           # mới — "male"|"female"|"other"|None
    date_of_birth = Column(Date, nullable=True)             # mới

    # Meta
    language      = Column(String, default="en")
    is_active     = Column(Boolean, default=True)           # mới — admin có thể disable
    created_at    = Column(DateTime, server_default=func.now())
    last_active   = Column(DateTime, onupdate=func.now())

    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    emotion_state = relationship("EmotionState", back_populates="user", uselist=False)


# ============================================================
# CONVERSATIONS
# ============================================================
class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id   = Column(String, primary_key=True, default=generate_uuid)
    user_id           = Column(String, ForeignKey("users.user_id"), nullable=False)

    # Tên hội thoại — auto-update từ tin nhắn đầu, user có thể đổi tên
    name_conversation = Column(String, nullable=True, default="New conversation")

    started_at = Column(DateTime, server_default=func.now())
    ended_at   = Column(DateTime, nullable=True)

    # Relationships
    user                 = relationship("User", back_populates="conversations")
    messages             = relationship("Message", back_populates="conversation")
    conversation_emotion = relationship(
        "ConversationEmotion", back_populates="conversation", uselist=False
    )


# ============================================================
# MESSAGES
# ============================================================
class Message(Base):
    __tablename__ = "messages"

    message_id      = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"), nullable=False)
    user_input      = Column(Text, nullable=False)
    bot_response    = Column(Text, nullable=True)
    intent          = Column(String, nullable=True)
    keyword_flagged = Column(Boolean, default=False)
    crisis_flagged  = Column(Boolean, default=False)
    timestamp       = Column(DateTime, server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


# ============================================================
# EMOTION STATE (global — tích lũy toàn bộ lịch sử user)
# ============================================================
class EmotionState(Base):
    __tablename__ = "emotion_state"

    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)

    anger        = Column(Float, default=0.0)
    disgust      = Column(Float, default=0.0)
    fear         = Column(Float, default=0.0)
    joy          = Column(Float, default=0.0)
    sadness      = Column(Float, default=0.0)
    surprise     = Column(Float, default=0.0)
    trust        = Column(Float, default=0.0)
    anticipation = Column(Float, default=0.0)

    dominant_emotion           = Column(String, nullable=True)
    consecutive_negative_turns = Column(Integer, default=0)
    turn_count                 = Column(Integer, default=0)
    crisis_flag                = Column(Boolean, default=False)
    last_crisis_at             = Column(DateTime, nullable=True)
    last_updated               = Column(DateTime, server_default=func.now(), onupdate=func.now())
    just_carried_over          = Column(Boolean, default=False)

    user = relationship("User", back_populates="emotion_state")


# ============================================================
# CONVERSATION EMOTION (per-conversation snapshot) — MỚI
# Reset mỗi conversation — dùng cho emotion panel + admin
# ============================================================
class ConversationEmotion(Base):
    __tablename__ = "conversation_emotion"

    conversation_id = Column(String, ForeignKey("conversations.conversation_id"), primary_key=True)

    anger        = Column(Float, default=0.0)
    disgust      = Column(Float, default=0.0)
    fear         = Column(Float, default=0.0)
    joy          = Column(Float, default=0.0)
    sadness      = Column(Float, default=0.0)
    surprise     = Column(Float, default=0.0)
    trust        = Column(Float, default=0.0)
    anticipation = Column(Float, default=0.0)

    dominant_emotion = Column(String, nullable=True)
    turn_count       = Column(Integer, default=0)
    last_updated     = Column(DateTime, server_default=func.now(), onupdate=func.now())

    conversation = relationship("Conversation", back_populates="conversation_emotion")


# ============================================================
# DB Init
# ============================================================
def init_db(db_url: str = "sqlite:///chatbot.db"):
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    print(f"✅ Database initialized: {db_url}")
    return engine


if __name__ == "__main__":
    init_db()