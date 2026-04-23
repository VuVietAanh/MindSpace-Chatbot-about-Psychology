# ============================================================
# pipeline.py — Intent-aware + Anti-repeat + High emotion check
# ============================================================

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from core.context import get_context_manager
from core.crisis import CrisisLevel, get_crisis_checker
from core.embedding import get_embedder
from core.emotion import get_emotion_analyzer
from core.generator import get_generator
from core.intent import RESPONSE_STRATEGY, Intent, get_intent_detector
from core.keyword_filter import RiskLevel, keyword_filter
from core.retrieval import DocumentChunker, get_retriever
from db.crud import (create_conversation, create_user, end_conversation,
                     get_emotion_state, get_recent_conversations,
                     get_recent_messages, get_user, save_message,
                     update_last_active)
from db.models import init_db

load_dotenv()


@dataclass
class PipelineResult:
    response:         str
    greeting:         str | None
    crisis_level:     str
    dominant_emotion: str
    was_flagged:      bool
    action_taken:     str
    emotion_method:   str
    intent:           str = "neutral"
    high_emotion:     str | None = None   # Tên cảm xúc nếu >= 75%


class ChatbotPipeline:

    def __init__(self, db_url: str = None):
        db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///chatbot.db")
        self._engine = init_db(db_url)

        from sqlalchemy.orm import sessionmaker
        self._Session = sessionmaker(bind=self._engine)

        print("\n⏳ Loading pipeline components...")
        self._intent_detector  = get_intent_detector()
        self._emotion_analyzer = get_emotion_analyzer()
        self._context_manager  = get_context_manager(alpha=float(os.getenv("EMA_ALPHA", 0.3)))
        self._embedder         = get_embedder()
        self._retriever        = self._load_retriever()
        self._generator        = get_generator()
        self._crisis_checker   = get_crisis_checker()
        print("✅ Pipeline ready!\n")

    def _load_retriever(self):
        retriever   = get_retriever(embedder=self._embedder)
        index_path  = "data/faiss.index"
        chunks_path = "data/chunks.json"
        kb_path     = "data/knowledge_base"

        if not os.getenv("QDRANT_HOST"):
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                retriever.load(index_path, chunks_path)
            elif os.path.exists(kb_path) and os.path.isdir(kb_path) and os.listdir(kb_path):
                print("⏳ Building FAISS index...")
                chunker = DocumentChunker(
                    chunk_size=int(os.getenv("CHUNK_SIZE", 256)),
                    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 30)),
                    window_size=int(os.getenv("WINDOW_SIZE", 1)),
                )
                chunks = chunker.chunk_directory(kb_path)
                retriever.build(chunks)
                retriever.save(index_path, chunks_path)
            else:
                print("⚠️  No knowledge base. RAG skipped.")
        else:
            if not retriever._is_built and os.path.exists(kb_path) and os.path.isdir(kb_path) and os.listdir(kb_path):
                print("⏳ Building Qdrant index...")
                chunker = DocumentChunker(
                    chunk_size=int(os.getenv("CHUNK_SIZE", 256)),
                    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 30)),
                    window_size=int(os.getenv("WINDOW_SIZE", 1)),
                )
                chunks = chunker.chunk_directory(kb_path)
                retriever.build(chunks)
        return retriever

    def setup_user(self, session: Session, user_id: str = None, name: str = None) -> tuple:
        if user_id:
            user = get_user(session, user_id)
            if not user:
                user = create_user(session, name=name)
        else:
            user = create_user(session, name=name)
        conv = create_conversation(session, user_id=user.user_id)
        return user, conv

    def get_greeting_if_returning(self, session: Session, user_id: str, name: str = None) -> str | None:
        emotion_state = get_emotion_state(session, user_id)
        if not emotion_state or emotion_state.turn_count == 0:
            return None
        last_topic = self._get_last_topic(session, user_id)
        result = self._generator.generate_greeting(
            name=name,
            dominant_emotion=emotion_state.dominant_emotion,
            last_topic=last_topic,
            consecutive_neg=emotion_state.consecutive_negative_turns,
        )
        return result.response

    def _get_last_topic(self, session: Session, user_id: str) -> str | None:
        convs = get_recent_conversations(session, user_id, limit=2)
        for conv in convs:
            messages = get_recent_messages(session, conv.conversation_id, limit=1)
            if messages:
                return messages[0].user_input[:100]
        return None

    def _get_recent_history_texts(self, session: Session, conversation_id: str, limit: int = 3) -> list[str]:
        messages = get_recent_messages(session, conversation_id, limit=limit)
        messages = list(reversed(messages))
        return [msg.user_input for msg in messages if msg.user_input]

    def process(
        self,
        user_input: str,
        user_id: str,
        conversation_id: str,
        greeting: str = None,
    ) -> PipelineResult:
        session = self._Session()
        try:
            update_last_active(session, user_id)

            # ══ Keyword Filter ══════════════════════════════
            filter_result = keyword_filter.check(user_input)
            if filter_result.is_flagged and filter_result.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                safe = self._generator.generate_safe_response(
                    risk_level=filter_result.risk_level.value,
                    recent_bot_responses=[],
                )
                save_message(session, conversation_id, user_input=user_input,
                             bot_response=safe.response, keyword_flagged=True, crisis_flagged=True)
                return PipelineResult(
                    response=safe.response, greeting=greeting,
                    crisis_level=filter_result.risk_level.value,
                    dominant_emotion="unknown", was_flagged=True,
                    action_taken="keyword_safe_response", emotion_method="skipped",
                    intent="unknown", high_emotion=None,
                )
            # ══ Denial Detection ═════════════════════════════
            import re as _re
            _DENIAL_PATTERNS = [
                r"\bno\b.{0,30}\b(hurt|harm|kill|die|suicid)\b",
                r"\b(don.t|do not|not)\b.{0,20}\b(want to|going to|mean to)?\b.{0,10}\b(hurt|harm|kill|die)\b",
                r"\b(i.m fine|i am fine|i.m okay|i.m ok|just feel|only feel)\b",
                r"\bjust (feel|sad|tired|bad|stressed)\b",
            ]
            user_denied_crisis = any(
                _re.search(p, user_input.lower()) for p in _DENIAL_PATTERNS
            )

            # ══ Intent Detection ═════════════════════════════
            intent_result = self._intent_detector.detect(user_input, use_llm=True)
            strategy      = RESPONSE_STRATEGY.get(intent_result.intent, RESPONSE_STRATEGY[Intent.NEUTRAL])

            # DEFER / REJECT → short-circuit
            if intent_result.intent in [Intent.DEFER, Intent.REJECT]:
                gen_result = self._generator.generate(
                    user_input=user_input,
                    intent_instruction=strategy["instruction"],
                    intent_max_tokens=strategy["max_tokens"],
                    intent_temperature=strategy["temperature"],
                )
                save_message(session, conversation_id, user_input=user_input,
                             bot_response=gen_result.response,
                             intent=intent_result.intent.value)
                return PipelineResult(
                    response=gen_result.response, greeting=greeting,
                    crisis_level="none", dominant_emotion="neutral",
                    was_flagged=False, action_taken=f"intent_{intent_result.intent.value}",
                    emotion_method="skipped", intent=intent_result.intent.value,
                    high_emotion=None,
                )

            # ══ Recent history ═══════════════════════════════
            recent_history = self._get_recent_history_texts(session, conversation_id, limit=3)

            # ══ Emotion Analysis ═════════════════════════════
            emotion_result = self._emotion_analyzer.analyze(
                text=user_input,
                recent_history=recent_history if recent_history else None,
            )

            # ══ Context + EMA + Shift + High Emotion ════════
            context_result = self._context_manager.process(
                session=session, user_id=user_id,
                conversation_id=conversation_id,
                current_emotion=emotion_result,
            )

            old_state        = get_emotion_state(session, user_id)
            reverse_question = self._context_manager.get_reverse_question(
                old_state=old_state, current_emotion=emotion_result,
            )

            # Venting → không hỏi ngược
            if intent_result.intent == Intent.VENTING:
                reverse_question = None

            # ══ High Emotion Check (>= 75%) ═════════════════
            high_alert  = context_result.high_emotion_alert
            high_emotion_name = None

            if high_alert.detected:
                high_emotion_name = high_alert.emotion
                # Inject check-in vào intent instruction
                strategy = dict(strategy)  # Copy để không mutate original
                strategy["instruction"] = (
                    f"{strategy['instruction']}\n\n"
                    f"IMPORTANT — HIGH EMOTION ALERT:\n"
                    f"The user's {high_alert.emotion} score is very high ({high_alert.score:.0%}). "
                    f"Before continuing, gently check if they want to keep talking:\n"
                    f'"{high_alert.check_in_prompt}"'
                )

            # ══ Crisis Check ═════════════════════════════════
            import re as _re
            _DENIAL_PATTERNS = [
                r"\bno\b.{0,30}\b(hurt|harm|kill|die|suicid)\b",
                r"\b(don.t|do not|not)\b.{0,20}\b(want to|going to|mean to)?\b.{0,10}\b(hurt|harm|kill|die)\b",
                r"\b(i.m fine|i am fine|i.m okay|i.m ok|just feel|only feel)\b",
                r"\bjust (feel|sad|tired|bad|stressed)\b",
            ]
            user_denied_crisis = any(
                _re.search(p, user_input.lower()) for p in _DENIAL_PATTERNS
            )

            crisis_result = self._crisis_checker.check(
                session=session, user_id=user_id, context_result=context_result,
            )

            if crisis_result.level == CrisisLevel.CRITICAL and not user_denied_crisis:
                safe = self._generator.generate_safe_response(
                    risk_level="critical",
                    recent_bot_responses=context_result.recent_bot_responses,  # ← THÊM
                )
                save_message(session, conversation_id, user_input=user_input,
                            bot_response=safe.response, intent=emotion_result.dominant_emotion,
                            keyword_flagged=filter_result.is_flagged, crisis_flagged=True)
                return PipelineResult(
                    response=safe.response, greeting=greeting,
                    crisis_level=crisis_result.level.value,
                    dominant_emotion=emotion_result.dominant_emotion,
                    was_flagged=filter_result.is_flagged,
                    action_taken="crisis_safe_response", emotion_method=emotion_result.method,
                    intent=intent_result.intent.value, high_emotion=high_emotion_name,
                )

            # ══ RAG ══════════════════════════════════════════════
            rag_context = ""
            # Mở rộng: dùng RAG cho tất cả intent trừ DEFER/REJECT
            use_rag = intent_result.intent not in [Intent.DEFER, Intent.REJECT]

            if use_rag and self._retriever._is_built:
                # Query kết hợp: user message + dominant emotion + topic từ history
                top_emotions = sorted(
                    context_result.combined_scores.items(), key=lambda x: -x[1]
                )[:2]
                emotion_summary = ", ".join(
                    f"{e} ({s:.0%})" for e, s in top_emotions if s > 0.05
                )

                # Tạo enriched query: emotion + nội dung user
                enriched_query = f"{emotion_summary}. {user_input}" if emotion_summary else user_input

                retrieval = self._retriever.retrieve(
                    query=enriched_query,          # ← dùng enriched query
                    top_k=int(os.getenv("RETRIEVAL_TOP_K", 3)),
                    emotion_summary=emotion_summary,
                    score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.25)),  # ← giảm ngưỡng
                )
                rag_context = retrieval.context

            # ══ Generate ═════════════════════════════════════
            emotion_shift = context_result.emotion_shift
            gen_result = self._generator.generate(
                user_input=user_input,
                rag_context=rag_context,
                emotion_summary=context_result.summary,
                recent_messages=context_result.recent_messages,
                has_history=context_result.has_history,
                consecutive_neg=context_result.consecutive_negative_turns,
                reverse_question=reverse_question,
                crisis_warning=(crisis_result.level == CrisisLevel.ALERT),
                emotion_shift_hint=emotion_shift.response_hint if emotion_shift.detected else None,
                intent_instruction=strategy["instruction"],
                intent_max_tokens=strategy["max_tokens"],
                intent_temperature=strategy["temperature"],
                recent_bot_responses=context_result.recent_bot_responses,
            )

            # ══ Save ══════════════════════════════════════════
            save_message(
                session, conversation_id,
                user_input=user_input,
                bot_response=gen_result.response,
                intent=emotion_result.dominant_emotion,
                keyword_flagged=filter_result.is_flagged,
                crisis_flagged=crisis_result.is_crisis,
            )

            return PipelineResult(
                response=gen_result.response,
                greeting=greeting,
                crisis_level=crisis_result.level.value,
                dominant_emotion=emotion_result.dominant_emotion,
                was_flagged=filter_result.is_flagged,
                action_taken=crisis_result.action,
                emotion_method=emotion_result.method,
                intent=intent_result.intent.value,
                high_emotion=high_emotion_name,
            )

        finally:
            session.close()

    def end_session(self, conversation_id: str):
        session = self._Session()
        try:
            end_conversation(session, conversation_id)
        finally:
            session.close()