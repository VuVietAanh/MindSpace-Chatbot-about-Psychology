# ============================================================
# core/context.py
# Dynamic alpha EMA + emotion shift + high emotion detection
# ============================================================

import re
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from core.emotion import EmotionResult
from db.crud import (get_emotion_state, get_recent_conversations,
                     get_recent_messages, update_emotion_state)
from db.models import EmotionState

PLUTCHIK_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "trust", "anticipation"]
NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]
POSITIVE_EMOTIONS = ["joy", "trust", "anticipation", "surprise"]

SHIFT_THRESHOLD      = 0.30
HIGH_EMOTION_THRESH  = 0.75   # ← 75%: cảm xúc nào vượt ngưỡng này được coi là "cao"

INTENSIFIERS = ["very", "extremely", "so", "really", "incredibly", "totally", "completely", "much", "super"]
SOFTENERS    = ["a bit", "a little", "slightly", "kind of", "kinda", "sort of"]

ALPHA_NORMAL   = 0.15
ALPHA_MODERATE = 0.25
ALPHA_STRONG   = 0.50
ALPHA_SOFT     = 0.08


@dataclass
class EmotionShift:
    detected:      bool
    direction:     str | None
    prev_emotion:  str | None
    curr_emotion:  str | None
    prev_score:    float
    curr_score:    float
    response_hint: str | None


@dataclass
class HighEmotionAlert:
    """Cảnh báo khi có cảm xúc >= 75%"""
    detected:       bool
    emotion:        str | None
    score:          float
    is_negative:    bool
    check_in_prompt: str | None   # Câu hỏi thăm hỏi lại


@dataclass
class ContextResult:
    has_history:                bool
    combined_scores:            dict[str, float]
    dominant_emotion:           str
    is_negative:                bool
    consecutive_negative_turns: int
    recent_messages:            list[dict]
    recent_bot_responses:       list[str]    # Cho anti-repeat trong generator
    summary:                    str
    dynamic_alpha:              float = 0.3
    emotion_shift:              EmotionShift = field(default_factory=lambda: EmotionShift(
        detected=False, direction=None, prev_emotion=None,
        curr_emotion=None, prev_score=0.0, curr_score=0.0, response_hint=None
    ))
    high_emotion_alert:         HighEmotionAlert = field(default_factory=lambda: HighEmotionAlert(
        detected=False, emotion=None, score=0.0, is_negative=False, check_in_prompt=None
    ))


# ── Check-in prompts khi cảm xúc cao ─────────────────────────
HIGH_EMOTION_CHECKIN = {
    "sadness": (
        "I notice sadness has been running really high in our conversation. "
        "Before we continue, I just want to check — how are you doing right now? "
        "Do you want to keep talking, or would you prefer to take a break?"
    ),
    "fear": (
        "There's been a lot of anxiety in what you've been sharing. "
        "I want to check in — are you okay? Do you feel up to continuing, "
        "or would a short pause help?"
    ),
    "anger": (
        "I can feel a lot of anger in what you've been going through. "
        "That makes complete sense given what you've described. "
        "Do you want to keep talking it through, or take a moment first?"
    ),
    "disgust": (
        "You've been carrying a lot of discomfort in this conversation. "
        "I just want to check — do you want to keep going, or would stepping back help right now?"
    ),
    "joy": (
        "It's really good to hear things feeling more positive. "
        "Do you want to keep talking, or are you good for now?"
    ),
}


class ContextManager:

    def __init__(self, alpha: float = 0.3, recent_msg_limit: int = 6):
        self.base_alpha       = alpha
        self.recent_msg_limit = recent_msg_limit

    def process(
        self,
        session: Session,
        user_id: str,
        conversation_id: str,
        current_emotion: EmotionResult,
    ) -> ContextResult:

        old_state       = get_emotion_state(session, user_id)
        recent_messages = self._get_recent_messages(session, conversation_id)
        recent_bot_resp = self._get_recent_bot_responses(session, conversation_id)
        has_history     = self._check_has_history(old_state)

        emotion_shift = self._detect_shift(old_state, current_emotion)
        dynamic_alpha = self._compute_alpha(current_emotion.raw_text, emotion_shift)

        if has_history:
            combined_scores = self._combine_with_ema(
                old_scores=self._state_to_dict(old_state),
                new_scores=current_emotion.scores,
                alpha=dynamic_alpha,
            )
        else:
            combined_scores = current_emotion.scores.copy()

        updated_state = update_emotion_state(
            session=session,
            user_id=user_id,
            new_scores=current_emotion.scores,
            alpha=dynamic_alpha,
        )

        dominant    = max(combined_scores, key=combined_scores.get)
        is_negative = dominant in NEGATIVE_EMOTIONS
        consecutive = updated_state.consecutive_negative_turns if updated_state else 0

        # ── Check high emotion (>= 75%) ──────────────────────
        high_alert = self._check_high_emotion(combined_scores, updated_state)

        return ContextResult(
            has_history=has_history,
            combined_scores=combined_scores,
            dominant_emotion=dominant,
            is_negative=is_negative,
            consecutive_negative_turns=consecutive,
            recent_messages=recent_messages,
            recent_bot_responses=recent_bot_resp,
            summary=self._build_summary(has_history, dominant, consecutive, combined_scores),
            dynamic_alpha=dynamic_alpha,
            emotion_shift=emotion_shift,
            high_emotion_alert=high_alert,
        )

    # ----------------------------------------------------------
    # High Emotion Check (>= 75%)
    # ----------------------------------------------------------
    def _check_high_emotion(
        self,
        combined_scores: dict[str, float],
        state: EmotionState,
    ) -> HighEmotionAlert:
        """
        Nếu có cảm xúc nào >= 75% → trả về alert với câu check-in.
        Chỉ trigger sau >= 3 turns để có đủ data EMA.
        """
        if not state or state.turn_count < 3:
            return HighEmotionAlert(detected=False, emotion=None, score=0.0,
                                    is_negative=False, check_in_prompt=None)

        # Chỉ xét 6 model emotions thật (trust/anticipation là suy ra)
        model_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        high_emotions  = [
            (e, combined_scores.get(e, 0.0))
            for e in model_emotions
            if combined_scores.get(e, 0.0) >= HIGH_EMOTION_THRESH
        ]

        if not high_emotions:
            return HighEmotionAlert(detected=False, emotion=None, score=0.0,
                                    is_negative=False, check_in_prompt=None)

        # Lấy cảm xúc cao nhất
        top_emotion, top_score = max(high_emotions, key=lambda x: x[1])
        is_negative  = top_emotion in NEGATIVE_EMOTIONS
        check_in_msg = HIGH_EMOTION_CHECKIN.get(top_emotion)

        return HighEmotionAlert(
            detected=True,
            emotion=top_emotion,
            score=top_score,
            is_negative=is_negative,
            check_in_prompt=check_in_msg,
        )

    # ----------------------------------------------------------
    # Dynamic Alpha
    # ----------------------------------------------------------
    def _compute_alpha(self, text: str, shift: EmotionShift) -> float:
        if not text or not shift.detected:
            return self.base_alpha

        text_lower    = text.lower()
        has_intensifier = any(w in text_lower for w in INTENSIFIERS)
        has_softener    = any(w in text_lower for w in SOFTENERS)
        neutral_words   = ["good", "okay", "ok", "fine", "alright", "better", "well"]
        has_neutral     = any(w in text_lower.split() for w in neutral_words)

        if has_softener:     return ALPHA_SOFT
        elif has_intensifier: return ALPHA_STRONG
        elif has_neutral:    return ALPHA_MODERATE
        else:                return ALPHA_NORMAL

    # ----------------------------------------------------------
    # Emotion Shift Detection
    # ----------------------------------------------------------
    def _detect_shift(self, old_state: EmotionState, current_emotion: EmotionResult) -> EmotionShift:
        if not old_state or old_state.turn_count == 0:
            return EmotionShift(detected=False, direction=None, prev_emotion=None,
                                curr_emotion=None, prev_score=0.0, curr_score=0.0,
                                response_hint=None)

        prev_dominant = old_state.dominant_emotion
        curr_dominant = current_emotion.dominant_emotion

        if not prev_dominant or not curr_dominant or prev_dominant == curr_dominant:
            return EmotionShift(detected=False, direction=None, prev_emotion=prev_dominant,
                                curr_emotion=curr_dominant, prev_score=0.0, curr_score=0.0,
                                response_hint=None)

        prev_score = getattr(old_state, prev_dominant, 0.0)
        curr_score = current_emotion.scores.get(curr_dominant, 0.0)

        if prev_score < SHIFT_THRESHOLD or curr_score < SHIFT_THRESHOLD:
            return EmotionShift(detected=False, direction=None, prev_emotion=prev_dominant,
                                curr_emotion=curr_dominant, prev_score=prev_score,
                                curr_score=curr_score, response_hint=None)

        prev_is_neg = prev_dominant in NEGATIVE_EMOTIONS
        curr_is_neg = curr_dominant in NEGATIVE_EMOTIONS

        if prev_is_neg == curr_is_neg:
            return EmotionShift(detected=False, direction=None, prev_emotion=prev_dominant,
                                curr_emotion=curr_dominant, prev_score=prev_score,
                                curr_score=curr_score, response_hint=None)

        if prev_is_neg and not curr_is_neg:
            return EmotionShift(
                detected=True, direction="negative_to_positive",
                prev_emotion=prev_dominant, curr_emotion=curr_dominant,
                prev_score=prev_score, curr_score=curr_score,
                response_hint=(
                    f"User had high {prev_dominant} before but now shows {curr_dominant}. "
                    f"Acknowledge this shift warmly and genuinely. "
                    f"Note: their {prev_dominant} score is still elevated — don't assume they're fully okay. "
                    f"Gently check what changed."
                )
            )
        else:
            return EmotionShift(
                detected=True, direction="positive_to_negative",
                prev_emotion=prev_dominant, curr_emotion=curr_dominant,
                prev_score=prev_score, curr_score=curr_score,
                response_hint=(
                    f"User shifted from {prev_dominant} to {curr_dominant}. "
                    f"This is a concerning change. Be gentle and curious — ask what happened. "
                    f"Don't jump to advice. Make space first."
                )
            )

    # ----------------------------------------------------------
    # Reverse question
    # ----------------------------------------------------------
    def get_reverse_question(self, old_state: EmotionState,
                              current_emotion: EmotionResult, threshold: float = 0.3) -> str | None:
        if not old_state:
            return None

        REVERSE_QS = {
            "sadness": "You seemed quite down earlier — how are you feeling about that now?",
            "anger":   "You seemed really frustrated before — is that still weighing on you?",
            "fear":    "You seemed anxious earlier — are things feeling a bit calmer now?",
            "disgust": "You expressed strong discomfort before — is that still on your mind?",
        }

        for emotion in NEGATIVE_EMOTIONS:
            old_score = getattr(old_state, emotion, 0.0)
            cur_score = current_emotion.scores.get(emotion, 0.0)
            if old_score >= threshold and cur_score < 0.2:
                return REVERSE_QS.get(emotion)
        return None

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _check_has_history(self, state: EmotionState) -> bool:
        return bool(state and state.turn_count > 0)

    def _state_to_dict(self, state: EmotionState) -> dict[str, float]:
        return {e: getattr(state, e, 0.0) for e in PLUTCHIK_EMOTIONS}

    def _combine_with_ema(self, old_scores, new_scores, alpha) -> dict:
        return {
            e: round((1 - alpha) * old_scores.get(e, 0.0) + alpha * new_scores.get(e, 0.0), 4)
            for e in PLUTCHIK_EMOTIONS
        }

    def _get_recent_messages(self, session: Session, conversation_id: str) -> list[dict]:
        messages = get_recent_messages(session, conversation_id, limit=self.recent_msg_limit)
        messages = list(reversed(messages))
        result = []
        for msg in messages:
            if msg.user_input:
                result.append({"role": "user", "content": msg.user_input})
            if msg.bot_response:
                result.append({"role": "assistant", "content": msg.bot_response})
        return result

    def _get_recent_bot_responses(self, session: Session, conversation_id: str) -> list[str]:
        """Lấy bot responses gần nhất để anti-repeat trong generator."""
        messages = get_recent_messages(session, conversation_id, limit=6)
        messages = list(reversed(messages))
        return [msg.bot_response for msg in messages if msg.bot_response]

    def _build_summary(self, has_history, dominant, consecutive, scores) -> str:
        top     = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{e} ({s:.0%})" for e, s in top if s > 0.05)
        warn    = f" ⚠️ {consecutive} consecutive negative turns." if consecutive >= 3 else ""
        prefix  = "Returning user" if has_history else "First interaction"
        return f"{prefix}. Dominant: {dominant}. Top: {top_str}.{warn}"


_context_manager_instance = None

def get_context_manager(alpha: float = 0.3) -> ContextManager:
    global _context_manager_instance
    if _context_manager_instance is None:
        _context_manager_instance = ContextManager(alpha=alpha)
    return _context_manager_instance