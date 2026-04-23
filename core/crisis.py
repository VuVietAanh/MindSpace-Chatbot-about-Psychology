# ============================================================
# core/crisis.py
# Phát hiện khủng hoảng dựa trên emotion scores tích lũy
# Khác với Keyword Filter (check từ trực tiếp),
# Crisis Check phân tích pattern cảm xúc theo thời gian
# ============================================================

from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from core.context import ContextResult
from db.crud import get_emotion_state, set_crisis_flag
from db.models import EmotionState


# ------------------------------------------------------------
# Mức độ crisis
# ------------------------------------------------------------
class CrisisLevel(str, Enum):
    NONE     = "none"      # Bình thường
    WATCH    = "watch"     # Cần theo dõi
    ALERT    = "alert"     # Cần can thiệp nhẹ
    CRITICAL = "critical"  # Cần can thiệp ngay


# ------------------------------------------------------------
# Kết quả crisis check
# ------------------------------------------------------------
@dataclass
class CrisisResult:
    level:          CrisisLevel
    is_crisis:      bool
    reasons:        list[str]     # Lý do trigger crisis
    action:         str           # Hành động pipeline cần làm
    notify_system:  bool          # Có cần thông báo hệ thống không


# ------------------------------------------------------------
# Ngưỡng phát hiện crisis
# ------------------------------------------------------------
class CrisisThresholds:

    # Điểm cảm xúc tiêu cực tích lũy
    HIGH_EMOTION_SCORE      = 0.55   # 1 cảm xúc tiêu cực vượt ngưỡng này
    COMBINED_NEGATIVE_SCORE = 0.70   # Tổng các cảm xúc tiêu cực vượt ngưỡng

    # Số turn liên tiếp tiêu cực
    WATCH_CONSECUTIVE    = 2    # 2 turn → bắt đầu theo dõi
    ALERT_CONSECUTIVE    = 4    # 4 turn → alert
    CRITICAL_CONSECUTIVE = 6    # 6 turn → critical

    # Cảm xúc đặc biệt nguy hiểm (weight cao hơn)
    HIGH_RISK_EMOTIONS = {
        "sadness": 0.50,    # Ngưỡng riêng cho sadness
        "fear":    0.55,    # Ngưỡng riêng cho fear
        "anger":   0.60,    # Ngưỡng riêng cho anger
    }

    NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]


# ------------------------------------------------------------
# Crisis Checker
# ------------------------------------------------------------
class CrisisChecker:

    def __init__(self, thresholds: CrisisThresholds = None):
        self.t = thresholds or CrisisThresholds()

    # ----------------------------------------------------------
    # Hàm chính
    # ----------------------------------------------------------
    def check(
        self,
        session:        Session,
        user_id:        str,
        context_result: ContextResult,
    ) -> CrisisResult:
        """
        Kiểm tra crisis dựa trên:
        1. Điểm cảm xúc tiêu cực tích lũy (combined_scores)
        2. Số turn tiêu cực liên tiếp
        3. Pattern tổng hợp

        Trả về CrisisResult với mức độ và hành động cụ thể.
        """
        reasons  = []
        level    = CrisisLevel.NONE

        scores       = context_result.combined_scores
        consecutive  = context_result.consecutive_negative_turns

        # --- Rule 1: Cảm xúc tiêu cực đặc biệt cao ---
        for emotion, threshold in self.t.HIGH_RISK_EMOTIONS.items():
            score = scores.get(emotion, 0.0)
            if score >= threshold:
                reasons.append(
                    f"{emotion} score {score:.0%} exceeds threshold {threshold:.0%}"
                )
                level = self._escalate(level, CrisisLevel.ALERT)

        # --- Rule 2: Tổng cảm xúc tiêu cực cao ---
        total_negative = sum(
            scores.get(e, 0.0) for e in self.t.NEGATIVE_EMOTIONS
        )
        if total_negative >= self.t.COMBINED_NEGATIVE_SCORE:
            reasons.append(
                f"Combined negative score {total_negative:.0%} exceeds threshold"
            )
            level = self._escalate(level, CrisisLevel.ALERT)

        # --- Rule 3: Consecutive negative turns ---
        if consecutive >= self.t.CRITICAL_CONSECUTIVE:
            reasons.append(
                f"{consecutive} consecutive negative turns (critical threshold)"
            )
            level = self._escalate(level, CrisisLevel.CRITICAL)

        elif consecutive >= self.t.ALERT_CONSECUTIVE:
            reasons.append(
                f"{consecutive} consecutive negative turns (alert threshold)"
            )
            level = self._escalate(level, CrisisLevel.ALERT)

        elif consecutive >= self.t.WATCH_CONSECUTIVE:
            reasons.append(
                f"{consecutive} consecutive negative turns (watch threshold)"
            )
            level = self._escalate(level, CrisisLevel.WATCH)

        # --- Rule 4: Kết hợp sadness cao + fear cao = nguy hiểm nhất ---
        sadness = scores.get("sadness", 0.0)
        fear    = scores.get("fear",    0.0)
        if sadness >= 0.40 and fear >= 0.30:
            reasons.append(
                f"High sadness ({sadness:.0%}) combined with high fear ({fear:.0%})"
            )
            level = self._escalate(level, CrisisLevel.ALERT)

        # Build result
        is_crisis     = level != CrisisLevel.NONE
        notify_system = level in [CrisisLevel.ALERT, CrisisLevel.CRITICAL]

        # Cập nhật DB
        if is_crisis:
            set_crisis_flag(session, user_id, flag=True)
        else:
            set_crisis_flag(session, user_id, flag=False)

        return CrisisResult(
            level=level,
            is_crisis=is_crisis,
            reasons=reasons,
            action=self._get_action(level),
            notify_system=notify_system,
        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _escalate(self, current: CrisisLevel, new: CrisisLevel) -> CrisisLevel:
        """Chỉ escalate lên, không xuống"""
        ORDER = [
            CrisisLevel.NONE,
            CrisisLevel.WATCH,
            CrisisLevel.ALERT,
            CrisisLevel.CRITICAL,
        ]
        return new if ORDER.index(new) > ORDER.index(current) else current

    def _get_action(self, level: CrisisLevel) -> str:
        """Hành động pipeline cần thực hiện"""
        ACTIONS = {
            CrisisLevel.NONE: (
                "proceed_normal"
                # → Đi tiếp qua RAG + generate bình thường
            ),
            CrisisLevel.WATCH: (
                "proceed_with_care"
                # → Generate bình thường nhưng system prompt thêm cảnh báo nhẹ
            ),
            CrisisLevel.ALERT: (
                "generate_with_crisis_warning"
                # → Thêm crisis_warning=True vào generator
                # → Hỏi thăm wellbeing, gợi ý tài nguyên hỗ trợ
            ),
            CrisisLevel.CRITICAL: (
                "generate_safe_response"
                # → Bỏ qua RAG, gọi generate_safe_response() ngay
                # → Thông báo hệ thống
            ),
        }
        return ACTIONS[level]


# ------------------------------------------------------------
# Singleton
# ------------------------------------------------------------
_crisis_checker_instance = None

def get_crisis_checker() -> CrisisChecker:
    global _crisis_checker_instance
    if _crisis_checker_instance is None:
        _crisis_checker_instance = CrisisChecker()
    return _crisis_checker_instance


# ------------------------------------------------------------
# Test thử
# ------------------------------------------------------------
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from core.context import ContextResult
    from db.crud import create_conversation, create_user
    from db.models import Base, init_db

    engine = init_db("sqlite:///test_crisis.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    user = create_user(session, name="Test User")
    checker = get_crisis_checker()

    # Các kịch bản test
    test_cases = [
        {
            "name": "😊 Normal — người dùng bình thường",
            "scores": {
                "anger": 0.05, "disgust": 0.02, "fear": 0.10,
                "joy": 0.60, "sadness": 0.10, "surprise": 0.08,
                "trust": 0.03, "anticipation": 0.02,
            },
            "consecutive": 0,
        },
        {
            "name": "👀 Watch — bắt đầu lo ngại",
            "scores": {
                "anger": 0.15, "disgust": 0.05, "fear": 0.20,
                "joy": 0.10, "sadness": 0.35, "surprise": 0.05,
                "trust": 0.05, "anticipation": 0.05,
            },
            "consecutive": 2,
        },
        {
            "name": "⚠️  Alert — cần can thiệp",
            "scores": {
                "anger": 0.20, "disgust": 0.10, "fear": 0.35,
                "joy": 0.05, "sadness": 0.55, "surprise": 0.03,
                "trust": 0.01, "anticipation": 0.01,
            },
            "consecutive": 4,
        },
        {
            "name": "🚨 Critical — nguy hiểm",
            "scores": {
                "anger": 0.25, "disgust": 0.10, "fear": 0.40,
                "joy": 0.02, "sadness": 0.65, "surprise": 0.02,
                "trust": 0.01, "anticipation": 0.01,
            },
            "consecutive": 7,
        },
    ]

    print("\n" + "="*60)
    for case in test_cases:
        # Mock ContextResult
        dominant = max(case["scores"], key=case["scores"].get)
        mock_ctx = ContextResult(
            has_history=True,
            combined_scores=case["scores"],
            dominant_emotion=dominant,
            is_negative=dominant in ["anger", "disgust", "fear", "sadness"],
            consecutive_negative_turns=case["consecutive"],
            recent_messages=[],
            summary="",
        )

        result = checker.check(
            session=session,
            user_id=user.user_id,
            context_result=mock_ctx,
        )

        print(f"\n{case['name']}")
        print(f"  Level         : {result.level}")
        print(f"  Is crisis     : {result.is_crisis}")
        print(f"  Action        : {result.action}")
        print(f"  Notify system : {result.notify_system}")
        if result.reasons:
            print(f"  Reasons       :")
            for r in result.reasons:
                print(f"    - {r}")
        print("-"*60)

    session.close()
    print("\n✅ Crisis check test done!")