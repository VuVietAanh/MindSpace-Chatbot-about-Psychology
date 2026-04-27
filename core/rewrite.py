# ============================================================
# core/rewrite.py
# Rewrite/clarify user input trước khi đưa vào RAG retrieval
# Giúp RAG tìm được tài liệu phù hợp hơn với câu hỏi mơ hồ
# ============================================================

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class RewriteResult:
    original:  str
    rewritten: str
    was_rewritten: bool   # True nếu thực sự được rewrite


class QueryRewriter:

    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    def rewrite(
        self,
        text: str,
        emotion_context: str = None,
        recent_history: list[str] = None,
    ) -> RewriteResult:
        """
        Rewrite câu input của user thành dạng rõ ràng hơn cho RAG.

        Ví dụ:
        Input:    "I don't know what to do anymore"
        Rewrite:  "How to cope with feeling lost and overwhelmed"

        Input:    "haizzzz"
        Rewrite:  "Feeling exhausted and emotionally drained"

        Không rewrite nếu:
        - Câu đã đủ rõ (> 8 từ và không mơ hồ)
        - Câu là câu hỏi trực tiếp
        """
        text = text.strip()
        if not text:
            return RewriteResult(original=text, rewritten=text, was_rewritten=False)

        # Không rewrite nếu đã là câu hỏi rõ ràng
        if text.endswith("?") and len(text.split()) > 5:
            return RewriteResult(original=text, rewritten=text, was_rewritten=False)

        # Không rewrite nếu đã đủ dài và rõ ràng
        if len(text.split()) > 12:
            return RewriteResult(original=text, rewritten=text, was_rewritten=False)

        try:
            # Build context
            context_parts = []
            if recent_history:
                context_parts.append("Recent conversation: " + " | ".join(recent_history[-2:]))
            if emotion_context:
                context_parts.append("Emotional context: " + emotion_context)

            context_str = "\n".join(context_parts) if context_parts else ""

            system_prompt = (
                "You rewrite vague or short emotional messages into clear search queries "
                "for a mental health knowledge base.\n"
                "Rules:\n"
                "- Output ONLY the rewritten query, nothing else\n"
                "- Keep it 5-10 words\n"
                "- Preserve the emotional tone exactly\n"
                "- Make it search-friendly (noun phrases work best)\n"
                "- If the message is already clear, return it unchanged\n"
                "Examples:\n"
                "  'haizzzz' → 'feeling exhausted and emotionally drained'\n"
                "  'idk' → 'feeling confused and uncertain'\n"
                "  'everything is falling apart' → 'coping with life falling apart overwhelmed'\n"
                "  'my dad keeps pressuring me' → 'dealing with family pressure and stress'\n"
            )

            user_msg = f"{context_str}\n\nRewrite this: \"{text}\"" if context_str else f"Rewrite this: \"{text}\""

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=30,
            )

            rewritten = response.choices[0].message.content.strip().strip('"').strip("'")

            # Nếu rewrite quá khác hoặc quá dài → giữ nguyên
            if not rewritten or len(rewritten.split()) > 15:
                return RewriteResult(original=text, rewritten=text, was_rewritten=False)

            was_rewritten = rewritten.lower() != text.lower()
            return RewriteResult(
                original=text,
                rewritten=rewritten,
                was_rewritten=was_rewritten,
            )

        except Exception as e:
            print(f"⚠️ Rewrite failed: {e}")
            return RewriteResult(original=text, rewritten=text, was_rewritten=False)


# ── Singleton ─────────────────────────────────────────────────
_rewriter_instance = None

def get_rewriter() -> QueryRewriter:
    global _rewriter_instance
    if _rewriter_instance is None:
        _rewriter_instance = QueryRewriter()
    return _rewriter_instance