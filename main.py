# ============================================================
# main.py — Entry point
# ============================================================

from sqlalchemy.orm import sessionmaker

from db.crud import get_emotion_state
from pipeline import ChatbotPipeline


def print_emotion_summary(pipeline, user_id: str):
    """Hiển thị emotion score hiện tại của user"""
    session = sessionmaker(bind=pipeline._engine)()
    try:
        state = get_emotion_state(session, user_id)
        if not state or state.turn_count == 0:
            print("\n📊 No emotion data yet.")
            return

        EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "trust", "anticipation"]

        print("\n" + "="*50)
        print("📊 EMOTION STATE SUMMARY")
        print("="*50)
        print(f"  Dominant emotion : {state.dominant_emotion}")
        print(f"  Turns analyzed   : {state.turn_count}")
        print(f"  Consecutive neg  : {state.consecutive_negative_turns}")
        print(f"  Crisis flag      : {'🚨 YES' if state.crisis_flag else '✅ No'}")
        print(f"\n  Scores:")
        for emotion in sorted(EMOTIONS, key=lambda e: -getattr(state, e, 0)):
            score = getattr(state, emotion, 0.0)
            bar   = "█" * int(score * 20)
            print(f"    {emotion:<14} {score:.2%}  {bar}")
        print("="*50)
    finally:
        session.close()


def main():
    pipeline = ChatbotPipeline()
    session  = sessionmaker(bind=pipeline._engine)()

    print("="*60)
    print("🤖 Mental Health Support Chatbot")
    print("="*60)
    print("💡 Commands: 'quit' to exit | 'score' to see emotion state")
    print("-"*60)

    name = input("What's your name? (Press Enter to skip): ").strip() or None

    user, conv = pipeline.setup_user(session, name=name)
    user_id         = user.user_id
    conversation_id = conv.conversation_id

    # --------------------------------------------------------
    # Check returning user — generate greeting nếu có lịch sử
    # --------------------------------------------------------
    greeting = pipeline.get_greeting_if_returning(
        session=session,
        user_id=user_id,
        name=name,
    )
    session.close()

    # Hiển thị greeting hoặc lời chào mặc định
    if greeting:
        # Returning user
        print(f"\n🤖 Bot: {greeting}\n")
    else:
        # New user
        if name:
            print(f"\nHello {name}! I'm here to listen. How are you feeling today?\n")
        else:
            print("\nHello! I'm here to listen. How are you feeling today?\n")

    # --------------------------------------------------------
    # Chat loop
    # --------------------------------------------------------
    first_message = True   # Để track tin nhắn đầu tiên trong phiên

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Xem emotion score
            if user_input.lower() == "score":
                print_emotion_summary(pipeline, user_id)
                continue

            # Kết thúc
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\n📊 Final emotion summary:")
                print_emotion_summary(pipeline, user_id)
                pipeline.end_session(conversation_id)
                print("\n🤖 Bot: Take care of yourself. Remember, it's okay to reach out anytime. 💙")
                break

            # Process input
            result = pipeline.process(
                user_input=user_input,
                user_id=user_id,
                conversation_id=conversation_id,
            )

            print(f"\n🤖 Bot: {result.response}\n")
            first_message = False

        except KeyboardInterrupt:
            pipeline.end_session(conversation_id)
            print("\n\nSession ended.")
            break


if __name__ == "__main__":
    main()