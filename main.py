import sys
from vinni.agent import ask_vinni
from vinni.voice import voice_loop


def text_mode():
    print("=" * 40)
    print("  Vinni AI — Text Mode")
    print("=" * 40)
    print("Type 'quit' to exit\n")

    conversation_history = []

    while True:
        question = input("You: ").strip()

        if question.lower() == "quit":
            print("\nGoodbye!")
            break

        if not question:
            continue

        reply, conversation_history = ask_vinni(question, conversation_history)
        print(f"\nVinni: {reply}\n")


def main():
    print("=" * 40)
    print("       Welcome to Vinni AI")
    print("=" * 40)
    print("1. Text mode")
    print("2. Voice mode")
    mode = input("\nChoose mode (1 or 2): ").strip()

    if mode == "2":
        voice_loop()
    else:
        text_mode()


if __name__ == "__main__":
    main()