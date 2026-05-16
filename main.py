from vinni.agent import ask_vinni


def main():
    print("=" * 40)
    print("  Vinni AI — Ask me about Vidhya")
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


if __name__ == "__main__":
    main()