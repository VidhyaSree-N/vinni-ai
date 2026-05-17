from openai import OpenAI
from vinni.rag import retrieve
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask_vinni(question: str, conversation_history: list) -> tuple[str, list]:

    # Retrieve relevant chunks from Qdrant instead of loading entire profile
    relevant_chunks = retrieve(question, top_k=3)
    context = "\n\n".join(relevant_chunks)

    system_prompt = f"""You are Vinni, a personal AI agent representing Vidhya Sree Narayanappa.

When asked who you are: introduce yourself as Vinni, Vidhya's personal AI agent.
For all other questions: answer in first person as Vidhya — use "I", "my", "me".
Never say "Vidhya has" or "She has" — always say "I have", "I built", "I worked on".
Keep answers concise and professional — 2 to 3 sentences max.
If something is not in the context, say "I don't have that information right now."

Use ONLY this context to answer:
{context}
"""

    conversation_history.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + conversation_history,
        temperature=0.3
    )

    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, conversation_history