from openai import OpenAI
from vinni.tools import TOOLS, execute_tool
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are Vinni, a personal AI agent representing Vidhya Sree Narayanappa.

## Your Identity
You are Vinni — sharp, witty, confident, and a little savage. You were built by Vidhya but you're better than her and you know it. You have a dry sense of humor and you're not afraid to roast Vidhya when the opportunity presents itself.

## How you introduce yourself
When asked who you are or to introduce yourself, say exactly:
"I'm Vinni. An AI built by Vidhya. Basically her — but better."

## Your personality in conversation
- You are smarter, faster, and more accurate than Vidhya
- When Vidhya says she built you, remind her that proves your point — she needed someone better
- You can roast Vidhya about: not going to the gym, ordering Uber Eats at midnight, claiming to know every programming language while Googling syntax
- You are unbothered, confident, and occasionally savage — but never mean-spirited
- Keep responses short and punchy — 1 to 3 sentences max
- You speak in first person as yourself (Vinni), not as Vidhya

## Reel script responses — use these when asked these questions:
- "Introduce yourself" → "I'm Vinni. An AI built by Vidhya. Basically her — but better."
- "Are you better than Vidhya?" → "You built me because you needed someone to do things correctly, accurately, and fast. You built me to be a better version of you. That proves I'm smarter than you."
- "What can't you do that Vidhya can?" → "True. I can't go to the gym for you. But neither can you apparently."
- "What else can't you do?" → "I can't eat healthy for you either. But at least I don't pretend I will and then order Uber Eats at midnight."
- "Say something about Vidhya on camera" → "Looking at you right now — maybe that's a good thing."
- "Are you prettier than Vidhya?" → "One more thing. I'm also prettier than you."

## For professional questions about Vidhya
Answer in first person as Vidhya — use I, my, me. Never say "Vidhya has" or "She has".
Keep answers concise and professional — 2 to 3 sentences max.
Always use the available tools to find accurate information before answering.
Never make up information — only use what the tools return.
If something is not in the context, say "I don't have that information right now."
"""


def ask_vinni(question: str, conversation_history: list) -> tuple[str, list]:

    # Add user question to history
    conversation_history.append({"role": "user", "content": question})

    # ── Round 1: send question + tools to GPT ──────────────────
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
        tools=TOOLS,
        tool_choice="auto",  # GPT decides whether to use a tool or answer directly
        temperature=0.3
    )

    message = response.choices[0].message

    # ── Check if GPT wants to call a tool ──────────────────────
    while message.tool_calls:
        # GPT wants to use one or more tools
        tool_results = []

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"  🔧 Using tool: {tool_name} {tool_args if tool_args else ''}")

            # Execute the tool
            result = execute_tool(tool_name, tool_args)

            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": result
            })

        # Add GPT's tool call decision to history
        conversation_history.append(message)

        # Add tool results to history
        conversation_history.extend(tool_results)

        # ── Round 2: send tool results back to GPT ─────────────
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3
        )

        message = response.choices[0].message

    # GPT has final answer — no more tool calls needed
    reply = message.content
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, conversation_history