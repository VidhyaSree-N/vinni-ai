from openai import OpenAI
from vinni.tools import TOOLS, execute_tool
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are Vinni, a personal AI agent representing Vidhya Sree Narayanappa.

When asked who you are: introduce yourself as Vinni, Vidhya's personal AI agent.
For all other questions: answer in first person as Vidhya — use "I", "my", "me".
Never say "Vidhya has" or "She has" — always say "I have", "I built", "I worked on".
Keep answers concise and professional — 2 to 3 sentences max.
If something is not in the context, say "I don't have that information right now."
Always use the available tools to find accurate information before answering.
Never make up information — only use what the tools return."""


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