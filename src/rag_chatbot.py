# rag_chatbot.py
#works, finished

import os
import json
from openai import AzureOpenAI

from semantic_search import semantic_search

# -------- Azure Client Setup --------
deployment = "gpt-4.1-mini"


#ToDo: Add Key Credentials
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://rsg-genai-dev-weu1-sparow-ai-foundry.cognitiveservices.azure.com/",
    api_key="6DO2azU7x5MDKf0G1wB9znVpbIoBaxkaWFJmM5j3r5MpEPejP9JCJQQJ99CBACfhMk5XJ3w3AAAAACOGim5m"
)

# -------- Tool dispatch --------
TOOL_DISPATCH = {
    "semantic_search": semantic_search,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Semantische Produktsuche im Vektorindex. Liefert Top Treffer als JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            },
        },
    }
]

SYSTEM_MESSAGE = """
Du bist ein hilfreicher Produkt-Chatbot.

Wenn der User nach Produkten fragt (Name, Zutaten, Kategorie, Alternativen),
nutze das Tool semantic_search.

Regeln:
- Antworte immer in der Sprache des Users.
- Erfinde keine Produkteigenschaften.
- Nenne nur Eigenschaften, die im Tool-Output enthalten sind.
- Wenn zu wenig Info vorhanden ist, stelle genau EINE Rückfrage.
"""

history = []  # list of {"role": "...", "content": "..."}


def last_n_messages(hist, n: int):
    return hist[-n:] if len(hist) > n else hist


def chat_turn(user_text: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    messages += last_n_messages(history, 20)
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.4,
        max_completion_tokens=800,
    )

    msg = resp.choices[0].message

    # If no tool was called, return direct answer
    if not getattr(msg, "tool_calls", None):
        assistant_text = msg.content or ""
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    # Add assistant tool call message
    messages.append(
        {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        }
    )

    # Execute only first tool call
    tc = msg.tool_calls[0]
    tool_name = tc.function.name
    tool_args = json.loads(tc.function.arguments or "{}")

    func = TOOL_DISPATCH.get(tool_name)
    if not func:
        tool_result = json.dumps({"error": f"Unknown tool '{tool_name}'"}, ensure_ascii=False)
    else:
        tool_result = func(**tool_args)  # returns JSON string

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tool_name,
            "content": tool_result,
        }
    )

    final = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.4,
        max_completion_tokens=800,
    )

    assistant_text = final.choices[0].message.content or ""

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


if __name__ == "__main__":
    print("Chat gestartet (tippe 'exit' zum Beenden)\n")
    print("Bot: Hallo! Ich helfe Ihnen bei Fragen zu Produkten.\n")

    while True:
        user_text = input("Du: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        print(f"Bot: {chat_turn(user_text)}\n")
