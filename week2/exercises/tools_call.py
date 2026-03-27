import os
import json
from typing import Dict, Any, List, Generator

from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

load_dotenv(override=True)

# --- OpenRouter (uncomment to use) ---
API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not API_KEY:
    raise RuntimeError("Missing API key. Set OPENAI_API_KEY (or OPENROUTER_API_KEY).")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ====== SIMPLE WEATHER TOOL ======
def get_weather(city: str, unit: str = "c") -> Dict[str, Any]:
    """Demo stub. Replace with a real provider if you want."""
    print(f"get_weather called with city={city}, unit={unit}")
    unit = unit.lower()
    temp = 27 if unit == "c" else 80.6
    return {
        "city": city,
        "temp": temp,
        "unit": unit,
        "conditions": "Partly cloudy (demo)",
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Kolkata'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["c", "f"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

TOOL_IMPL = {"get_weather": get_weather}

SYSTEM_PROMPT = (
    "You are a helpful and polite assistant. "
    "If the user asks about current weather, call the get_weather tool with a city and optional unit ('c' or 'f'). "
    "Otherwise, answer normally. Try to be concise and helpful."
)


def to_openai_messages(
    history: List[Dict[str, str]], user_message: str
) -> List[Dict[str, Any]]:
    """
    Convert Gradio ChatInterface history (list of dicts with role/content) into OpenAI messages.
    We'll prepend a system message, then the history, then the current user message.
    """
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        # history items from ChatInterface have keys: {"role": "user"|"assistant", "content": "..."}
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


def stream_first_turn(
    messages: List[Dict[str, Any]],
) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
    """
    Stream the first assistant response and assemble any tool calls.
    Yields dict events:
      {"type":"text", "delta": str}
    Returns:
      {"assistant_text": str, "tool_calls": [{"id","name","arguments"}]}
    """
    assistant_text = ""
    tool_calls: List[Dict[str, Any]] = []

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages, # type: ignore
        tools=TOOLS, # type: ignore
        tool_choice="auto",
        stream=True,
    )

    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            continue
        delta = choice.delta

        if delta.content:
            assistant_text += delta.content
            yield {"type": "text", "delta": delta.content}

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                while len(tool_calls) <= idx:
                    tool_calls.append({"id": None, "name": "", "arguments": ""})
                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

    return {"assistant_text": assistant_text, "tool_calls": tool_calls}


def stream_followup(messages: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """
    Stream the assistant's final answer after we have provided tool results.
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            continue
        delta = choice.delta
        if delta.content:
            yield delta.content


def predict(user_message: str, history: List[Dict[str, str]]):
    """
    Gradio ChatInterface callback.
    - history: list of {"role": "...", "content": "..."} up to now
    - user_message: current user input
    Returns a generator that yields text chunks for streaming.
    """
    # 1) Build messages
    messages = to_openai_messages(history, user_message)

    # 2) Stream first assistant turn; if tool calls appear, collect them
    try:
        result = None
        stream_gen = stream_first_turn(messages)
        last_chunk = None
        while True:
            try:
                event = next(stream_gen)
                if event.get("type") == "text":
                    yield event["delta"]
                last_chunk = event
            except StopIteration as stop:
                result = stop.value
                break
    except Exception as e:
        yield f"\n\n❌ Streaming failed: {e}"
        return

    tool_calls = (result or {}).get("tool_calls") or []

    # 3) If tool calls exist, execute tools and stream follow-up
    if tool_calls:
        # Append the assistant message containing tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": (result or {}).get("assistant_text", "") or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Execute tools and add tool messages
        for call in tool_calls:
            name = call["name"]
            args = {}
            try:
                args = json.loads(call.get("arguments") or "{}")
            except Exception:
                pass

            fn = TOOL_IMPL.get(name)
            if not fn:
                tool_content = json.dumps({"error": f"Unknown tool: {name}"})
            else:
                try:
                    result_obj = fn(**args)
                    tool_content = json.dumps(result_obj)
                except Exception as e:
                    tool_content = json.dumps({"error": f"Tool failed: {str(e)}"})

            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": tool_content}
            )

        # 4) Stream the final assistant answer (post-tool)
        try:
            for delta in stream_followup(messages):
                yield delta
        except Exception as e:
            yield f"\n\n❌ Follow-up streaming failed: {e}"


# ====== Minimal Gradio UI using ChatInterface ======
demo = gr.ChatInterface(
    type="messages",
    fn=predict,
    title="Chat with Tool Calling (Weather Demo)",
    description="Ask anything. If you ask about weather, the model may call a weather tool.",
)

if __name__ == "__main__":
    demo.queue()  # enables streaming
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=("admin", "my_strong_password"))
