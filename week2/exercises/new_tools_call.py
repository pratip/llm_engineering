import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)

# --- OpenRouter (uncomment to use) ---
API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not API_KEY:
    raise RuntimeError("Missing API key. Set OPENAI_API_KEY (or OPENROUTER_API_KEY).")

openai = OpenAI(api_key=API_KEY, base_url=BASE_URL)

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


def get_ticket_price(destination_city):
    print(f"Tool called for city {destination_city}")
    price = ticket_prices.get(destination_city.lower(), "Unknown ticket price")
    return f"The price of a ticket to {destination_city} is {price}"


price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}
tools = [{"type": "function", "function": price_function}]


def handle_tool_calls(message):
    responses = []
    if getattr(message, "tool_calls", None):
        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_ticket_price":
                arguments = json.loads(tool_call.function.arguments)
                city = arguments.get("destination_city")
                price_details = get_ticket_price(city)
                responses.append(
                    {"role": "tool", "content": price_details, "tool_call_id": tool_call.id}
                )
    return responses


def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )
    while True:
        stream = openai.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, stream=True
        )
        content = ""
        finish_reason = None
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices and hasattr(chunk.choices[0], "delta"):
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    content += delta
                    yield content  # Yield accumulated content for Gradio
                finish_reason = getattr(chunk.choices[0], "finish_reason", None)
        if finish_reason == "tool_calls":
            response = openai.chat.completions.create(
                model=MODEL, messages=messages, tools=tools
            )
            message = response.choices[0].message
            responses = handle_tool_calls(message)
            messages.append(message)
            messages.extend(responses)
        else:
            break


gr.ChatInterface(fn=chat, type="messages").launch(
    server_name="0.0.0.0", server_port=7860, auth=("admin", "my_strong_password")
)
