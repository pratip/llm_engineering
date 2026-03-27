from json import tool
import os

from dotenv import load_dotenv

import gradio as gr
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv(override=True)
OPEN_ROUTER_BASE_URL = os.getenv(
    "OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1/"
)
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")
openai_client: OpenAI = OpenAI(
    base_url=OPEN_ROUTER_BASE_URL, api_key=OPEN_ROUTER_API_KEY
)
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a helpful and polite assistant.",
}


def converse(
    message: str, model: str, history: list
):  # -> Generator[Any, Any, Literal[''] | None]:
    if history is None:
        history = []
    else:
        history = [{"role": h["role"], "content": h["content"]} for h in history]

    if not message or not message.strip():
        yield history, ""
        return

    # Show the user message immediately and create an empty assistant stub
    running = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},  # we'll fill this as tokens arrive
    ]
    yield running, ""  # clears textbox on first yield

    messages = [SYSTEM_MESSAGE] + history + [{"role": "user", "content": message}]

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            stream=True,
        )

        parts = []
        for chunk in response:
            # 1) Guard: chunk.choices may be empty
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue

            choice0 = choices[0]

            # 2) Some frames carry finish_reason or other metadata only
            delta = getattr(choice0, "delta", None)
            if not delta:
                continue

            # 3) delta may carry role first; ignore it
            token = getattr(delta, "content", None)
            if not token:
                # could be tool_calls or a role-only frame; skip
                continue

            parts.append(token)
            running[-1]["content"] = "".join(parts)
            yield running, ""

        # Final flush
        running[-1]["content"] = "".join(parts)
        yield running, ""

    except Exception as e:
        running[-1]["content"] = f"❌ Streaming error: {e}"
        yield running, ""


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    # gr.ChatInterface(fn=converse, type="messages").launch()
    with gr.Row(equal_height=True):
        model_select = gr.Dropdown(
            choices=[
                "openrouter/free",
                "deepseek/deepseek-r1-0528:free",
                "openai/gpt-oss-120b:free",
                "nvidia/nemotron-3-nano-30b-a3b:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemma-3-27b-it:free",
            ],
            value="openrouter/free",
            label="Model",
            scale=0,
        )
        textbox = gr.Textbox(placeholder="Type message here...", scale=1)

    textbox.submit(
        converse, inputs=[textbox, model_select, chatbot], outputs=[chatbot, textbox]
    )

demo.launch(
    server_name="0.0.0.0", server_port=7860, auth=("admin", "my_strong_password")
)
