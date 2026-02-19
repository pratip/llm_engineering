import os
from typing import Iterable
from dotenv import load_dotenv

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
from time import monotonic

from openai import OpenAI, Stream

load_dotenv(override=True)

console = Console()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
OPEN_ROUTER_BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1/")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")
openai_client: OpenAI = OpenAI(base_url=OPEN_ROUTER_BASE_URL, api_key=OPEN_ROUTER_API_KEY)

conversation_history:list = []


def chat_completion(
    stream: bool = False, model: str = "openrouter/free", **kwargs
):  # -> Any | Generator[str, None, None]:
    if not stream:
        return openai_client.chat.completions.create(
      model=model,
      messages=conversation_history,
      **kwargs
    )
    def stream_generator():
        response_stream = openai_client.chat.completions.create(
      model=model,
      extra_headers={
        "HTTP-Referer": "http://localhost/", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "LocalHost", # Optional. Site title for rankings on openrouter.ai.
      },
      messages=conversation_history,
      stream=True,
      **kwargs
    )
        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if not delta:
                continue
            text = getattr(delta, "content", None)
            if text:
                yield text
    return stream_generator()


# --- Renderer: consume any text-delta iterator and show live Markdown ---
def render_stream_as_markdown(
  deltas: Iterable[str],
  *,
  min_interval: float = 0.05,
  refresh_per_second: int = 30,
  final_newline: bool = True,
) -> str:
    """
  Consumes an iterator of text deltas and renders them as live Markdown using rich.
  Returns the final aggregated text when done.
  """
    buffer = "🤖 "
    last_render = 0.0

    with Live(
        Markdown(""), console=console, refresh_per_second=refresh_per_second
    ) as live:
        for piece in deltas:
            buffer += piece
            now = monotonic()
            if now - last_render >= min_interval:
                live.update(Markdown(buffer), refresh=True)
                last_render = now

        # Final render
        live.update(Markdown(buffer), refresh=True)

    if final_newline:
        console.print()  # cosmetic newline after live block
    return buffer


if __name__ == "__main__":
  print("🚀 Starting chat conversation with Ollama...")
  role_prompt = input(
    "Enter what role you want your assistant to play (e.g., 'You are a helpful assistant.'). Press enter for this default: \n👤 >>> "
  )
  conversation_history.append({"role": "system", "content": role_prompt or "You are a helpful assistant."})
  
  while True:
    print("❓ Ask your question to Ollama! (type 'exit' to quit 🛑)")
    user_input = input("⌨ >>> ")
    if user_input.lower() == "exit":
      print("👋 Ending conversation. Goodbye!")
      break

    conversation_history.append({"role": "user", "content": user_input})
    print("🖊 Ollama is typing...")

    
    delta_iter = chat_completion(True)
    _final_streamed = render_stream_as_markdown(delta_iter, min_interval=0.08)
    conversation_history.append({"role": "assistant", "content": _final_streamed})
