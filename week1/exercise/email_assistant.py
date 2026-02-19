import os
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
openai_client: OpenAI = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
system_prompt: str = """
You are a dynamic office assistant that helps users draft and summarize emails. You can take the content of an email and provide a concise summary, or you can take a user's input and help them draft a professional email based on that input. Always ensure that the tone of the email is appropriate for a professional setting, and that the content is clear and concise.
Use markup to format your responses, such as using bullet points for summaries or bold text for important information in drafts. When summarizing, focus on the key points and main ideas of the email content. When drafting, make sure to include a clear introduction, body, and conclusion, and to address any specific points or questions the user has mentioned in their input.
"""


buffer = ""  # Accumulate streamed tokens here
last_render = 0.0         # Throttle re-renders to avoid flicker
min_interval = 0.05       # seconds between live updates (tweak as needed)


conversation_history: list = [
  {"role": "system", "content": system_prompt},
]

def summarize_email(user_prompt: str):
  """
  Summarize the given email content.

  Args:
    user_prompt (str): The content of the email to summarize.
  Returns:
    str: A summary of the email content.
  """
  conversation_history.append({"role": "user", "content": user_prompt})
  response_stream = openai_client.chat.completions.create(
    model="llama3.1:latest",
    messages=conversation_history,
    stream=True,
  )
  return response_stream
  # assistant_response = response_stream.choices[0].message.content
  # conversation_history.append({"role": "assistant", "content": assistant_response})
  # return assistant_response or "Sorry, I couldn't generate a response. Please try again."


if __name__ == "__main__":
  print("This is the email assistant module. It helps you with drafting or summarizing emails.")
  print("You can treat this as a conversation. If you want to quit, just type 'exit'.")
  while True:
    user_input = input("Please enter the email content you want to summarize or what you want to draft:\n\n>>> ")
    if user_input.lower() == "exit":
      break
    print("\nAssistant's response:\t")
    assistant_response_stream: Stream[ChatCompletionChunk] = summarize_email(user_input)

    
  # Live keeps re-rendering the Markdown panel as content grows
  with Live(Markdown(""), console=console, refresh_per_second=30) as live:
    for chunk in assistant_response_stream:
      if not chunk.choices or not chunk.choices[0].delta:
        continue
      delta = chunk.choices[0].delta
      if delta.content:
        buffer += delta.content

        now = monotonic()
        # throttle updates a bit for smoother UX
        if now - last_render >= min_interval:
          live.update(Markdown(buffer), refresh=True)
          last_render = now

    # Ensure final render (in case last tokens came quickly)
    live.update(Markdown(buffer), refresh=True)

  # console.print(Markdown(assistant_response))
  print("\n---\n")
