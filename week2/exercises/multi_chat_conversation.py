import os
from typing import List

from dotenv import load_dotenv

from rich.markdown import Markdown
from rich.console import Console

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv(override=True)
console = Console()
OPEN_ROUTER_BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1/")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")
openai_client: OpenAI = OpenAI(base_url=OPEN_ROUTER_BASE_URL, api_key=OPEN_ROUTER_API_KEY)

deepseek_model = "deepseek/deepseek-r1-0528:free"
gpt_oss_model = "openai/gpt-oss-20b:free"
router_model = "openrouter/free"

# assistant_conversation_history: list[ChatCompletionMessageParam] = [
#     {
#         "role": "system",
#         "content": "You are a helpful customer support agent assisting a customer with their order issue.",
#     },
#     {"role": "assistant", "content": "Hello! How can I assist you today?"},
#     {
#         "role": "user",
#         "content": "Hi, I'm having trouble with my order. It hasn't arrived yet.",
#     },
# ]

# customer_conversation_history: list[ChatCompletionMessageParam] = [
#   {"role": "system", "content": "You are a troubled and agitated customer asking for help from a customer support agent. Your package hasn't arrived and you are very frustrated about it. You want the issue resolved ASAP and want to know exactly when you can expect the delivery."},
#   {"role": "user", "content": "Hello! How can I assist you today?"},
#   {"role": "assistant", "content": "Hi, I'm having trouble with my order. It hasn't arrived yet."},
# ]

# conversation_history: list[ChatCompletionMessageParam] = []

# Persona system messages only (no assistant greetings)
assistant_conversation_history: List[ChatCompletionMessageParam] = [
  {
    "role": "system",
    "content": (
      "You are a helpful customer support agent assisting a customer "
      "with their order issue. Be empathetic and provide concrete steps."
    ),
  }
]

customer_conversation_history: List[ChatCompletionMessageParam] = [
  {
    "role": "system",
    "content": (
      "You are a troubled and agitated customer. Your package hasn't arrived. "
      "Be impatient but not abusive; push for a concrete delivery date."
    ),
  }
]

conversation_history: List[ChatCompletionMessageParam] = []

def print_conversation(user: str, message: str|None):
  if message is not None:
    console.print(Markdown(f"**{user}:** {message}\n\n"))

def save_conversation(conversation: list, type: str):
  # Save the conversation history in a plain text file.
  with open(f"{type}_conversation_history.txt", "a+") as f:
    for message in conversation:
      f.write(f"{message['role']}: {message['content']}\n")
    f.write("\n=========\n")

def seed_dialogue():
    conversation_history.clear()
    conversation_history.append({
        "role": "user",
        "content": "Hi, I'm having trouble with my order. It hasn't arrived yet.",
    })

def get_text_or_raise(resp) -> str:
    msg = resp.choices[0].message
    if msg.content is None:
        raise ValueError("Model returned no text content (possibly a tool call).")
    return msg.content


def converse(repeats: int = 5):
  seed_dialogue()

  for _ in range(repeats):
    # ----- Assistant turn -----
    if conversation_history[-1]["role"] != "user":
      raise RuntimeError("Expected last role to be 'user' before assistant turns.")

    temp_assistant = assistant_conversation_history + conversation_history
    # Save exactly what you're sending (optional)
    save_conversation(temp_assistant, "assistant_request")

    assistant_client = openai_client.chat.completions.create(
      model=router_model,
      messages=temp_assistant,
      # temperature=0.5,
    )
    assistant_reply = get_text_or_raise(assistant_client)
    print_conversation("Assistant", assistant_reply)

    conversation_history.append({"role": "assistant", "content": assistant_reply})
    # Save shared history after appending (optional)
    save_conversation(conversation_history, "shared_after_assistant")

    # ----- Customer turn -----
    if conversation_history[-1]["role"] != "assistant":
      raise RuntimeError("Expected last role to be 'assistant' before customer turns.")

    temp_customer = customer_conversation_history + conversation_history
    save_conversation(temp_customer, "customer_request")

    customer_client = openai_client.chat.completions.create(
      model=router_model,
      messages=temp_customer,
      # temperature=0.7,
    )
    customer_reply = get_text_or_raise(customer_client)
    print_conversation("Customer", customer_reply)

    # Coerce customer persona to 'user' in the shared dialogue
    conversation_history.append({"role": "user", "content": customer_reply})
    save_conversation(conversation_history, "shared_after_customer")


if __name__ == "__main__":
    # print_conversation("Assistant", assistant_conversation_history[1]["content"])
    # print_conversation("Customer", assistant_conversation_history[2]["content"])
    # We would try to converse with multiple chat models in the same conversation, and see how they respond to each other.
    converse()
