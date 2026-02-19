from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from IPython.display import Markdown, display

from openai import OpenAI

# Add parent directory to path to import scraper
sys.path.insert(0, str(Path(__file__).parent.parent))
from scraper import fetch_website_contents

load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")

def chat_completion(system_content: str, user_content: str, base_url: str, api_key: str, model: str = "llama3.1:latest") -> str|None:
    """
  Generate a chat completion using the specified system and user content.
  
  Args:
    system_content (str): The content for the system role.
    user_content (str): The content for the user role.
    base_url (str): The base URL for the Ollama API.
    api_key (str): The API key for authentication.
    model (str): The model to use for the chat completion (default is "llama3.1:latest").
  Returns:
    str: The generated chat completion.
  """
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
      ],
  )
    return response.choices[0].message.content


def display_summary(website_summary: str|None) -> None:
    """
    Display the summary of the website in a readable format.

    Args:
      website_summary (str): The summary of the website to display.
    Returns:
      str: The formatted summary for display.
    """
    summary = summarize(url)
    display(Markdown(summary))


def summarize(url):
  """
  Summarize the contents of the website at the given url.
  
  Args:
    url (str): The URL of the website to summarize.
  Returns:
    str: A summary of the website's contents.
  """ 
  website = fetch_website_contents(url)
  response = chat_completion(
      system_content="You are a helpful assistant that summarizes websites.",
      user_content=f"Summarize the following website contents: {website}",
      base_url=OLLAMA_BASE_URL,
      api_key="ollama",
  )
  display_summary(response)

if __name__ == "__main__":
    # response = chat_completion(
    #   system_content="You are a helpful French assistant.",
    #   user_content="Tell me a fun fact about France.",
    #   base_url=OLLAMA_BASE_URL,
    #   api_key="ollama",
    # )
    # print(response)

    url = "https://edwarddonner.com"
    summary = summarize(url)
    print(summary)
