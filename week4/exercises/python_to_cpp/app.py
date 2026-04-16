import io
import sys
import os
import time
from pathlib import Path
import traceback
from dotenv import load_dotenv

from openai import OpenAI
import gradio as gr

# Import variable named "CSS" from week4/styles.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from styles import CSS
from system_info import retrieve_system_info

system_info = retrieve_system_info()

language = "C++"
language_map = {
    "C++": "cpp",
    "Java": "java",
    "JavaScript": "javascript",
    "Go": "go",
    "Rust": "rust"
}

models = [
    ("Google: Gemma 3 27B (free)", "google/gemma-3-27b-it:free"),
    ("Meta: Llama 3.2 3B Instruct (free)", "llama-3.2-3b-instruct:free"),
    ("OS: OpenAI: gpt-oss-120b (free)", "openai/gpt-oss-120b:free"),
    ("Qwen: Qwen3 Coder 480B A35B (free)", "qwen/qwen3-coder:free"),
    ("NVIDIA: Nemotron 3 Super (free)", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("OpenAI: GPT-5.3-Codex", "openai/gpt-5.3-codex"),
    ("Anthropic: Claude Sonnet 4.5", "anthropic/claude-sonnet-4.5"),
    ("Google: Gemini 2.5 Pro", "google/gemini-2.5-pro")
]

load_dotenv(override=True)

# --- OpenRouter (uncomment to use) ---
API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not API_KEY:
    raise RuntimeError("Missing API key. Set OPENAI_API_KEY (or OPENROUTER_API_KEY).")

openai = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_compilation_suggestion():
    system_prompt = f"""
    You are a helpful assistant who is capable of understanding a particular system's hardware and software specifications.
    Use that knowledge to provide insights, as to how to compile a CPP program
    in the most efficient way on this system.
    You dont need to go into details. Just provide the exact command that should be used to compile.
    Your response should not contain anything else. I dont need to install any other packages. You
    will have to figure the best possible approach to compile a CPP program using the existing system capabilities. 
    """
    user_prompt = f"""
        Here are the system details:
        {system_info}
    """
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

# suggested_compilation = get_compilation_suggestion()
# print(f"Suggested compilation approach based on system details:\n{suggested_compilation}")

compilation_command = ["g++", "-O3", "-march=native", "-flto", "-fomit-frame-pointer", "-pipe", "-o", "program", "main.cpp"]


def port(model: str, python_code: str):
    system_prompt = f"""
    You are an expert programmer who is proficient in both Python and {language}.
    Your task is to convert the given Python code into {language} code.
    Do not provide anything other than the code itself. No explanations needed.
    But thorough commenting is preferred.
    Make sure to use the best practices of {language} in the generated code.
    Keep in mind the performance impact as well.
    """

    user_prompt = f"""
    Convert the following Python code to {language} code:
    ```python
    {python_code}
    ```
    It needs to run in the following environment:
    {system_info}
    Your response will be written to a file called main.cpp and then compiled using the following command:
    {' '.join(compilation_command)}
    """
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    reply = response.choices[0].message.content
    reply = reply.replace('```cpp','').replace('```rust','').replace('```','') if reply else None
    return reply

def run_python(python_code: str):
    """
    Executes the given Python code and returns the EXACT text output
    that would appear in a real terminal, including print() output,
    exceptions, and timing information.
    """

    # Capture stdout exactly
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        # Restricted globals for execution
        globals_dict = {"__builtins__": __builtins__}

        start_time = time.time()
        exec(python_code, globals_dict)
        elapsed_time = time.time() - start_time

        # Collect everything printed by the executed code
        output = sys.stdout.getvalue()

        # Restore stdout
        sys.stdout = old_stdout

        # Build final markdown output
        return (
            "✅ **Python Execution Successful**\n\n"
            f"⏱️ **Execution time:** {elapsed_time:.4f} seconds\n\n"
            "**Output:**\n"
            "```\n"
            f"{output}"
            "```\n"
        )

    except Exception:
        # Capture full traceback
        tb = traceback.format_exc()
        sys.stdout = old_stdout

        return (
            "❌ **Python Execution Failed**\n\n"
            "**Error:**\n"
            "```\n"
            f"{tb}"
            "```\n"
        )


def run_generated_code(generated_code: str):
    try:
        with open("main.cpp", "w") as f:
            f.write(generated_code)

        # Compile the code
        compile_process = os.system(' '.join(compilation_command))
        if compile_process != 0:
            return "Error compiling the generated code."

        # Run the compiled program and capture the output
        start_time = time.time()
        result = os.popen("./program").read()
        elapsed_time = time.time() - start_time
        print(f"Execution time for {language}: {elapsed_time:.4f} seconds")
        return f"✅ **{language} Execution Successful**\n\n⏱️ **Execution time:** {elapsed_time:.4f} seconds\n\n**Output:**\n```\n{result}\n```"
    except Exception as e:
        return f"Error executing {language} code: {e}"

with gr.Blocks(
    css=CSS, theme=gr.themes.Monochrome(), title=f"Port from Python to {language}"
) as ui:
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            python_code = gr.Code(
                label="Python (Original)",
                value="",
                language="python",
                lines=26)

        with gr.Column(scale=6):
            generated_code = gr.Code(
                label=f"{language} (Translated)",
                value="",
                language=language_map[language],  # type: ignore
                lines=26)

    with gr.Row(elem_classes=["control"]):
        model_dropdown = gr.Dropdown(
            label="Select model",
            choices=models,
            show_label=False,
            # value=models[0]
        )
        python_run = gr.Button("Run Python code", elem_classes=["run-btn", "py"])
        code_port_button = gr.Button(f"Port to {language}", elem_classes=["convert-btn"])
        generated_code_run = gr.Button(f"Run {language} code", elem_classes=["run-btn", "cpp"])
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            python_out = gr.Markdown(elem_classes=["py-out"])
        with gr.Column(scale=6):
            # generated_code_out = gr.TextArea(label=f"{language} result", lines=8, elem_classes=["cpp-out"])
            generated_code_out = gr.Markdown(elem_classes=["cpp-out"])

    code_port_button.click(fn=port, inputs=[model_dropdown, python_code], outputs=[generated_code])
    python_run.click(fn=run_python, inputs=[python_code], outputs=[python_out])
    generated_code_run.click(fn=run_generated_code, inputs=[generated_code], outputs=[generated_code_out])

# ui.launch(server_name="0.0.0.0", server_port=7860, auth=("admin", "my_strong_password"))
ui.launch(server_name="0.0.0.0", server_port=7860)
