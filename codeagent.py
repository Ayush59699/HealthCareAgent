## MINI CLAUDE CODE

import json
import os
import subprocess
from openai import OpenAI

def _load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"\'')
                    if k and not os.getenv(k):
                        os.environ[k] = v

_load_env_file()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") 
api_key = os.getenv("AZURE_OPENAI_API_KEY") 

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

SYSTEM_PROMPT = """YOU ARE A CODING AGENT RUNNING IN THE USER'S TERMINAL. YOU CAN LIST FILES, READ FILES, WRITE FILES, AND RUN SHELL COMMANDS.
USE YOUR TOOLS TO COMPLETE THE USER'S TASK, THEN BRIEFLY SUMMARIZE WHAT YOU DID.    
THE WORKING DIRECTORY IS THE FOLDER THE USER LAUNCHED YOU FROM"""


def list_file(path):
    try:
        files = os.listdir(path)
        return json.dumps(files)
    except Exception as e:
        return str(e)

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return str(e)

def write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return str(e)

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output
    except Exception as e:
        return str(e)


TOOLS = {
    "list_file": list_file,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command
}

# Responses API uses a flat tool schema (no nested "function" wrapper)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "list_file",
        "description": "List the files in the directory. Folders end with /.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "directory to list, eg. '.' "}
            },
            "required": ["path"]
        }
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a file and return its components",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "path of the file to read "}
            },
            "required": ["path"]
        }
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file. If the file already exists, its contents will be overwritten.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "path of the file to write to, eg. './newfile.txt' "},
                "content": {"type": "string", "description": "content to be written"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "type": "function",
        "name": "run_command",
        "description": "Execute a shell command and return the output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "shell command to run, eg. 'dir' "}
            },
            "required": ["command"]
        }
    }
]


def run_tool(item):
    """Execute a function_call output item from the Responses API."""
    name = item.name
    args = json.loads(item.arguments)
    print(f"\n Agent Thinking.... (tool: '{name}' args: {args})")
    try:
        return str(TOOLS[name](**args))
    except Exception as error:
        return f"error: {error}"


def run_agent(user_input, previous_response_id=None):
    """Single turn of the Responses API agentic loop.

    Returns (final_text, last_response_id) so the caller can thread
    subsequent turns via previous_response_id.
    """
    # Build the input for this turn
    input_payload = user_input  # plain string on first call

    while True:
        kwargs = dict(
            model=deployment_name,
            instructions=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            input=input_payload,
        )
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id

        response = client.responses.create(**kwargs)
        previous_response_id = response.id

        # Collect function calls from the output items
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            # No tool calls — return the text response
            return response.output_text, previous_response_id

        # Execute each tool call and build function_call_output items
        tool_outputs = []
        for fc in function_calls:
            result = run_tool(fc)
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": result,
            })

        input_payload = tool_outputs


def main():
    print("Mini agent is ready")
    last_response_id = None

    while True:
        user_input = input("\nUSER: ")
        if user_input.strip().lower() in ["exit", "break"]:
            break

        reply, last_response_id = run_agent(user_input, last_response_id)
        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    main()
