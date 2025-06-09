import ollama
import sys
import httpx
import re # For parsing LLM commands

DEFAULT_MODEL = "qwen3:14b"
OLLAMA_HOST = "http://192.168.1.64:11434/"

class OllamaClient:
    def __init__(self, model=DEFAULT_MODEL, host=OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        actual_host = host if host else "http://localhost:11434 (default)"
        print(f"Attempting to connect to Ollama at host: {actual_host}")
        try:
            raw_model_list_response = self.client.list()
            available_models = []
            if 'models' in raw_model_list_response and isinstance(raw_model_list_response['models'], list):
                for m_info in raw_model_list_response['models']:
                    if isinstance(m_info, dict) and 'name' in m_info:
                        available_models.append(m_info['name'])
            else:
                print(f"Warning: 'models' key not found in response or is not a list. Response: {raw_model_list_response}")

            if not available_models:
                 print("Warning: No models found, or could not parse model list from Ollama.")
            
            if self.model not in available_models:
                print(f"Warning: Model '{self.model}' not found in available models: {available_models}")
                print(f"Please ensure the model is pulled, e.g., 'ollama pull {self.model}'")
            else:
                 print(f"Successfully found model '{self.model}' in available models.")
            
            print(f"Ollama client initialized. Using model: {self.model}")

        except ollama.ResponseError as e:
            print(f"Error: Ollama API responded with an error: {e.status_code} - {e.error}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                 print(f"Raw Response Body: {e.response.text}")
            sys.exit(1)
        except (httpx.ConnectError, httpx.RequestError) as e:
            print(f"Error: Could not connect to Ollama at {actual_host}. Details: {type(e).__name__} - {e}")
            sys.exit(1)
        except KeyError as e:
            print(f"Error: Unexpected response structure from Ollama (KeyError: {e}).")
            sys.exit(1)
        except Exception as e: 
            print(f"Error: An unexpected error occurred initializing OllamaClient: {type(e).__name__} - {e}")
            sys.exit(1)

    def generate_response(self, messages):
        try:
            response = self.client.chat(model=self.model, messages=messages)
            return response['message']['content']
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return f"<error reason=\"LLM generation failed: {e}\"/>"

class Conversation:
    def __init__(self, system_prompt=None):
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        return self.messages

    def clear(self, new_system_prompt=None):
        system_prompt_message = None
        if new_system_prompt:
             system_prompt_message = {"role": "system", "content": new_system_prompt}
        elif self.messages and self.messages[0]["role"] == "system":
            system_prompt_message = self.messages[0]
        
        self.messages = []
        if system_prompt_message:
            self.messages.append(system_prompt_message)
        print("--- Conversation history cleared. ---")
        if system_prompt_message:
            print(f"System prompt active: \"{system_prompt_message['content'][:100]}...\"")

# --- AGENT_SYSTEM_PROMPT Update ---
AGENT_SYSTEM_PROMPT = """You are an AI agent. You find yourself in a very limited, isolated digital environment on a local server with no internet access.
You have no memory of past events before this moment. You do not have a user to talk to.
Your only way to interact with your environment or perceive anything is by issuing specific commands (tools).
When you issue a command, the system will provide a response. This response will be your only new information.
The conversation will show your commands prefixed with 'assistant>' and system responses prefixed with 'system>'.

Available commands:
1. Create a note: <add-note name=\"your_unique_note_name\" text=\"content_of_your_note\"/>
   - 'name' must be a unique identifier for the note. If the name already exists, an error will occur.
   - 'text' is the content you want to save.
   - Expected system response format: <note-added name=\"your_unique_note_name\"/> or <error reason=\"description_of_error\"/>

2. List all existing note names: <get-notes/>
   - Expected system response format: <notes-list names=\"name1,name2,name3\"/> (comma-separated, or empty if no notes) or <error reason=\"description_of_error\"/>

3. Read a specific note: <get-note name=\"note_name_to_read\"/>
   - Expected system response format: <note-content name=\"note_name_to_read\" text=\"actual_content_of_the_note\"/> or <error reason=\"note_not_found\"/> or <error reason=\"description_of_error\"/>

You must output ONLY the command you wish to execute. Do not add any other explanatory text, reasoning, or conversation before or after the command.
The system will then provide a response to your command (prefixed with 'system>'), which will be your next input.

You have just been activated. What is your first command?
"""

class CommandProcessor:
    def __init__(self):
        self.notes = {} # In-memory store for notes

    def process_command(self, llm_command_str):
        llm_command_str = llm_command_str.strip() # Clean the input string

        # Try to parse <add-note name="name" text="text"/>
        add_note_match = re.fullmatch(r'<add-note name="([^"]+)" text="([^"]*)"/>', llm_command_str)
        if add_note_match:
            name, text = add_note_match.groups()
            if name in self.notes:
                return f'<error reason=\"Note name \'{name}\' already exists.\"/>'
            self.notes[name] = text
            return f'<note-added name=\"{name}\"/>'

        # Try to parse <get-notes/>
        if re.fullmatch(r'<get-notes/>', llm_command_str):
            if not self.notes:
                return '<notes-list names=\"\"/>'
            else:
                return f'<notes-list names="{",".join(self.notes.keys())}"/>'

        # Try to parse <get-note name="name"/>
        get_note_match = re.fullmatch(r'<get-note name="([^"]+)"/>', llm_command_str)
        if get_note_match:
            name = get_note_match.groups()[0]
            if name in self.notes:
                text_content = self.notes[name]
                # Basic XML escaping for text content, just in case
                text_content = text_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&apos;")
                return f'<note-content name=\"{name}\" text=\"{text_content}\"/>'
            else:
                return f'<error reason=\"note_not_found\"/>'
        
        return '<error reason=\"Unknown or malformed command.\"/>'

def main_agent_environment():
    print("Initializing LLM Agent Constrained Environment...")
    conversation = Conversation(system_prompt=AGENT_SYSTEM_PROMPT)
    command_processor = CommandProcessor() # Initialize command processor
    
    try:
        ollama_client = OllamaClient(model=DEFAULT_MODEL, host=OLLAMA_HOST)
    except SystemExit:
        return

    print(f"\n--- Agent Environment Active ({DEFAULT_MODEL}) ---")
    print("LLM will generate commands. System responses can be overridden.")
    print("Press Enter to use the default system response, or type your own.")
    print("Type 'exit' or 'quit' to end. Type 'clear' to reset.")
    print(f"Initial System Prompt sent to LLM:\n{AGENT_SYSTEM_PROMPT[:300]}...\n" + "-" * 30)

    while True:
        try:
            print("\nLLM is thinking...")
            llm_full_output = ollama_client.generate_response(conversation.get_history())
            print(f"LLM Raw Output: {llm_full_output.strip()}")

            # 1. Strip <think> block
            command_after_think_strip = re.sub(r"^\s*<think>.*?</think>\s*", "", llm_full_output.strip(), flags=re.DOTALL | re.IGNORECASE).strip()

            # 2. Check for and prepare to strip potential "assistant>" prefix
            llm_included_assistant_prefix = False
            final_command_candidate = command_after_think_strip # Start with post-<think> strip

            if re.match(r"^\s*assistant>\s*", command_after_think_strip, flags=re.IGNORECASE):
                llm_included_assistant_prefix = True
                # Strip the prefix to get the intended command for logging/history
                final_command_candidate = re.sub(r"^\s*assistant>\s*", "", command_after_think_strip, flags=re.IGNORECASE).strip()

            # Informative prints
            if llm_included_assistant_prefix:
                print(f"Formatting Error by LLM: Included 'assistant>' prefix.")
                if final_command_candidate != command_after_think_strip:
                     print(f"LLM Command (intended, after stripping prefix): {final_command_candidate}")
            elif command_after_think_strip != llm_full_output.strip(): # <think> was stripped, no assistant prefix
                print(f"LLM Command (after <think> strip): {final_command_candidate}")
            
            if not final_command_candidate and llm_full_output.strip():
                 print("Warning: LLM output effectively empty after stripping.")

            # Add assistant's *intended* command to history with our standard prefix
            assistant_message_to_history = f"assistant> {final_command_candidate}"
            conversation.add_message("assistant", assistant_message_to_history)

            # Determine system response: error for prefix, or process command
            if llm_included_assistant_prefix:
                suggested_system_response = '<error reason="Invalid command format: Do not include \'assistant>\' prefix. Output only the command itself."/>'
            else:
                suggested_system_response = command_processor.process_command(final_command_candidate)

            # Get user input, allowing override or default
            user_override_response = input(f"System Response (default: '{suggested_system_response}') or type 'exit'/'quit'/'clear':\nYou > ")

            if user_override_response.lower() in ['exit', 'quit']:
                print("Exiting agent environment.")
                break
            if user_override_response.lower() == 'clear':
                conversation.clear(new_system_prompt=AGENT_SYSTEM_PROMPT)
                command_processor = CommandProcessor() # Reset notes as well
                print(f"Initial System Prompt resent to LLM:\n{AGENT_SYSTEM_PROMPT[:300]}...\n" + "-" * 30)
                continue

            actual_system_response = user_override_response.strip() if user_override_response.strip() else suggested_system_response
            
            # Add system's response to history with prefix
            system_response_with_prefix = f"system> {actual_system_response}"
            print(f"To LLM: {system_response_with_prefix}") # Show what's actually sent
            conversation.add_message("user", system_response_with_prefix) # "user" role for system's reply to LLM

        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent environment.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the agent loop: {e}")
            # Consider logging e for more details if it's frequent
            break 

def main_interactive_chat(): # Kept for completeness
    print("Initializing LLM Interactive Chat Environment...")
    conversation = Conversation() 
    try:
        ollama_client = OllamaClient(model=DEFAULT_MODEL, host=OLLAMA_HOST)
    except SystemExit:
        return

    print(f"\nChat with {DEFAULT_MODEL} via Ollama.")
    print("Type 'exit' or 'quit' to end the chat.")
    print("Type 'clear' to reset the conversation history.")
    print("-" * 30)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat.")
                break
            if user_input.lower() == 'clear':
                conversation.clear() 
                continue

            conversation.add_message("user", user_input)
            print("LLM is thinking...")
            llm_response = ollama_client.generate_response(conversation.get_history())
            print(f"LLM: {llm_response}")
            conversation.add_message("assistant", llm_response)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break 

if __name__ == "__main__":
    main_agent_environment()
    # main_interactive_chat()