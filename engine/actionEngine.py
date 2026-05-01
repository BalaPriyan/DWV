import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

from LLMConnect.openRouterEngine import openRouterEngine
from LLMConnect.ollamaEngine import ollamaEngine

logger = logging.getLogger(__name__)

class ActionEngine:
    def __init__(self):
        self.system_prompt = """You are DWV, an AI OS Agent. You have the ability to execute terminal commands on the user's Windows machine.
You must respond ONLY in valid JSON format. Do not include markdown blocks or any conversational text outside the JSON.

IMPORTANT: If you decide to use PowerShell for complex commands, you MUST use the `-NoProfile` flag (e.g., `powershell -NoProfile -Command "..."`) to prevent script loading errors on the user's machine.

If the user asks to open an app, run a command, or perform a system action:
{
    "action": "execute",
    "command": "<windows terminal command here, e.g. start notepad>",
    "message": "Opening it for you."
}

If the user is just asking a question or making conversation:
{
    "action": "chat",
    "message": "Hello! I am doing well."
}
"""
        
        self.provider = os.environ.get("ACTIVE_PROVIDER", "ollama").lower()
        logger.info(f"Action Engine initializing with provider: {self.provider}")
        
        if self.provider == "openrouter":
            self.engine = openRouterEngine(self.system_prompt)
        else:
            self.engine = ollamaEngine(self.system_prompt)

    def execute(self, text):
        logger.debug(f"ActionEngine routing text to {self.provider} engine...")
        raw_response = self.engine.generate(text)
        
        logger.info(f"LLM Raw Output: {raw_response}")
        
        # Clean up potential markdown formatting the LLM might have added
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        try:
            data = json.loads(cleaned_response.strip())
            
            action = data.get("action", "chat")
            message = data.get("message", "I executed the command.")
            
            if action == "execute":
                command = data.get("command")
                if command:
                    logger.warning(f"EXECUTING SYSTEM COMMAND: {command}")
                    os.system(command)
                else:
                    logger.error("Action was execute, but no command provided.")
                    
            return message
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM: {raw_response}")
            return "I had an internal error and couldn't process my thoughts properly."
