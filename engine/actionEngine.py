import os
import json
import logging
import subprocess
from dotenv import load_dotenv

load_dotenv()

from LLMConnect.openRouterEngine import openRouterEngine
from LLMConnect.ollamaEngine import ollamaEngine
from exceptions import CommandExecutionError

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
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.max_history = 10
        
        self.provider = os.environ.get("ACTIVE_PROVIDER", "ollama").lower()
        logger.info(f"Action Engine initializing with provider: {self.provider}")
        
        if self.provider == "openrouter":
            self.engine = openRouterEngine()
        else:
            self.engine = ollamaEngine()

    def execute(self, text):
        self.messages.append({"role": "user", "content": text})
        
        logger.debug(f"ActionEngine routing text to {self.provider} engine...")
        raw_response = self.engine.generate(self.messages)
        
        logger.info(f"LLM Raw Output: {raw_response}")
        
        self.messages.append({"role": "assistant", "content": raw_response})
        
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        cleaned_response = cleaned_response.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            
        try:
            data = json.loads(cleaned_response.strip())
            
            action = data.get("action", "chat")
            message = data.get("message", "I executed the command.")
            
            if action == "execute":
                command = data.get("command")
                if command:
                    logger.warning(f"EXECUTING SYSTEM COMMAND: {command}")
                    try:
                        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            feedback = f"Command succeeded. Output: {result.stdout}"
                        else:
                            feedback = f"Command failed with error: {result.stderr}"
                            message = f"I tried to run that, but there was an error: {result.stderr.splitlines()[0] if result.stderr else 'Unknown error'}"
                            raise CommandExecutionError(feedback)
                            
                        logger.info(feedback)
                        self.messages.append({"role": "system", "content": f"EXECUTION_RESULT: {feedback}"})
                        
                    except Exception as e:
                        error_msg = f"Failed to execute command: {e}"
                        logger.error(error_msg)
                        self.messages.append({"role": "system", "content": f"EXECUTION_ERROR: {error_msg}"})
                        if not isinstance(e, CommandExecutionError):
                            message = f"I couldn't execute that command: {e}"
                else:
                    logger.error("Action was execute, but no command provided.")
                    
            if len(self.messages) > self.max_history * 2:
                self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2):]
                
            return message
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM: {raw_response}")
            return "I had an internal error and couldn't process my thoughts properly."
