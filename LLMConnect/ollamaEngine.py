import os
import ollama
import logging

logger = logging.getLogger(__name__)

class ollamaEngine:
    def __init__(self, system_prompt):
        self.model = os.environ.get("OLLAMA_MODEL", "qwen3.5")
        self.system_prompt = system_prompt

    def generate(self, messages):
        logger.info(f"Querying Ollama ({self.model})...")
        try:
            response = ollama.chat(model=self.model, messages=messages)
            
            if hasattr(response, 'message'):
                ai_reply = response.message.content
            else:
                ai_reply = response.get('message', {}).get('content', '')
                
            return ai_reply.strip()
            
        except Exception as e:
            logger.error(f"ollamaEngine Error: {e}")
            return "I'm sorry, I couldn't connect to my local brain. Is Ollama running?"
