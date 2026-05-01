import os
import ollama
import logging

logger = logging.getLogger(__name__)

class ollamaEngine:
    def __init__(self):
        self.model = os.environ.get("OLLAMA_MODEL", "qwen2.5")

    def generate(self, messages, stream=False):
        logger.info(f"Querying Ollama ({self.model})...")
        try:
            if stream:
                return ollama.chat(model=self.model, messages=messages, stream=True)
                
            response = ollama.chat(model=self.model, messages=messages)
            
            if hasattr(response, 'message'):
                ai_reply = response.message.content
            else:
                ai_reply = response.get('message', {}).get('content', '')
                
            return ai_reply.strip()
            
        except Exception as e:
            logger.error(f"ollamaEngine Error: {e}")
            return "I'm sorry, I couldn't connect to my local brain. Is Ollama running?"
