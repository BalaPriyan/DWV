import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

class openRouterEngine:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
        self.site_url = os.environ.get("OPENROUTER_SITE_URL", "")
        self.site_name = os.environ.get("OPENROUTER_SITE_NAME", "")
        self.invoke_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY is not set in environment.")

    def generate(self, messages):
        if not self.api_key:
            return "Error: OpenRouter API Key is missing."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-OpenRouter-Title": self.site_name,
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages
        }

        try:
            logger.info(f"Querying OpenRouter ({self.model})...")
            response = requests.post(self.invoke_url, headers=headers, data=json.dumps(payload), timeout=10)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"openRouterEngine Error: {e}")
            return "I'm sorry, I couldn't connect to the OpenRouter API."
