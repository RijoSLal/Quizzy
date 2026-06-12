import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "password"),
    timeout=60.0 # 60 seconds timeout
)

MODEL = "qwen3:1.7b"
TEMPERATURE = 0.3


