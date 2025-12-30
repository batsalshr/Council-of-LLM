"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Credit-safe defaults
COUNCIL_MODELS = [
    "google/gemini-2.5-flash",        # ✅ cheap, reliable
    "openai/gpt-5.1",                 # optional (will fail if credits low)
    "google/gemini-2.5-pro-preview",  # optional (will fail if credits low)
    "anthropic/claude-sonnet-4.5",    # optional (will fail if credits low)
    "x-ai/grok-4",                    # optional (will fail if credits low)
]

# ✅ make chairman cheap so Stage 3 always succeeds
CHAIRMAN_MODEL = "google/gemini-2.5-flash"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DATA_DIR = "data/conversations"
