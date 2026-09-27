"""Environment/config loading."""

import os

from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.environ.get("COHERE_API")
# Read but not currently wired into any request path -- Cohere is the sole
# LLM provider in use. Reserved for a future fallback, not an active one.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
RESEARCH_MEMORY_DB_PATH = os.environ.get("RESEARCH_MEMORY_DB_PATH", "./data/research_memory.sqlite")
