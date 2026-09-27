"""Environment/config loading."""

import os

from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.environ.get("COHERE_API")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
RESEARCH_MEMORY_DB_PATH = os.environ.get("RESEARCH_MEMORY_DB_PATH", "./data/research_memory.sqlite")
