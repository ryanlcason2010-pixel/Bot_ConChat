"""
Utils package for Framework Assistant.

This package contains utility modules for:
- loader: Load and validate Excel framework data
- embedder: Generate and cache embeddings
- search: Semantic search functionality
- llm: OpenAI API interface
- intent: Intent detection
- session: Session state management
"""

from .loader import load_frameworks, validate_frameworks
from .embedder import EmbeddingEngine
from .search import SemanticSearch
from .llm import LLMClient
from .intent import detect_intent
from .session import SessionManager

__all__ = [
    'load_frameworks',
    'validate_frameworks',
    'EmbeddingEngine',
    'SemanticSearch',
    'LLMClient',
    'detect_intent',
    'SessionManager',
]
