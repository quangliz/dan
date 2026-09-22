"""dan: System One inference and serving on open-weight causal LMs."""
__version__ = "0.0.1"

from .entrypoints.llm import LLM

__all__ = ["LLM"]
