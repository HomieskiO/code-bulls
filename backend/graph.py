"""
Thin compatibility facade.

Workflows live in workflows/; LLM in llm/; backtest in backtest/; screening in screening/.
"""
from workflows.single import app
from workflows.multi import multi_app
from llm.client import call_llm as _call_llm

# Backward-compat alias used by main.py
_call_gemini = _call_llm

__all__ = ["app", "multi_app", "_call_llm", "_call_gemini"]
