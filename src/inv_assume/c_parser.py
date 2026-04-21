"""Compatibility wrapper for invariant C parsing."""

try:
    from invariant_module.inv_assume.c_parser import CParser
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.c_parser import CParser

__all__ = ["CParser"]
