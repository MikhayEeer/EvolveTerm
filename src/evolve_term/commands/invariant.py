"""Compatibility wrapper for the invariant command handler."""

try:
    from invariant_module.command import InvariantHandler
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.command import InvariantHandler

__all__ = ["InvariantHandler"]
