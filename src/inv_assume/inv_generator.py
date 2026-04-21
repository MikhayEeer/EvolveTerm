"""Compatibility wrapper for invariant generation."""

try:
    from invariant_module.inv_assume.inv_generator import InvariantGenerator
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.inv_generator import InvariantGenerator

__all__ = ["InvariantGenerator"]
