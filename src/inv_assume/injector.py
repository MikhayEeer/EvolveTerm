"""Compatibility wrapper for invariant injection."""

try:
    from invariant_module.inv_assume.injector import Injector
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.injector import Injector

__all__ = ["Injector"]
