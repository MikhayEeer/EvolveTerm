"""Compatibility wrapper for invariant SeaHorn verification."""

try:
    from invariant_module.inv_assume.verifier import SeaHornVerifier
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.verifier import SeaHornVerifier

__all__ = ["SeaHornVerifier"]
