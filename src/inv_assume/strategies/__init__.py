"""Compatibility package for invariant generation strategies."""

try:
    from invariant_module.inv_assume.strategies import TwoStageStrategy
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.strategies import TwoStageStrategy

__all__ = ["TwoStageStrategy"]
