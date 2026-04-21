"""Compatibility package for the moved invariant instrumentation module."""

try:
    from invariant_module.inv_assume.pipeline import ASTInstrumentationPipeline
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.pipeline import ASTInstrumentationPipeline

__all__ = ["ASTInstrumentationPipeline"]
