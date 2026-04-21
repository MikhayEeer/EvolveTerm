"""Compatibility wrapper for invariant instrumentation pipeline."""

try:
    from invariant_module.inv_assume.pipeline import ASTInstrumentationPipeline, main
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.pipeline import ASTInstrumentationPipeline, main

__all__ = ["ASTInstrumentationPipeline", "main"]


if __name__ == "__main__":
    main()
