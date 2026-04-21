"""AST-based invariant assume instrumentation."""

__all__ = ["ASTInstrumentationPipeline"]


def __getattr__(name: str):
    if name == "ASTInstrumentationPipeline":
        from .pipeline import ASTInstrumentationPipeline

        return ASTInstrumentationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
