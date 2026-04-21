"""EvolveTerm - termination analysis via LLM + RAG."""

__all__ = ["TerminationPipeline"]


def __getattr__(name: str):
    if name == "TerminationPipeline":
        from .pipeline import TerminationPipeline

        return TerminationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
