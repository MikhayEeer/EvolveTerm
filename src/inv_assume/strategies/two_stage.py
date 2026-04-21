"""Compatibility wrapper for the two-stage invariant strategy."""

try:
    from invariant_module.inv_assume.strategies.two_stage import PromptPair, TwoStageStrategy, load_prompt_pair
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.strategies.two_stage import PromptPair, TwoStageStrategy, load_prompt_pair

__all__ = ["PromptPair", "TwoStageStrategy", "load_prompt_pair"]
