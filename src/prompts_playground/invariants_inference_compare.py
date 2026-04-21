"""Compatibility wrapper for invariant playground experiments."""

try:
    from invariant_module.playground.invariants_inference_compare import run_experiments
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.playground.invariants_inference_compare import run_experiments

__all__ = ["run_experiments"]


if __name__ == "__main__":
    run_experiments()
