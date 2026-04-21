"""Compatibility wrapper for invariant batch instrumentation."""

try:
    from invariant_module.inv_assume.batch_runner import BatchRunner, main
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.batch_runner import BatchRunner, main

__all__ = ["BatchRunner", "main"]


if __name__ == "__main__":
    main()
