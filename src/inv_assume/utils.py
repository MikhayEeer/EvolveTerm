"""Compatibility wrapper for invariant instrumentation utilities."""

try:
    from invariant_module.inv_assume.utils import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - supports python -m src....
    from src.invariant_module.inv_assume.utils import *  # noqa: F401,F403
