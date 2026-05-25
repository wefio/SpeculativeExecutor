"""Speculative execution for agent tool calls.

Usage:
    from speculative import SpeculativeExecutor, SpecConfig, matcher

    spec = SpeculativeExecutor(provider, tools, SpecConfig(level="prefix"))
    spec.predict(messages, user_query)            # fire-and-forget
    spec.match(name, args)                        # cache hit?
    spec.cached_result(name, args)                # get result
    spec.stats                                    # timing/errors
    matcher.match_ground_truth(name, args, gt)    # BFCL evaluation
"""
from .executor import SpeculativeExecutor, SpecConfig, build_signatures
from .types import PredictionResult, PredictionStats
from . import matcher

__all__ = [
    "SpecConfig", "SpeculativeExecutor", "PredictionResult",
    "PredictionStats", "build_signatures", "matcher",
]
