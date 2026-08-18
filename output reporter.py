"""
Simple text dashboard. No scoring, no ranking — just a clean readout.
"""

from typing import List

from models.types import MarketStateResult


def format_source(result: MarketStateResult) -> str:
    if result.error:
        return "NO DATA"
    if result.is_fallback:
        return f"{result.source} (fallback)"
    return result.source


def format_state(result: MarketStateResult) -> str:
    if result.error:
        return "ERROR"
    return result.state.value


def print_report(results: List[MarketStateResult]) -> None:
    header = f"{'COIN':<6} {'TF':<5} {'STATE':<12} {'SOURCE':<18} {'STRUCTURE (recent -> latest)'}"
    print(header)
    print("-" * len(header))

    for r in results:
        structure_str = (
            " -> ".join(l.value for l in r.recent_structure)
            if r.recent_structure else "(insufficient swings)"
        )
        if r.error:
            print(f"{r.symbol:<6} {r.timeframe:<5} {'NO DATA':<12} {'-':<18} {r.error}")
            continue

        print(
            f"{r.symbol:<6} {r.timeframe:<5} {format_state(r):<12} "
            f"{format_source(r):<18} {structure_str}"
        )
