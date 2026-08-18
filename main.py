"""
Market State Detector — Phase 1

Pipeline: OHLCV -> Swing Detection -> Market Structure -> Market State -> Report

Run:
    python main.py
"""

import logging

from config.settings import WATCHLIST, TIMEFRAMES, SWING_N, STRUCTURE_WINDOW
from data.data_router import DataRouter
from core.swings import detect_swings
from core.structure import build_structure
from core.state_classifier import classify_state
from models.types import MarketStateResult
from output.reporter import print_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def run() -> None:
    router = DataRouter()
    results = []

    for symbol in WATCHLIST:
        for timeframe in TIMEFRAMES:
            ohlcv = router.get_ohlcv(symbol, timeframe)

            if ohlcv.error:
                results.append(MarketStateResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    state=None,
                    source=None,
                    is_fallback=False,
                    recent_structure=[],
                    swing_count=0,
                    error=ohlcv.error,
                ))
                continue

            swings = detect_swings(ohlcv.candles, n=SWING_N)
            structure_events = build_structure(swings)
            state = classify_state(structure_events, window=STRUCTURE_WINDOW)

            recent_labels = [
                e.label for e in
                (structure_events[-STRUCTURE_WINDOW:] if len(structure_events) >= STRUCTURE_WINDOW
                 else structure_events)
            ]

            results.append(MarketStateResult(
                symbol=symbol,
                timeframe=timeframe,
                state=state,
                source=ohlcv.source,
                is_fallback=ohlcv.is_fallback,
                recent_structure=recent_labels,
                swing_count=len(swings),
                error=None,
            ))

    print_report(results)


if __name__ == "__main__":
    run()
