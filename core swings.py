"""
Confirmed swing high/low detection.

A swing is only "confirmed" once N candles exist on BOTH sides of it.
This means the most recent N candles can never produce a confirmed swing —
that's intentional: Phase 1 must not use unconfirmed/current pivots.
"""

from typing import List

from models.types import Candle, SwingPoint, SwingType


def detect_swings(candles: List[Candle], n: int) -> List[SwingPoint]:
    """
    A candle at index i is a confirmed swing high if its high is strictly
    greater than the highs of the n candles immediately before AND the n
    candles immediately after it. Swing lows are the mirror on lows.

    Returns swings in chronological order.
    """
    swings: List[SwingPoint] = []

    if len(candles) < (2 * n + 1):
        return swings

    for i in range(n, len(candles) - n):
        window = candles[i - n:i + n + 1]
        pivot = candles[i]

        is_swing_high = all(
            pivot.high > c.high for c in window if c is not pivot
        )
        if is_swing_high:
            swings.append(SwingPoint(
                index=i,
                timestamp=pivot.timestamp,
                price=pivot.high,
                swing_type=SwingType.HIGH,
            ))
            continue  # a candle is treated as either a high or low pivot, not both

        is_swing_low = all(
            pivot.low < c.low for c in window if c is not pivot
        )
        if is_swing_low:
            swings.append(SwingPoint(
                index=i,
                timestamp=pivot.timestamp,
                price=pivot.low,
                swing_type=SwingType.LOW,
            ))

    return swings
