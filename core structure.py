"""
Turns a chronological list of confirmed swing points into a chronological
list of structure events (HH/HL/LH/LL) by comparing each swing to the
PREVIOUS swing of the same type (high-to-high, low-to-low).
"""

from typing import List, Optional

from models.types import StructureEvent, StructureLabel, SwingPoint, SwingType


def build_structure(swings: List[SwingPoint]) -> List[StructureEvent]:
    events: List[StructureEvent] = []

    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None

    for swing in swings:
        if swing.swing_type == SwingType.HIGH:
            if last_high is not None:
                label = StructureLabel.HH if swing.price > last_high.price else StructureLabel.LH
                events.append(StructureEvent(swing=swing, label=label))
            last_high = swing
        else:  # SwingType.LOW
            if last_low is not None:
                label = StructureLabel.HL if swing.price > last_low.price else StructureLabel.LL
                events.append(StructureEvent(swing=swing, label=label))
            last_low = swing

    # Events must remain chronological (by the underlying swing's index),
    # since highs and lows are interleaved when built as above.
    events.sort(key=lambda e: e.swing.index)
    return events
