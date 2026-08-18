"""
Shared data structures passed between layers.

Keeping these as plain dataclasses (no exchange or indicator knowledge)
is what lets core/ stay completely independent of data/.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class StructureLabel(str, Enum):
    HH = "HH"   # Higher High
    HL = "HL"   # Higher Low
    LH = "LH"   # Lower High
    LL = "LL"   # Lower Low


class MarketState(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"
    TRANSITION = "TRANSITION"


@dataclass(frozen=True)
class Candle:
    timestamp: int   # epoch ms
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OHLCVResult:
    """Raw candle data plus provenance. This is the boundary object between
    data/ and core/ — core/ never sees an exchange name or API response,
    only this."""
    symbol: str
    timeframe: str
    candles: List[Candle]
    source: Optional[str]        # "OKX", "MEXC", or None if all failed
    is_fallback: bool = False
    error: Optional[str] = None  # set when no provider succeeded


@dataclass(frozen=True)
class SwingPoint:
    index: int            # index into the candle list this swing was confirmed from
    timestamp: int
    price: float
    swing_type: SwingType


@dataclass(frozen=True)
class StructureEvent:
    swing: SwingPoint
    label: StructureLabel


@dataclass(frozen=True)
class MarketStateResult:
    symbol: str
    timeframe: str
    state: MarketState
    source: Optional[str]
    is_fallback: bool
    recent_structure: List[StructureLabel]   # the evaluation window, oldest -> newest
    swing_count: int
    error: Optional[str] = None              # set if state could not be computed
