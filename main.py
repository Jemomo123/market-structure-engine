"""
Market State Detector — Phase 1 (single-file build)
Pipeline: OHLCV -> Swing Detection -> Market Structure -> Market State -> Report
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import ccxt

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("market_state_detector")

WATCHLIST = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "DOGE", "TON", "AVAX", "LINK", "SUI",
]

TIMEFRAMES = ["5m", "15m", "1h"]
EXCHANGE_PRIORITY = ["OKX", "MEXC"]
QUOTE_CURRENCY = "USDT"
CANDLE_FETCH_LIMIT = 300
MIN_VALID_CANDLES = 100
SWING_N = 2
STRUCTURE_WINDOW = 4


class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class StructureLabel(str, Enum):
    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"


class MarketState(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"
    TRANSITION = "TRANSITION"


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OHLCVResult:
    symbol: str
    timeframe: str
    candles: List[Candle]
    source: Optional[str]
    is_fallback: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class SwingPoint:
    index: int
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
    state: Optional[MarketState]
    source: Optional[str]
    is_fallback: bool
    recent_structure: List[StructureLabel]
    swing_count: int
    error: Optional[str] = None


class ProviderError(Exception):
    pass


class BaseProvider(ABC):
    name: str = "BASE"

    @abstractmethod
    def to_exchange_symbol(self, base_asset: str, quote: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def fetch_ohlcv(self, base_asset: str, timeframe: str, limit: int) -> List[Candle]:
        raise NotImplementedError


class OKXProvider(BaseProvider):
    name = "OKX"

    def __init__(self):
        self._client = ccxt.okx({"enableRateLimit": True})

    def to_exchange_symbol(self, base_asset: str, quote: str) -> str:
        return f"{base_asset}/{quote}"

    def fetch_ohlcv(self, base_asset: str, timeframe: str, limit: int) -> List[Candle]:
        symbol = self.to_exchange_symbol(base_asset, QUOTE_CURRENCY)
        try:
            raw = self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:
            raise ProviderError(f"OKX fetch failed for {symbol} {timeframe}: {exc}") from exc
        if not raw:
            raise ProviderError(f"OKX returned empty OHLCV for {symbol} {timeframe}")
        return [
            Candle(timestamp=row[0], open=row[1], high=row[2], low=row[3], close=row[4], volume=row[5])
            for row in raw
        ]


class MEXCProvider(BaseProvider):
    name = "MEXC"

    def __init__(self):
        self._client = ccxt.mexc({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    def to_exchange_symbol(self, base_asset: str, quote: str) -> str:
        return f"{base_asset}/{quote}:{quote}"

    def fetch_ohlcv(self, base_asset: str, timeframe: str, limit: int) -> List[Candle]:
        symbol = self.to_exchange_symbol(base_asset, QUOTE_CURRENCY)
        try:
            raw = self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:
            raise ProviderError(f"MEXC fetch failed for {symbol} {timeframe}: {exc}") from exc
        if not raw:
            raise ProviderError(f"MEXC returned empty OHLCV for {symbol} {timeframe}")
        return [
            Candle(timestamp=row[0], open=row[1], high=row[2], low=row[3], close=row[4], volume=row[5])
            for row in raw
        ]


def _validate_candles(candles: List[Candle]) -> bool:
    if not candles:
        return False
    if len(candles) < MIN_VALID_CANDLES:
        return False
    timestamps = [c.timestamp for c in candles]
    if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        return False
    for c in candles:
        if any(v is None for v in (c.open, c.high, c.low, c.close, c.volume)):
            return False
        if c.high <= 0 or c.low <= 0 or c.open <= 0 or c.close <= 0:
            return False
        if c.low > c.high:
            return False
        if c.volume < 0:
            return False
    return True


class DataRouter:
    def __init__(self):
        self._providers: List[BaseProvider] = [OKXProvider(), MEXCProvider()]

    def get_ohlcv(self, base_asset: str, timeframe: str) -> OHLCVResult:
        errors = []
        for i, provider in enumerate(self._providers):
            is_fallback = i > 0
            try:
                candles = provider.fetch_ohlcv(base_asset, timeframe, CANDLE_FETCH_LIMIT)
            except ProviderError as exc:
                logger.warning(str(exc))
                errors.append(f"{provider.name}: {exc}")
                continue
            if not _validate_candles(candles):
                msg = f"{provider.name} returned invalid/insufficient OHLCV for {base_asset} {timeframe}"
                logger.warning(msg)
                errors.append(msg)
                continue
            return OHLCVResult(
                symbol=base_asset, timeframe=timeframe, candles=candles,
                source=provider.name, is_fallback=is_fallback, error=None,
            )
        return OHLCVResult(
            symbol=base_asset, timeframe=timeframe, candles=[], source=None,
            is_fallback=False, error="; ".join(errors) if errors else "No providers available",
        )


def detect_swings(candles: List[Candle], n: int) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    if len(candles) < (2 * n + 1):
        return swings
    for i in range(n, len(candles) - n):
        window = candles[i - n:i + n + 1]
        pivot = candles[i]
        is_swing_high = all(pivot.high > c.high for c in window if c is not pivot)
        if is_swing_high:
            swings.append(SwingPoint(index=i, timestamp=pivot.timestamp, price=pivot.high, swing_type=SwingType.HIGH))
            continue
        is_swing_low = all(pivot.low < c.low for c in window if c is not pivot)
        if is_swing_low:
            swings.append(SwingPoint(index=i, timestamp=pivot.timestamp, price=pivot.low, swing_type=SwingType.LOW))
    return swings


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
        else:
            if last_low is not None:
                label = StructureLabel.HL if swing.price > last_low.price else StructureLabel.LL
                events.append(StructureEvent(swing=swing, label=label))
            last_low = swing
    events.sort(key=lambda e: e.swing.index)
    return events


BULLISH_LABELS = {StructureLabel.HH, StructureLabel.HL}
BEARISH_LABELS = {StructureLabel.LH, StructureLabel.LL}
MIN_EVENTS_FOR_DIRECTIONAL_CALL = 3


def _is_clean_bullish(labels: List[StructureLabel]) -> bool:
    return all(l in BULLISH_LABELS for l in labels) and StructureLabel.HH in labels and StructureLabel.HL in labels


def _is_clean_bearish(labels: List[StructureLabel]) -> bool:
    return all(l in BEARISH_LABELS for l in labels) and StructureLabel.LH in labels and StructureLabel.LL in labels


def classify_state(structure_events: List[StructureEvent], window: int) -> MarketState:
    if len(structure_events) < MIN_EVENTS_FOR_DIRECTIONAL_CALL:
        return MarketState.RANGING
    recent = structure_events[-window:] if len(structure_events) >= window else structure_events
    labels = [e.label for e in recent]
    if _is_clean_bullish(labels):
        return MarketState.BULLISH
    if _is_clean_bearish(labels):
        return MarketState.BEARISH
    if len(labels) >= 3:
        prior, tail = labels[:-1], labels[-1]
        prior_was_bullish = all(l in BULLISH_LABELS for l in prior) and StructureLabel.HH in prior and StructureLabel.HL in prior
        prior_was_bearish = all(l in BEARISH_LABELS for l in prior) and StructureLabel.LH in prior and StructureLabel.LL in prior
        if prior_was_bullish and tail in BEARISH_LABELS:
            return MarketState.TRANSITION
        if prior_was_bearish and tail in BULLISH_LABELS:
            return MarketState.TRANSITION
    return MarketState.RANGING


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
        structure_str = " -> ".join(l.value for l in r.recent_structure) if r.recent_structure else "(insufficient swings)"
        if r.error:
            print(f"{r.symbol:<6} {r.timeframe:<5} {'NO DATA':<12} {'-':<18} {r.error}")
            continue
        print(f"{r.symbol:<6} {r.timeframe:<5} {format_state(r):<12} {format_source(r):<18} {structure_str}")


def run() -> None:
    router = DataRouter()
    results = []
    for symbol in WATCHLIST:
        for timeframe in TIMEFRAMES:
            ohlcv = router.get_ohlcv(symbol, timeframe)
            if ohlcv.error:
                results.append(MarketStateResult(
                    symbol=symbol, timeframe=timeframe, state=None, source=None,
                    is_fallback=False, recent_structure=[], swing_count=0, error=ohlcv.error,
                ))
                continue
            swings = detect_swings(ohlcv.candles, n=SWING_N)
            structure_events = build_structure(swings)
            state = classify_state(structure_events, window=STRUCTURE_WINDOW)
            recent_labels = [
                e.label for e in
                (structure_events[-STRUCTURE_WINDOW:] if len(structure_events) >= STRUCTURE_WINDOW else structure_events)
            ]
            results.append(MarketStateResult(
                symbol=symbol, timeframe=timeframe, state=state, source=ohlcv.source,
                is_fallback=ohlcv.is_fallback, recent_structure=recent_labels,
                swing_count=len(swings), error=None,
            ))
    print_report(results)


if __name__ == "__main__":
    run()
