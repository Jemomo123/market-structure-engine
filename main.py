"""
Market State Detector — Phase 1 (single-file build)

Everything lives in this one file on purpose — it removes any risk of
folder/import mistakes when deploying from a phone.

Pipeline: OHLCV -> Swing Detection -> Market Structure -> Market State -> Report

Run:
    pip install -r requirements.txt
    python main.py
"""

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional

import ccxt

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("market_state_detector")


# ============================================================================
# CONFIG
# ============================================================================

WATCHLIST = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "DOGE", "SHIB", "AVAX", "LINK", "SUI",
]

TIMEFRAMES = ["5m", "15m", "1h"]

EXCHANGE_PRIORITY = ["OKX", "MEXC"]
QUOTE_CURRENCY = "USDT"
CANDLE_FETCH_LIMIT = 300
MIN_VALID_CANDLES = 100

SWING_N = 2

# Timeframe -> duration in milliseconds. Used to determine whether the
# most recent fetched candle has actually closed yet.
TIMEFRAME_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}

# How often to rerun the full detection pass, in seconds. 5m is the
# shortest tracked timeframe, so refreshing much faster than that gains
# little and risks exchange rate limits.
REFRESH_INTERVAL_SECONDS = int(os.environ.get("REFRESH_INTERVAL_SECONDS", 300))

# Render (and similar platforms) expect a Web Service to have something
# listening on a port, or the deploy is eventually flagged unhealthy even
# though the detection loop itself is running fine.
PORT = int(os.environ.get("PORT", 10000))

# Holds the most recent report text so the health server can serve it.
_latest_report_lock = threading.Lock()
_latest_report_text = "No report generated yet."


# ============================================================================
# MODELS
# ============================================================================

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
    INSUFFICIENT = "INSUFFICIENT"


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
    # Candle-cleaning pipeline diagnostics
    raw_candle_count: int = 0
    duplicates_removed: int = 0
    forming_candle_removed: bool = False
    closed_candle_count: int = 0


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
    # Pipeline / classification diagnostics
    raw_candle_count: int = 0
    duplicates_removed: int = 0
    forming_candle_removed: bool = False
    closed_candle_count: int = 0
    swing_high_count: int = 0
    swing_low_count: int = 0
    structure_event_count: int = 0
    classification_reason: str = ""


# ============================================================================
# DATA PROVIDERS
# ============================================================================

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


def normalize_and_clean_candles(raw_candles: List[Candle], timeframe: str):
    """
    fetch -> NORMALIZE -> SORT -> DEDUPLICATE -> REMOVE FORMING CANDLE

    Returns (cleaned_candles, diagnostics_dict). Validation of the cleaned
    result happens separately in _validate_candles, so this function's job
    is strictly cleaning/normalization, not accept/reject decisions.
    """
    diag = {
        "raw_candle_count": len(raw_candles),
        "duplicates_removed": 0,
        "forming_candle_removed": False,
        "closed_candle_count": 0,
    }

    if not raw_candles:
        return [], diag

    # NORMALIZE: coerce timestamp to int ms, OHLCV values to float. This
    # also naturally surfaces malformed rows (raises) rather than silently
    # passing bad types downstream.
    normalized = [
        Candle(
            timestamp=int(c.timestamp),
            open=float(c.open),
            high=float(c.high),
            low=float(c.low),
            close=float(c.close),
            volume=float(c.volume),
        )
        for c in raw_candles
    ]

    # SORT CHRONOLOGICALLY: exchanges are expected to return ascending
    # order already, but this must not be assumed.
    normalized.sort(key=lambda c: c.timestamp)

    # DEDUPLICATE: deterministic — keep the first occurrence of each
    # timestamp after sorting, drop any repeats.
    seen_timestamps = set()
    deduped: List[Candle] = []
    for c in normalized:
        if c.timestamp in seen_timestamps:
            continue
        seen_timestamps.add(c.timestamp)
        deduped.append(c)
    diag["duplicates_removed"] = len(normalized) - len(deduped)

    # REMOVE CURRENTLY FORMING CANDLE: a candle is still forming if its
    # close time (open time + timeframe duration) is in the future
    # relative to now (UTC). Only remove it when it is ACTUALLY still
    # forming — never blindly drop the last row.
    forming_removed = False
    if deduped:
        tf_ms = TIMEFRAME_MS.get(timeframe)
        if tf_ms is not None:
            now_ms = int(time.time() * 1000)
            last_candle = deduped[-1]
            candle_close_time_ms = last_candle.timestamp + tf_ms
            if candle_close_time_ms > now_ms:
                deduped = deduped[:-1]
                forming_removed = True
    diag["forming_candle_removed"] = forming_removed
    diag["closed_candle_count"] = len(deduped)

    return deduped, diag


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
                raw_candles = provider.fetch_ohlcv(base_asset, timeframe, CANDLE_FETCH_LIMIT)
            except ProviderError as exc:
                logger.warning(str(exc))
                errors.append(f"{provider.name}: {exc}")
                continue

            # fetch -> NORMALIZE -> SORT -> DEDUPLICATE -> REMOVE FORMING CANDLE
            cleaned_candles, clean_diag = normalize_and_clean_candles(raw_candles, timeframe)

            # VALIDATE the cleaned (closed-only) series, not the raw fetch.
            if not _validate_candles(cleaned_candles):
                msg = (
                    f"{provider.name} returned invalid/insufficient OHLCV for {base_asset} {timeframe} "
                    f"(raw={clean_diag['raw_candle_count']}, "
                    f"dupes_removed={clean_diag['duplicates_removed']}, "
                    f"forming_removed={clean_diag['forming_candle_removed']}, "
                    f"closed={clean_diag['closed_candle_count']})"
                )
                logger.warning(msg)
                errors.append(msg)
                continue

            return OHLCVResult(
                symbol=base_asset,
                timeframe=timeframe,
                candles=cleaned_candles,
                source=provider.name,
                is_fallback=is_fallback,
                error=None,
                raw_candle_count=clean_diag["raw_candle_count"],
                duplicates_removed=clean_diag["duplicates_removed"],
                forming_candle_removed=clean_diag["forming_candle_removed"],
                closed_candle_count=clean_diag["closed_candle_count"],
            )

        return OHLCVResult(
            symbol=base_asset,
            timeframe=timeframe,
            candles=[],
            source=None,
            is_fallback=False,
            error="; ".join(errors) if errors else "No providers available",
        )


# ============================================================================
# CORE — SWING DETECTION
# ============================================================================

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


# ============================================================================
# CORE — MARKET STRUCTURE
# ============================================================================

def build_structure(swings: List[SwingPoint]) -> List[StructureEvent]:
    """
    Compares each swing to the previous swing of the same type.

    An EQUAL price (e.g. price retests the exact prior swing high/low
    before continuing) is neither a genuine higher/lower high nor a
    genuine higher/lower low — it's a tie. Forcing a tie into HH/HL/LH/LL
    via a strict > or < comparison would fabricate a directional signal
    that didn't actually happen (e.g. a bullish trend pausing at the same
    resistance level would get mislabeled LH and could falsely flip the
    state away from BULLISH). Ties are therefore not emitted as a
    structure event — the last_high/last_low reference still advances,
    but no HH/HL/LH/LL label is produced for that swing.
    """
    events: List[StructureEvent] = []

    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None

    for swing in swings:
        if swing.swing_type == SwingType.HIGH:
            if last_high is not None and swing.price != last_high.price:
                label = StructureLabel.HH if swing.price > last_high.price else StructureLabel.LH
                events.append(StructureEvent(swing=swing, label=label))
            last_high = swing
        else:
            if last_low is not None and swing.price != last_low.price:
                label = StructureLabel.HL if swing.price > last_low.price else StructureLabel.LL
                events.append(StructureEvent(swing=swing, label=label))
            last_low = swing

    events.sort(key=lambda e: e.swing.index)
    return events


# ============================================================================
# CORE — STATE CLASSIFICATION
# ============================================================================
#
# Redesigned per Phase 1 classifier correction: RANGING is no longer a
# default fallback. Every state (BULLISH, BEARISH, TRANSITION, RANGING)
# must be positively supported by evidence. When evidence is insufficient
# for any of them, the result is INSUFFICIENT — never RANGING by default.

BULLISH_LABELS = {StructureLabel.HH, StructureLabel.HL}
BEARISH_LABELS = {StructureLabel.LH, StructureLabel.LL}

# How many recent confirmed events are considered as evidence. Wider than
# the old fixed window so a single counter-swing doesn't consume the
# entire evaluation window.
EVIDENCE_WINDOW = 6

# Minimum same-direction events required to call a trend established.
MIN_TREND_EVIDENCE = 3

# Maximum opposing-direction events tolerated inside an otherwise-valid
# trend before it stops qualifying as BULLISH/BEARISH outright.
MAX_TOLERATED_COUNTER = 1

# Minimum pure same-direction events required BEFORE a disruption for
# that disruption to count as an established trend breaking (TRANSITION).
MIN_TRANSITION_PRIOR_EVIDENCE = 2

# Minimum events required on EACH side to positively declare genuine
# two-sided/choppy behavior (RANGING must be evidenced, not assumed).
MIN_RANGING_EACH_SIDE = 2

# Maximum allowed difference between bull_count and bear_count for a
# window to qualify as genuinely balanced/two-sided RANGING. A skewed
# split (e.g. 4-2) is directional evidence that fell short of the
# BULLISH/BEARISH threshold, not genuine chop, and must NOT be labeled
# RANGING just because both sides cleared MIN_RANGING_EACH_SIDE.
MAX_RANGING_IMBALANCE = 1

# Below this many total confirmed events, there simply isn't enough
# data to classify anything — INSUFFICIENT, not RANGING.
MIN_EVENTS_FOR_ANY_CALL = 3


def _is_pure_bullish(labels: List[StructureLabel]) -> bool:
    """All labels are HH/HL, AND both types are represented."""
    return (
        all(l in BULLISH_LABELS for l in labels)
        and StructureLabel.HH in labels
        and StructureLabel.HL in labels
    )


def _is_pure_bearish(labels: List[StructureLabel]) -> bool:
    return (
        all(l in BEARISH_LABELS for l in labels)
        and StructureLabel.LH in labels
        and StructureLabel.LL in labels
    )


def _find_transition_split(labels: List[StructureLabel]) -> Optional[str]:
    """
    Searches for a split point where an established pure trend
    (>= MIN_TRANSITION_PRIOR_EVIDENCE events, both labels of that
    direction present) is followed by a tail made ENTIRELY of the
    opposite direction's labels (>= 1 event). Tries splits from the
    earliest valid point forward, so the largest possible "established"
    run is checked first. Not restricted to one fixed pattern shape.

    Returns "bullish" or "bearish" (the direction that was established
    and then broke) or None if no such split exists.
    """
    n = len(labels)
    for split in range(MIN_TRANSITION_PRIOR_EVIDENCE, n):
        prior = labels[:split]
        tail = labels[split:]

        if not tail:
            continue

        if _is_pure_bullish(prior) and all(l in BEARISH_LABELS for l in tail):
            return "bullish"
        if _is_pure_bearish(prior) and all(l in BULLISH_LABELS for l in tail):
            return "bearish"

    return None


def classify_state_detailed(
    structure_events: List[StructureEvent], window: int = EVIDENCE_WINDOW
):
    """
    Returns (MarketState, reason_string).

    Correction applied: INSUFFICIENT is reserved EXCLUSIVELY for genuine
    data scarcity (Step 1). Once there is enough confirmed structure to
    evaluate (>= MIN_EVENTS_FOR_ANY_CALL), the classifier ALWAYS resolves
    to a positive state — BULLISH, BEARISH, TRANSITION, or RANGING.
    RANGING is not a "nothing else matched" dump: it is evidence-based,
    with the reason string distinguishing genuinely balanced two-sided
    structure from abundant-but-indeterminate structure, but both cases
    are legitimately RANGING (real confirmed data, no clean directional
    or transition signal) rather than a data-quality problem.
    """
    # Step 1 — the ONLY path to INSUFFICIENT: genuine data scarcity.
    if len(structure_events) < MIN_EVENTS_FOR_ANY_CALL:
        return (
            MarketState.INSUFFICIENT,
            f"only {len(structure_events)} confirmed structure event(s); "
            f"need at least {MIN_EVENTS_FOR_ANY_CALL} to classify",
        )

    recent = structure_events[-window:] if len(structure_events) >= window else structure_events
    labels = [e.label for e in recent]

    bull_count = sum(1 for l in labels if l in BULLISH_LABELS)
    bear_count = sum(1 for l in labels if l in BEARISH_LABELS)
    has_both_bull_labels = StructureLabel.HH in labels and StructureLabel.HL in labels
    has_both_bear_labels = StructureLabel.LH in labels and StructureLabel.LL in labels
    last_label = labels[-1]

    # Step 2 — BULLISH: sufficient evidence, tolerable counter-evidence,
    # and the most recent event still belongs to the bullish direction
    # (i.e. the trend is currently intact, not actively breaking now).
    if (
        has_both_bull_labels
        and bull_count >= MIN_TREND_EVIDENCE
        and bear_count <= MAX_TOLERATED_COUNTER
        and last_label in BULLISH_LABELS
    ):
        return (
            MarketState.BULLISH,
            f"established bullish structure (HH/HL evidence={bull_count}, "
            f"counter-evidence={bear_count}, last event={last_label.value})",
        )

    # Step 3 — BEARISH: mirror of step 2.
    if (
        has_both_bear_labels
        and bear_count >= MIN_TREND_EVIDENCE
        and bull_count <= MAX_TOLERATED_COUNTER
        and last_label in BEARISH_LABELS
    ):
        return (
            MarketState.BEARISH,
            f"established bearish structure (LH/LL evidence={bear_count}, "
            f"counter-evidence={bull_count}, last event={last_label.value})",
        )

    # Step 4 — TRANSITION: established trend followed by an opposite-
    # direction tail, at any valid split point (not one fixed pattern).
    broken_direction = _find_transition_split(labels)
    if broken_direction is not None:
        return (
            MarketState.TRANSITION,
            f"established {broken_direction} trend broken by a meaningful "
            f"opposite-direction tail",
        )

    # Step 5 — RANGING: the evidence-based catch-all for every remaining
    # case that reaches this point. By construction, len(structure_events)
    # >= MIN_EVENTS_FOR_ANY_CALL is already guaranteed (Step 1 passed), so
    # this is NEVER a data-scarcity situation — it is abundant confirmed
    # structure that simply does not show a clean directional or
    # transition pattern. That is what RANGING means; it must not be
    # reported as INSUFFICIENT just because it failed the narrow
    # BULLISH/BEARISH/TRANSITION rules above.
    if (
        min(bull_count, bear_count) >= MIN_RANGING_EACH_SIDE
        and abs(bull_count - bear_count) <= MAX_RANGING_IMBALANCE
    ):
        return (
            MarketState.RANGING,
            f"genuine balanced two-sided structure (bull={bull_count}, bear={bear_count})",
        )

    return (
        MarketState.RANGING,
        f"mixed/indeterminate structure with sufficient confirmed evidence "
        f"(bull={bull_count}, bear={bear_count}); no clean directional or "
        f"transition pattern met",
    )


def classify_state(structure_events: List[StructureEvent], window: int = EVIDENCE_WINDOW) -> MarketState:
    state, _reason = classify_state_detailed(structure_events, window)
    return state


# ============================================================================
# OUTPUT
# ============================================================================

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


def build_report_text(results: List[MarketStateResult]) -> str:
    header = f"{'COIN':<6} {'TF':<5} {'STATE':<12} {'SOURCE':<18} {'STRUCTURE (recent -> latest)'}"
    lines = [header, "-" * len(header)]

    for r in results:
        structure_str = (
            " -> ".join(l.value for l in r.recent_structure)
            if r.recent_structure else "(insufficient swings)"
        )
        if r.error:
            lines.append(f"{r.symbol:<6} {r.timeframe:<5} {'NO DATA':<12} {'-':<18} {r.error}")
            continue

        lines.append(
            f"{r.symbol:<6} {r.timeframe:<5} {format_state(r):<12} "
            f"{format_source(r):<18} {structure_str}"
        )

    lines.append("")
    lines.append(build_diagnostics_text(results))

    return "\n".join(lines)


def build_diagnostics_text(results: List[MarketStateResult]) -> str:
    """
    Per-row pipeline/classification diagnostics, as required:
    provider, raw candles, duplicates removed, forming candle detected/
    removed, closed candle count, swing high/low counts, total swings,
    structure-event count, final state, and the classification reason.
    """
    header = (
        f"{'COIN':<6} {'TF':<5} {'PROV':<6} {'RAW':<5} {'DUPES':<6} "
        f"{'FORMING':<8} {'CLOSED':<7} {'SWH':<4} {'SWL':<4} {'SWINGS':<7} "
        f"{'EVENTS':<7} {'STATE':<12} REASON"
    )
    lines = ["DIAGNOSTICS", header, "-" * len(header)]

    for r in results:
        if r.error:
            lines.append(f"{r.symbol:<6} {r.timeframe:<5} NO DATA — {r.error}")
            continue

        lines.append(
            f"{r.symbol:<6} {r.timeframe:<5} {(r.source or '-'):<6} "
            f"{r.raw_candle_count:<5} {r.duplicates_removed:<6} "
            f"{str(r.forming_candle_removed):<8} {r.closed_candle_count:<7} "
            f"{r.swing_high_count:<4} {r.swing_low_count:<4} {r.swing_count:<7} "
            f"{r.structure_event_count:<7} {format_state(r):<12} {r.classification_reason}"
        )

    return "\n".join(lines)


def print_report(results: List[MarketStateResult]) -> None:
    print(build_report_text(results))


# ============================================================================
# MAIN
# ============================================================================

def run_once(router: DataRouter) -> List[MarketStateResult]:
    """Runs a single full detection pass over the whole watchlist and
    returns the results. Pure — does no printing or I/O of its own."""
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
            state, reason = classify_state_detailed(structure_events, window=EVIDENCE_WINDOW)

            recent_labels = [
                e.label for e in
                (structure_events[-EVIDENCE_WINDOW:] if len(structure_events) >= EVIDENCE_WINDOW
                 else structure_events)
            ]

            swing_high_count = sum(1 for s in swings if s.swing_type == SwingType.HIGH)
            swing_low_count = sum(1 for s in swings if s.swing_type == SwingType.LOW)

            results.append(MarketStateResult(
                symbol=symbol,
                timeframe=timeframe,
                state=state,
                source=ohlcv.source,
                is_fallback=ohlcv.is_fallback,
                recent_structure=recent_labels,
                swing_count=len(swings),
                error=None,
                raw_candle_count=ohlcv.raw_candle_count,
                duplicates_removed=ohlcv.duplicates_removed,
                forming_candle_removed=ohlcv.forming_candle_removed,
                closed_candle_count=ohlcv.closed_candle_count,
                swing_high_count=swing_high_count,
                swing_low_count=swing_low_count,
                structure_event_count=len(structure_events),
                classification_reason=reason,
            ))

    return results


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP server so the platform sees something listening on
    PORT. GET / (or anything) returns the most recent report as plain
    text. This has nothing to do with the detection logic itself."""

    def do_GET(self):  # noqa: N802 (stdlib method name)
        with _latest_report_lock:
            body = _latest_report_text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        # Silence default request logging; the detection loop's own
        # logging is what matters here.
        pass


def _start_health_server() -> None:
    server = HTTPServer(("0.0.0.0", PORT), _HealthHandler)
    logger.info(f"Health/report server listening on port {PORT}")
    server.serve_forever()


def run_forever() -> None:
    global _latest_report_text

    router = DataRouter()

    # The HTTP server runs in a background thread so it can respond to
    # health checks / report requests at any time, independent of where
    # the detection loop currently is in its cycle.
    health_thread = threading.Thread(target=_start_health_server, daemon=True)
    health_thread.start()

    while True:
        started_at = time.time()
        logger.info("Starting detection pass...")

        try:
            results = run_once(router)
            report_text = build_report_text(results)
            with _latest_report_lock:
                _latest_report_text = report_text
            print(report_text)
        except Exception:
            # A single bad pass should never kill the whole service —
            # log it and try again next cycle rather than crashing.
            logger.exception("Detection pass failed; will retry next cycle.")

        elapsed = time.time() - started_at
        sleep_for = max(0.0, REFRESH_INTERVAL_SECONDS - elapsed)
        logger.info(f"Pass complete in {elapsed:.1f}s. Sleeping {sleep_for:.1f}s.")
        time.sleep(sleep_for)


if __name__ == "__main__":
    run_forever()
