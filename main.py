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
    # Phase 2A — Range Detection result (SSOT2, independent of SSOT1 state)
    range_result: Optional["RangeResult"] = None
    # Phase 2B — Breakout Readiness (only meaningful when range_result.detected)
    breakout_readiness: Optional["BreakoutReadinessResult"] = None


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


# ============================================================================
# SSOT2 — PHASE 2A: RANGE DETECTION
# ============================================================================
#
# Range Detection is deliberately independent of the SSOT1 market-state
# classifier above. A coin classified RANGING is NOT treated as proof a
# genuine range exists, and a coin classified TRANSITION/BULLISH/BEARISH
# is not excluded from having a valid prior range on record. This section
# consumes SSOT1's already-computed outputs (closed candles + confirmed
# swings) as its only inputs — it does not fetch, normalize, or dedupe
# anything itself, and it never modifies SSOT1's behavior.

# Two confirmed swing prices are considered part of the same boundary
# cluster if they fall within this percentage of each other. Tightened
# from the initial 0.50% baseline after live validation showed 0/30
# ranges detected, all rejected at containment — a 0.50% cluster was
# permitted to span wider than the 0.10% containment buffer could
# accept. Containment buffer is intentionally left unchanged (it exists
# to absorb normal wick/price noise around an established boundary, not
# to compensate for an overly loose boundary cluster).
BOUNDARY_TOLERANCE_PCT = 0.10

# Containment check allows candle highs/lows to exceed the boundary by
# this percentage before it counts as a genuine breach.
CONTAINMENT_BUFFER_PCT = 0.10

# A boundary (upper or lower) needs at least this many qualifying
# confirmed swings in its cluster to count as established.
MIN_BOUNDARY_TESTS = 2

# --- Adaptive width threshold (replaces the old fixed MIN_RANGE_WIDTH_PCT) ---
#
# A single fixed percentage treats every coin the same, which doesn't
# make sense: what counts as "meaningfully separated" should scale with
# how much that specific coin/timeframe normally moves. Instead of one
# flat number, the required minimum width is now derived from that
# coin's own typical candle range (high-low as % of price) over the
# lookback window, then clamped to a floor/ceiling band.

# The required width is this many multiples of the coin's own average
# candle range (as a %). Initial estimate, not tuned.
MIN_RANGE_WIDTH_MULTIPLIER = 2.0

# The adaptive threshold is never allowed to go below/above these
# bounds, regardless of the coin's own volatility.
MIN_RANGE_WIDTH_FLOOR_PCT = 0.30
MIN_RANGE_WIDTH_CEILING_PCT = 0.50

# Minimum number of the timeframe's OWN closed candles the range must
# span (24 on 5m = 24 five-minute candles; 24 on 1h = 24 one-hour
# candles — never converted to a fixed wall-clock duration).
MIN_RANGE_CANDLES = 24

# Compression check: the recent portion of the range's high-low envelope
# must be at least this percentage narrower than the earlier portion's
# envelope to be flagged as compressing. See _detect_compression() for
# the exact calculation. This is a candle-geometry comparison only —
# no indicators are involved.
COMPRESSION_THRESHOLD_PCT = 20.0

# --- Quality corrections (supervisor decision, post-live-review) ---
#
# Live validation showed boundary clustering + test count + containment +
# duration was NOT sufficient: candidates like "LOW -> strong directional
# move -> HIGH" with 2 touches near each extreme passed as "ranges" when
# they were actually one-directional moves with brief consolidation at
# each end, not genuine back-and-forth oscillation. Two new deterministic
# checks address this — see _check_alternation() and the directional-
# dominance calculation in detect_range().

# Minimum number of alternating transitions required among the
# chronologically-ordered qualifying boundary touches (upper/lower). A
# transition is any adjacent pair of touches on OPPOSITE boundaries.
# Default of 3 requires at least 4 touches in strict alternation
# (e.g. LOW -> HIGH -> LOW -> HIGH) — exactly matching MIN_BOUNDARY_TESTS's
# existing minimum of 2 touches per side, just requiring them to actually
# interleave rather than cluster in time (LOW,LOW,HIGH,HIGH fails this).
MIN_BOUNDARY_ALTERNATIONS = 3

# Directional-dominance check: computed as
# net_displacement / total_zigzag_path_length across all confirmed swings
# between the first and last qualifying boundary touch. A ratio near 1.0
# means the price path was essentially one straight run (little real
# back-and-forth); a ratio well below 1.0 means genuine oscillation
# covered much more ground than the net start-to-end move. A candidate
# is rejected if its ratio is >= this threshold. This is pure swing/price
# geometry — no indicators.
MAX_DIRECTIONAL_DOMINANCE_RATIO = 0.50

# --- Bounded lookback correction (v2 redesign, post 5x live-failure review) ---
#
# Root cause of 5/5 live passes returning 0/30 confirmed ranges: boundary
# candidates were being searched across the ENTIRE fetched history (up to
# 299 candles), and containment was required to hold unbroken all the way
# to the present candle. Proven via synthetic test that this rejects even
# a perfectly genuine historical range purely because price moved on
# afterward. Fix: bound the candidate search to a recent window, and
# classify (rather than blanket-reject) whatever happened after a range's
# contained life ends.

# Only swings within this many most-recent closed candles are eligible to
# form boundary candidates. An ancient touch can never pair with a recent
# one — they're no longer in the same search universe at all. Initial
# estimate (4x MIN_RANGE_CANDLES, giving room for a minimum-length range
# to form AND still be observable afterward) — not empirically tuned.
RANGE_LOOKBACK_CANDLES = 96

# How many candles ago a boundary breach must have occurred to still
# count as "recent" rather than "the market has moved on a while ago".
# Initial estimate (~1/3 of MIN_RANGE_CANDLES) — not empirically tuned.
RECENT_BREAK_CANDLES = 8

# --- Generalized range shapes (v3, Sunday build) ---
#
# Generalizes "boundary" from a fixed price to a straight line through
# 2+ confirmed touches. A flat boundary is just a line with ~zero slope
# — the original flat-range logic becomes one special case of this,
# not a separate code path. Two lines (upper, lower), each independently
# flat/rising/falling, are looked up against a fixed table to name one
# of 6 shapes (flat range, ascending/descending triangle, rising/falling
# channel, symmetrical wedge). The 7th combination (diverging/expanding)
# is explicitly excluded, matching the approved scope.

# How close two slopes' PERCENTAGE difference needs to be to count as
# "roughly parallel" (channel) rather than meaningfully different. Used
# only to distinguish a channel (roughly constant width) from a
# converging/diverging shape. Initial estimate, not tuned.
PARALLEL_SLOPE_TOLERANCE_PCT = 50.0


@dataclass(frozen=True)
class RangeResult:
    detected: bool  # True ONLY when status == "ACTIVE" — preserved for
                     # Phase 2B's existing dependency check.
    reason: str
    # Four-way outcome: "ACTIVE", "RECENTLY_BROKEN", "EXPIRED", "NO_RANGE"
    status: str = "NO_RANGE"
    upper_boundary: Optional[float] = None
    lower_boundary: Optional[float] = None
    width_absolute: Optional[float] = None
    width_percent: Optional[float] = None
    duration_candles: Optional[int] = None
    duration_time: Optional[str] = None
    upper_tests: Optional[int] = None
    lower_tests: Optional[int] = None
    current_price: Optional[float] = None
    current_position_percent: Optional[float] = None
    compression: Optional[bool] = None
    boundary_touch_sequence: Optional[str] = None
    directional_dominance_ratio: Optional[float] = None
    # Raw touch swings, populated only for ACTIVE/RECENTLY_BROKEN/EXPIRED
    # (i.e. whenever a genuine candidate was found), so Phase 2B can reuse
    # the EXACT same boundary-defining touches without recomputing
    # clustering.
    upper_touch_swings: Optional[List[SwingPoint]] = None
    lower_touch_swings: Optional[List[SwingPoint]] = None
    # New in v2: breach diagnostics
    breach_index: Optional[int] = None
    candles_since_breach: Optional[int] = None
    lookback_candles_used: Optional[int] = None
    # v3: generalized shape fields
    shape: str = "NO_RANGE"  # one of: FLAT_RANGE, ASCENDING_TRIANGLE,
                              # DESCENDING_TRIANGLE, RISING_CHANNEL,
                              # FALLING_CHANNEL, SYMMETRICAL_WEDGE, NO_RANGE
    upper_slope_per_candle: Optional[float] = None
    lower_slope_per_candle: Optional[float] = None
    current_width_percent: Optional[float] = None  # width AT PRESENT (vs width_percent, which is width at genesis)
    convergence_candles_ahead: Optional[int] = None  # only for converging shapes


def _cluster_by_tolerance(prices_with_swings, tolerance_pct: float):
    """
    Deterministic 1D clustering: sort ascending, then greedily merge each
    next price into the current cluster if it's within tolerance_pct of
    that cluster's RUNNING AVERAGE (not just the last point added) — this
    keeps the whole cluster tightly bounded around a representative level
    rather than letting it drift arbitrarily wide through chained
    pairwise comparisons. prices_with_swings is a list of
    (price, SwingPoint) tuples. Returns a list of clusters, each a list
    of (price, SwingPoint) tuples.
    """
    if not prices_with_swings:
        return []

    ordered = sorted(prices_with_swings, key=lambda ps: ps[0])
    clusters = [[ordered[0]]]

    for price, swing in ordered[1:]:
        current_cluster = clusters[-1]
        running_average = sum(p for p, _ in current_cluster) / len(current_cluster)
        tolerance = running_average * (tolerance_pct / 100.0)
        if abs(price - running_average) <= tolerance:
            current_cluster.append((price, swing))
        else:
            clusters.append([(price, swing)])

    return clusters


@dataclass(frozen=True)
class _LineFit:
    """A straight line through 2+ chronologically-ordered touches.
    price_at(index) gives the line's interpolated/extrapolated value at
    any candle index."""
    touches: List[SwingPoint]  # chronologically ordered, all members

    @property
    def first(self) -> SwingPoint:
        return self.touches[0]

    @property
    def last(self) -> SwingPoint:
        return self.touches[-1]

    @property
    def slope_per_candle(self) -> float:
        span = self.last.index - self.first.index
        if span == 0:
            return 0.0
        return (self.last.price - self.first.price) / span

    def price_at(self, index: int) -> float:
        return self.first.price + self.slope_per_candle * (index - self.first.index)


def _fit_line_clusters(touches_with_swings, tolerance_pct: float) -> List[_LineFit]:
    """
    Generalizes _cluster_by_tolerance from "cluster by price" to "cluster
    by consistency with a straight line". Touches are processed in
    CHRONOLOGICAL order (not sorted by price). A touch joins the current
    line if its actual price is within tolerance_pct of the line's
    PREDICTED value at that touch's index (predicted via the line's
    current first/last anchor points) — else it starts a new line.

    When all touches in a resulting line happen to sit at nearly the
    same price, the fitted line's slope comes out ~0 — this is how a
    flat boundary emerges as a special case of the same mechanism,
    rather than needing separate logic.

    touches_with_swings: list of (price, SwingPoint) tuples, any order.
    Returns a list of _LineFit, each covering a chronologically
    contiguous run of consistent touches.
    """
    if not touches_with_swings:
        return []

    ordered = sorted((s for _, s in touches_with_swings), key=lambda s: s.index)

    lines: List[List[SwingPoint]] = [[ordered[0]]]

    for swing in ordered[1:]:
        current = lines[-1]
        if len(current) == 1:
            # Second point always joins — a single point has no slope to
            # test consistency against yet.
            current.append(swing)
            continue

        anchor_first = current[0]
        anchor_last = current[-1]
        span = anchor_last.index - anchor_first.index
        slope = (anchor_last.price - anchor_first.price) / span if span != 0 else 0.0
        predicted = anchor_first.price + slope * (swing.index - anchor_first.index)

        tolerance = predicted * (tolerance_pct / 100.0) if predicted > 0 else 0.0
        if abs(swing.price - predicted) <= abs(tolerance):
            current.append(swing)
        else:
            lines.append([swing])

    return [_LineFit(touches=line) for line in lines]


def _line_total_drift_pct(line: _LineFit) -> float:
    """Total price change from the line's first to last touch, as a
    percentage of the first touch's price. Used to classify a line as
    flat vs sloped using the SAME tolerance already used for clustering
    — a flat line is simply one whose own total drift stays within that
    tolerance."""
    if line.first.price == 0:
        return 0.0
    return abs(line.last.price - line.first.price) / line.first.price * 100.0


def _compute_adaptive_min_width_pct(candles: List[Candle], lookback_start_index: int) -> float:
    """
    Derives the required minimum boundary separation from this specific
    coin/timeframe's own typical candle range, rather than one fixed
    number for every coin. Pure candle geometry (high-low as % of
    price), no indicators.

    avg_candle_range_pct = average of (high-low)/close*100 across the
    lookback window. Required width = MIN_RANGE_WIDTH_MULTIPLIER times
    that, clamped to [MIN_RANGE_WIDTH_FLOOR_PCT, MIN_RANGE_WIDTH_CEILING_PCT].
    """
    span = candles[lookback_start_index:]
    if not span:
        return MIN_RANGE_WIDTH_FLOOR_PCT

    ranges_pct = [
        ((c.high - c.low) / c.close * 100.0)
        for c in span if c.close > 0
    ]
    if not ranges_pct:
        return MIN_RANGE_WIDTH_FLOOR_PCT

    avg_candle_range_pct = sum(ranges_pct) / len(ranges_pct)
    required = avg_candle_range_pct * MIN_RANGE_WIDTH_MULTIPLIER
    return max(MIN_RANGE_WIDTH_FLOOR_PCT, min(MIN_RANGE_WIDTH_CEILING_PCT, required))


def _detect_compression(candles: List[Candle], start_index: int, end_index: Optional[int] = None) -> Optional[bool]:
    """
    Splits candles[start_index:end_index] (end_index exclusive; defaults
    to the end of the list) into an earlier half and a recent half (by
    candle count). For each half, the "effective envelope" is
    max(high) - min(low) across that half's candles — pure candle
    geometry, no indicators. Compression is flagged True only when the
    recent envelope is at least COMPRESSION_THRESHOLD_PCT narrower than
    the earlier envelope. Returns None when there isn't enough data
    (fewer than 2 candles in either half) to make the comparison
    meaningful.

    v2 note: end_index lets compression be scoped to a range's own
    CONTAINED lifetime (genesis through its breach point, or through
    present if still active) rather than always running to the very end
    of the fetched candle list — compression should describe the range's
    own life, not whatever happened to price after it broke.
    """
    span = candles[start_index:end_index] if end_index is not None else candles[start_index:]
    if len(span) < 4:
        return None

    midpoint = len(span) // 2
    earlier_half = span[:midpoint]
    recent_half = span[midpoint:]

    if len(earlier_half) < 2 or len(recent_half) < 2:
        return None

    earlier_envelope = max(c.high for c in earlier_half) - min(c.low for c in earlier_half)
    recent_envelope = max(c.high for c in recent_half) - min(c.low for c in recent_half)

    if earlier_envelope <= 0:
        return None

    narrowing_pct = (1 - (recent_envelope / earlier_envelope)) * 100.0
    return narrowing_pct >= COMPRESSION_THRESHOLD_PCT


def _check_alternation(upper_cluster, lower_cluster):
    """
    Combines the qualifying upper-boundary and lower-boundary touches,
    orders them chronologically by swing index, and checks whether they
    alternate between the two boundaries.

    Returns (fully_alternating: bool, transition_count: int, sequence_str: str).

    A transition is any adjacent pair of touches on OPPOSITE boundaries.
    fully_alternating is True only when EVERY adjacent pair differs (no
    two consecutive touches on the same boundary) — e.g. L,H,L,H passes;
    L,L,H,H fails (grouped, not oscillating) even though it has 2 tests
    on each side.
    """
    touches = [(s.index, "U") for _, s in upper_cluster] + [(s.index, "L") for _, s in lower_cluster]
    touches.sort(key=lambda t: t[0])
    labels = [side for _, side in touches]

    transition_count = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    fully_alternating = all(labels[i] != labels[i - 1] for i in range(1, len(labels))) if len(labels) > 1 else False
    sequence_str = "-".join(labels)

    return fully_alternating, transition_count, sequence_str


def detect_range(candles: List[Candle], swings: List[SwingPoint], timeframe: str) -> RangeResult:
    """
    Phase 2A Range Detection v3 — generalized shapes via line-fitting.

    Consumes SSOT1's already-computed closed candles and confirmed swings
    only — no fetching, no re-validation of candle data, no look-ahead.

    v3: a boundary is now a straight line through 2+ confirmed touches
    (a flat boundary is just a line with ~zero slope — the original flat
    range becomes one special case of this, not separate logic). Each
    side is independently classified flat/rising/falling; the pair is
    looked up against a fixed table to name one of 6 supported shapes.
    The 7th combination (diverging/expanding) is explicitly excluded.
    Bounded lookback and breach-classification (ACTIVE/RECENTLY_BROKEN/
    EXPIRED/NO_RANGE) from v2 are preserved unchanged in mechanism, now
    operating against each candle's LINE-implied boundary value at that
    candle's own index instead of one fixed number.
    """
    total_candles = len(candles)
    lookback_start_index = max(0, total_candles - RANGE_LOOKBACK_CANDLES)
    lookback_candles_used = total_candles - lookback_start_index

    lookback_swings = [s for s in swings if s.index >= lookback_start_index]

    swing_highs = [(s.price, s) for s in lookback_swings if s.swing_type == SwingType.HIGH]
    swing_lows = [(s.price, s) for s in lookback_swings if s.swing_type == SwingType.LOW]

    if not swing_highs or not swing_lows:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=f"no coherent boundary clusters within lookback (last {lookback_candles_used} candles)",
            lookback_candles_used=lookback_candles_used,
        )

    upper_lines = _fit_line_clusters(swing_highs, BOUNDARY_TOLERANCE_PCT)
    lower_lines = _fit_line_clusters(swing_lows, BOUNDARY_TOLERANCE_PCT)

    qualifying_upper_lines = [l for l in upper_lines if len(l.touches) >= MIN_BOUNDARY_TESTS]
    qualifying_lower_lines = [l for l in lower_lines if len(l.touches) >= MIN_BOUNDARY_TESTS]

    if not qualifying_upper_lines and not qualifying_lower_lines:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=f"insufficient boundary tests on both sides within lookback (last {lookback_candles_used} candles)",
            lookback_candles_used=lookback_candles_used,
        )
    if not qualifying_upper_lines:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=f"insufficient boundary tests on upper side within lookback (last {lookback_candles_used} candles)",
            lookback_candles_used=lookback_candles_used,
        )
    if not qualifying_lower_lines:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=f"insufficient boundary tests on lower side within lookback (last {lookback_candles_used} candles)",
            lookback_candles_used=lookback_candles_used,
        )

    # Pick the most recent qualifying line on each side (most relevant to
    # "now"), tie-broken by most touches.
    upper_line = max(qualifying_upper_lines, key=lambda l: (l.last.index, len(l.touches)))
    lower_line = max(qualifying_lower_lines, key=lambda l: (l.last.index, len(l.touches)))

    upper_tests = len(upper_line.touches)
    lower_tests = len(lower_line.touches)

    all_boundary_swings = list(upper_line.touches) + list(lower_line.touches)
    genesis_index = min(s.index for s in all_boundary_swings)
    last_touch_index = max(s.index for s in all_boundary_swings)

    upper_at_genesis = upper_line.price_at(genesis_index)
    lower_at_genesis = lower_line.price_at(genesis_index)

    if upper_at_genesis <= lower_at_genesis:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason="no coherent boundary clusters (lines cross at genesis)",
            lookback_candles_used=lookback_candles_used,
        )

    width_absolute = upper_at_genesis - lower_at_genesis
    width_percent = (width_absolute / lower_at_genesis) * 100.0

    required_width_pct = _compute_adaptive_min_width_pct(candles, lookback_start_index)

    if width_percent < required_width_pct:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=(
                f"boundaries not sufficiently separated "
                f"(width={width_percent:.3f}% < required {required_width_pct:.3f}% "
                f"[adaptive, coin's own volatility, clamped {MIN_RANGE_WIDTH_FLOOR_PCT}-{MIN_RANGE_WIDTH_CEILING_PCT}%])"
            ),
            width_percent=width_percent,
            lookback_candles_used=lookback_candles_used,
        )

    # Alternation check — UNCHANGED mechanic, now over each line's
    # member touches instead of a fixed-price cluster's members.
    upper_cluster_pairs = [(s.price, s) for s in upper_line.touches]
    lower_cluster_pairs = [(s.price, s) for s in lower_line.touches]
    fully_alternating, transition_count, touch_sequence = _check_alternation(upper_cluster_pairs, lower_cluster_pairs)

    if not fully_alternating:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=f"boundary touches not alternating (grouped, not oscillating) — sequence: {touch_sequence}",
            boundary_touch_sequence=touch_sequence,
            lookback_candles_used=lookback_candles_used,
        )
    if transition_count < MIN_BOUNDARY_ALTERNATIONS:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=f"insufficient alternating interactions ({transition_count} < {MIN_BOUNDARY_ALTERNATIONS}) — sequence: {touch_sequence}",
            boundary_touch_sequence=touch_sequence,
            lookback_candles_used=lookback_candles_used,
        )

    # Directional-dominance check — UNCHANGED mechanic.
    span_swings = [s for s in swings if genesis_index <= s.index <= last_touch_index]
    span_swings.sort(key=lambda s: s.index)

    if len(span_swings) >= 2:
        total_path = sum(
            abs(span_swings[i].price - span_swings[i - 1].price)
            for i in range(1, len(span_swings))
        )
        net_displacement = abs(span_swings[-1].price - span_swings[0].price)
        dominance_ratio = (net_displacement / total_path) if total_path > 0 else 1.0
    else:
        dominance_ratio = 1.0

    if dominance_ratio >= MAX_DIRECTIONAL_DOMINANCE_RATIO:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=(
                f"movement is a single dominant directional run "
                f"(net/path ratio={dominance_ratio:.2f} >= {MAX_DIRECTIONAL_DOMINANCE_RATIO}), "
                f"not genuine oscillation — sequence: {touch_sequence}"
            ),
            boundary_touch_sequence=touch_sequence,
            directional_dominance_ratio=dominance_ratio,
            lookback_candles_used=lookback_candles_used,
        )

    # --- Shape classification ---
    upper_drift_pct = _line_total_drift_pct(upper_line)
    lower_drift_pct = _line_total_drift_pct(lower_line)
    upper_is_flat = upper_drift_pct <= BOUNDARY_TOLERANCE_PCT
    lower_is_flat = lower_drift_pct <= BOUNDARY_TOLERANCE_PCT

    def _direction(line, is_flat):
        if is_flat:
            return "flat"
        return "rising" if line.slope_per_candle > 0 else "falling"

    upper_dir = _direction(upper_line, upper_is_flat)
    lower_dir = _direction(lower_line, lower_is_flat)

    def _slopes_roughly_parallel(line_a, line_b):
        a, b = line_a.slope_per_candle, line_b.slope_per_candle
        if a == 0 or b == 0:
            return False
        if (a > 0) != (b > 0):
            return False
        larger, smaller = max(abs(a), abs(b)), min(abs(a), abs(b))
        if larger == 0:
            return False
        diff_pct = (1 - smaller / larger) * 100.0
        return diff_pct <= PARALLEL_SLOPE_TOLERANCE_PCT

    upper_at_present_preview = upper_line.price_at(total_candles - 1)
    lower_at_present_preview = lower_line.price_at(total_candles - 1)
    width_at_present_preview = upper_at_present_preview - lower_at_present_preview
    is_narrowing = width_at_present_preview < width_absolute

    shape = None
    if upper_dir == "flat" and lower_dir == "flat":
        shape = "FLAT_RANGE"
    elif upper_dir == "flat" and lower_dir == "rising":
        shape = "ASCENDING_TRIANGLE"
    elif upper_dir == "falling" and lower_dir == "flat":
        shape = "DESCENDING_TRIANGLE"
    elif upper_dir == "rising" and lower_dir == "rising":
        shape = "RISING_CHANNEL" if _slopes_roughly_parallel(upper_line, lower_line) else None
    elif upper_dir == "falling" and lower_dir == "falling":
        shape = "FALLING_CHANNEL" if _slopes_roughly_parallel(upper_line, lower_line) else None
    elif upper_dir == "falling" and lower_dir == "rising":
        shape = "SYMMETRICAL_WEDGE" if is_narrowing else None
    elif upper_dir == "rising" and lower_dir == "falling":
        shape = None  # Type 7 (expanding/diverging) — explicitly excluded
    else:
        # (flat, falling) or (rising, flat) — one-sided WIDENING shapes,
        # not in the approved 6-shape set.
        shape = None

    if shape is None:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=(
                f"unsupported or inconsistent shape (upper={upper_dir}, lower={lower_dir}) — "
                f"either a widening/diverging pattern (excluded) or slopes too dissimilar to be a channel"
            ),
            boundary_touch_sequence=touch_sequence,
            directional_dominance_ratio=dominance_ratio,
            lookback_candles_used=lookback_candles_used,
        )

    # --- Forward breach scan from genesis, against each candle's
    # LINE-implied boundary value at that candle's own index. A
    # converging shape's lines crossing each other (even before the
    # buffer is applied) is ALSO a breach — the range has geometrically
    # run out of room, not just been pierced by a wick. ---
    breach_index = None
    for idx in range(genesis_index, total_candles):
        c = candles[idx]
        upper_val = upper_line.price_at(idx)
        lower_val = lower_line.price_at(idx)
        if upper_val <= lower_val:
            breach_index = idx
            break
        upper_limit = upper_val * (1 + CONTAINMENT_BUFFER_PCT / 100.0)
        lower_limit = lower_val * (1 - CONTAINMENT_BUFFER_PCT / 100.0)
        if c.high > upper_limit or c.low < lower_limit:
            breach_index = idx
            break

    present_index = total_candles - 1
    contained_end_index = breach_index - 1 if breach_index is not None else present_index
    contained_duration = contained_end_index - genesis_index + 1

    if contained_duration < MIN_RANGE_CANDLES:
        return RangeResult(
            detected=False, status="NO_RANGE",
            reason=(
                f"range too short (contained duration {contained_duration} "
                f"< {MIN_RANGE_CANDLES} candles)"
            ),
            boundary_touch_sequence=touch_sequence,
            directional_dominance_ratio=dominance_ratio,
            lookback_candles_used=lookback_candles_used,
        )

    compression = _detect_compression(candles, genesis_index, contained_end_index + 1)
    duration_time_str = f"{contained_duration} x {timeframe} candles"

    # current_price / position are reported relative to each line's value
    # AT THE PRESENT candle — for a sloped boundary this is the trendline's
    # current level, not its old genesis-time level.
    current_price = candles[-1].close
    upper_at_present = upper_line.price_at(present_index)
    lower_at_present = lower_line.price_at(present_index)
    width_at_present = upper_at_present - lower_at_present
    current_width_percent = (width_at_present / lower_at_present * 100.0) if lower_at_present > 0 else None
    current_position_percent = (
        ((current_price - lower_at_present) / width_at_present) * 100.0
        if width_at_present > 0 else None
    )

    # Convergence estimate (only meaningful for converging shapes: the
    # two triangle types and the wedge). Solve upper_line.price_at(idx)
    # == lower_line.price_at(idx) algebraically.
    convergence_candles_ahead = None
    if shape in ("ASCENDING_TRIANGLE", "DESCENDING_TRIANGLE", "SYMMETRICAL_WEDGE"):
        slope_diff = lower_line.slope_per_candle - upper_line.slope_per_candle
        if slope_diff != 0:
            # upper.first.price + upper.slope*(x - upper.first.index) == lower.first.price + lower.slope*(x - lower.first.index)
            # Solve for x:
            numerator = (
                upper_line.first.price - upper_line.slope_per_candle * upper_line.first.index
                - lower_line.first.price + lower_line.slope_per_candle * lower_line.first.index
            )
            x_convergence = numerator / slope_diff
            candles_ahead = x_convergence - present_index
            if candles_ahead > 0:
                convergence_candles_ahead = int(round(candles_ahead))

    base_stats = dict(
        upper_boundary=upper_at_present,
        lower_boundary=lower_at_present,
        width_absolute=width_absolute,
        width_percent=width_percent,
        duration_candles=contained_duration,
        duration_time=duration_time_str,
        upper_tests=upper_tests,
        lower_tests=lower_tests,
        current_price=current_price,
        current_position_percent=current_position_percent,
        compression=compression,
        boundary_touch_sequence=touch_sequence,
        directional_dominance_ratio=dominance_ratio,
        upper_touch_swings=list(upper_line.touches),
        lower_touch_swings=list(lower_line.touches),
        lookback_candles_used=lookback_candles_used,
        shape=shape,
        upper_slope_per_candle=upper_line.slope_per_candle,
        lower_slope_per_candle=lower_line.slope_per_candle,
        current_width_percent=current_width_percent,
        convergence_candles_ahead=convergence_candles_ahead,
    )

    if breach_index is None:
        current_width_str = f"{current_width_percent:.2f}%" if current_width_percent is not None else "n/a"
        reason = (
            f"ACTIVE {shape}: {upper_tests} upper tests, {lower_tests} lower tests, "
            f"width@genesis={width_percent:.2f}%, width@now={current_width_str}, "
            f"duration={contained_duration} candles, "
            f"fully contained through present candle (buffer={CONTAINMENT_BUFFER_PCT}%), "
            f"alternating touches ({touch_sequence}, {transition_count} transitions), "
            f"dominance_ratio={dominance_ratio:.2f}, "
            f"upper_slope={upper_line.slope_per_candle:.6g}/candle, lower_slope={lower_line.slope_per_candle:.6g}/candle, "
            f"compression={compression if compression is not None else 'insufficient data'}"
        )
        return RangeResult(detected=True, status="ACTIVE", reason=reason, **base_stats)

    candles_since_breach = present_index - breach_index

    if candles_since_breach <= RECENT_BREAK_CANDLES:
        reason = (
            f"RECENTLY BROKEN {shape}: contained for {contained_duration} candles "
            f"(genesis->breach), breached {candles_since_breach} candles ago "
            f"(<= {RECENT_BREAK_CANDLES}), width@genesis={width_percent:.2f}%, "
            f"{upper_tests} upper / {lower_tests} lower tests"
        )
        return RangeResult(
            detected=False, status="RECENTLY_BROKEN", reason=reason,
            breach_index=breach_index, candles_since_breach=candles_since_breach,
            **base_stats,
        )
    else:
        reason = (
            f"HISTORICAL/EXPIRED {shape}: contained for {contained_duration} candles "
            f"(genesis->breach), breached {candles_since_breach} candles ago "
            f"(> {RECENT_BREAK_CANDLES}) — price has moved on, width@genesis={width_percent:.2f}%, "
            f"{upper_tests} upper / {lower_tests} lower tests"
        )
        return RangeResult(
            detected=False, status="EXPIRED", reason=reason,
            breach_index=breach_index, candles_since_breach=candles_since_breach,
            **base_stats,
        )


# ============================================================================
# SSOT2 — PHASE 2B: BREAKOUT READINESS (structural tension only)
# ============================================================================
#
# Phase 2B answers exactly one question, symmetrically, for each boundary
# of an ALREADY-CONFIRMED range: who is winning the fight at that
# boundary — buyers or sellers? It does NOT predict which way the range
# will ultimately break, does NOT combine both boundaries into one
# verdict, and is NEVER computed for a coin/timeframe where Phase 2A did
# not confirm an ACTIVE range (there's nothing to evaluate readiness
# against — RECENTLY_BROKEN/EXPIRED ranges are informational only in v2,
# not fed into Phase 2B).
#
# Per-boundary meaning:
#   - Resistance (upper), buyers winning  -> BULLISH BREAKOUT IMMINENT
#   - Resistance (upper), sellers winning -> SELLERS DEFENDING RESISTANCE
#   - Support (lower), buyers winning     -> BUYERS DEFENDING SUPPORT
#   - Support (lower), sellers winning    -> BEARISH BREAKOUT IMMINENT
#   - Neither side clearly ahead          -> CONTESTED
#
# "Winning" is measured purely from each boundary-touch candle's own
# OHLC — no indicators, no volume (deliberately deferred), no lookahead:
# only the exact same confirmed touch candles Phase 2A already used to
# define that boundary.

# A touch's close-position-in-candle is
# (close - low) / (high - low), a value from 0.0 (closed at the low) to
# 1.0 (closed at the high). This single measure is interpreted the SAME
# way at both boundaries: a close near the HIGH of its own candle means
# buyers pushed price back up before the candle closed (buyers won that
# test), regardless of whether the test was at resistance (pushing
# through and holding) or at support (defending, bouncing back up). A
# close near the LOW means sellers won that test, at either boundary.
#
# Average this across all confirmed touches at a boundary:
#   average >= BUYER_CONTROL_THRESHOLD  -> buyers winning that boundary
#   average <= SELLER_CONTROL_THRESHOLD -> sellers winning that boundary
#   otherwise                            -> CONTESTED, no clear control
BUYER_CONTROL_THRESHOLD = 0.60
SELLER_CONTROL_THRESHOLD = 0.40


@dataclass(frozen=True)
class BoundaryControlResult:
    verdict: str  # one of the 5 labels above (or "CONTESTED")
    avg_close_position: float
    touch_count: int
    per_touch_positions: List[float]


@dataclass(frozen=True)
class BreakoutReadinessResult:
    applicable: bool
    reason: str
    upper: Optional[BoundaryControlResult] = None
    lower: Optional[BoundaryControlResult] = None


def _close_position_in_candle(candle: Candle) -> float:
    """(close - low) / (high - low), clamped to [0, 1]. Returns 0.5 for
    a degenerate zero-range candle (high == low) — genuinely ambiguous,
    not a lean toward either side."""
    span = candle.high - candle.low
    if span <= 0:
        return 0.5
    position = (candle.close - candle.low) / span
    return max(0.0, min(1.0, position))


def _evaluate_boundary_control(candles: List[Candle], touch_swings: List[SwingPoint], is_upper: bool) -> BoundaryControlResult:
    positions = [_close_position_in_candle(candles[s.index]) for s in touch_swings]
    avg_position = sum(positions) / len(positions)

    buyers_winning = avg_position >= BUYER_CONTROL_THRESHOLD
    sellers_winning = avg_position <= SELLER_CONTROL_THRESHOLD

    if is_upper:
        if buyers_winning:
            verdict = "BULLISH BREAKOUT IMMINENT"
        elif sellers_winning:
            verdict = "SELLERS DEFENDING RESISTANCE"
        else:
            verdict = "CONTESTED"
    else:
        if buyers_winning:
            verdict = "BUYERS DEFENDING SUPPORT"
        elif sellers_winning:
            verdict = "BEARISH BREAKOUT IMMINENT"
        else:
            verdict = "CONTESTED"

    return BoundaryControlResult(
        verdict=verdict,
        avg_close_position=avg_position,
        touch_count=len(touch_swings),
        per_touch_positions=positions,
    )


def compute_breakout_readiness(candles: List[Candle], range_result: RangeResult) -> BreakoutReadinessResult:
    """
    Only ever computed for an ACTIVE confirmed range (range_result.detected
    == True, status == "ACTIVE"). Reuses Phase 2A's exact touch swings —
    no re-fetching, no re-clustering, no new candle data.
    """
    if not range_result.detected:
        return BreakoutReadinessResult(applicable=False, reason=f"no active confirmed range (status={range_result.status})")

    if not range_result.upper_touch_swings or not range_result.lower_touch_swings:
        return BreakoutReadinessResult(applicable=False, reason="range confirmed but touch data unavailable")

    upper_control = _evaluate_boundary_control(candles, range_result.upper_touch_swings, is_upper=True)
    lower_control = _evaluate_boundary_control(candles, range_result.lower_touch_swings, is_upper=False)

    return BreakoutReadinessResult(
        applicable=True,
        reason="evaluated from confirmed boundary-touch candles",
        upper=upper_control,
        lower=lower_control,
    )


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
    lines.append("")
    lines.append(build_range_report_text(results))
    lines.append("")
    lines.append(build_breakout_readiness_text(results))

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


def build_range_report_text(results: List[MarketStateResult]) -> str:
    """
    Phase 2A Range Detection v2 diagnostic table:
    COIN | TF | MARKET STATE | STATUS | UPPER | LOWER | WIDTH |
    DURATION | UPPER TESTS | LOWER TESTS | POSITION | COMPRESSION | REASON

    STATUS is the 4-way outcome: ACTIVE, RECENTLY_BROKEN, EXPIRED, NO_RANGE.
    Boundary/width/duration/test/position/compression stats are shown for
    ACTIVE, RECENTLY_BROKEN, and EXPIRED alike (a broken/expired range's
    old boundaries and current price position relative to them are still
    informative) — only NO_RANGE shows blank stats.
    """
    header = (
        f"{'COIN':<6} {'TF':<5} {'STATE':<12} {'STATUS':<16} {'UPPER':<12} "
        f"{'LOWER':<12} {'WIDTH%':<8} {'DUR':<10} {'UTEST':<6} {'LTEST':<6} "
        f"{'POS%':<8} {'COMPR':<8} {'SINCE':<7} REASON"
    )
    lines = ["RANGE DETECTION (Phase 2A v2 — bounded lookback)", header, "-" * len(header)]

    for r in results:
        if r.error:
            lines.append(f"{r.symbol:<6} {r.timeframe:<5} NO DATA")
            continue

        rr = r.range_result
        if rr is None:
            lines.append(f"{r.symbol:<6} {r.timeframe:<5} {format_state(r):<12} (not computed)")
            continue

        if rr.status == "NO_RANGE":
            width_str = f"{rr.width_percent:.3f}" if rr.width_percent is not None else "-"
            lines.append(
                f"{r.symbol:<6} {r.timeframe:<5} {format_state(r):<12} {'NO_RANGE':<16} "
                f"{'-':<12} {'-':<12} {width_str:<8} {'-':<10} {'-':<6} {'-':<6} "
                f"{'-':<8} {'-':<8} {'-':<7} {rr.reason}"
            )
            continue

        compr_str = "YES" if rr.compression is True else ("NO" if rr.compression is False else "N/A")
        since_str = str(rr.candles_since_breach) if rr.candles_since_breach is not None else "-"
        pos_str = f"{rr.current_position_percent:.2f}" if rr.current_position_percent is not None else "-"
        lines.append(
            f"{r.symbol:<6} {r.timeframe:<5} {format_state(r):<12} {rr.status:<16} "
            f"{rr.upper_boundary:<12.6f} {rr.lower_boundary:<12.6f} "
            f"{rr.width_percent:<8.2f} {rr.duration_time:<10} "
            f"{rr.upper_tests:<6} {rr.lower_tests:<6} "
            f"{pos_str:<8} {compr_str:<8} {since_str:<7} {rr.reason}"
        )

    return "\n".join(lines)


def build_breakout_readiness_text(results: List[MarketStateResult]) -> str:
    """
    Phase 2B Breakout Readiness diagnostic table. Only ever populated for
    rows where Phase 2A confirmed a range — every other row shows
    N/A (no confirmed range), never a guessed or defaulted verdict.
    Each boundary is reported independently; there is no combined
    "overall" verdict, since the two sides can disagree.
    """
    header = (
        f"{'COIN':<6} {'TF':<5} {'RANGE':<7} "
        f"{'UPPER VERDICT':<28} {'U-POS':<7} {'UTESTS':<7} "
        f"{'LOWER VERDICT':<26} {'L-POS':<7} {'LTESTS':<7}"
    )
    lines = ["BREAKOUT READINESS (Phase 2B) — structural tension only, NOT a directional prediction",
              header, "-" * len(header)]

    for r in results:
        if r.error:
            lines.append(f"{r.symbol:<6} {r.timeframe:<5} NO DATA")
            continue

        br = r.breakout_readiness
        if br is None or not br.applicable:
            reason = br.reason if br is not None else "not computed"
            lines.append(f"{r.symbol:<6} {r.timeframe:<5} {'NO':<7} N/A — {reason}")
            continue

        lines.append(
            f"{r.symbol:<6} {r.timeframe:<5} {'YES':<7} "
            f"{br.upper.verdict:<28} {br.upper.avg_close_position:<7.2f} {br.upper.touch_count:<7} "
            f"{br.lower.verdict:<26} {br.lower.avg_close_position:<7.2f} {br.lower.touch_count:<7}"
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

            # Phase 2A — Range Detection (SSOT2). Reuses SSOT1's own
            # closed candles and confirmed swings; no new fetching, no
            # re-validation, no modification of SSOT1's outputs.
            range_result = detect_range(ohlcv.candles, swings, timeframe)

            # Phase 2B — Breakout Readiness. Only meaningful for an
            # already-confirmed range; compute_breakout_readiness itself
            # returns applicable=False otherwise.
            breakout_readiness = compute_breakout_readiness(ohlcv.candles, range_result)

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
                range_result=range_result,
                breakout_readiness=breakout_readiness,
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
