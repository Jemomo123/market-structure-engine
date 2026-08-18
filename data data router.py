"""
The ONLY module in this project that knows exchange fallback logic exists.

core/ receives an OHLCVResult and nothing more — it never knows whether
the data came from OKX or MEXC, or that a fallback even happened.
"""

import logging
from typing import List

from config.settings import CANDLE_FETCH_LIMIT, MIN_VALID_CANDLES
from data.base_provider import BaseProvider, ProviderError
from data.okx_provider import OKXProvider
from data.mexc_provider import MEXCProvider
from models.types import Candle, OHLCVResult

logger = logging.getLogger("data_router")


def _validate_candles(candles: List[Candle], timeframe: str) -> bool:
    """Reject anything that isn't real, usable OHLCV data. This is the
    guard against ever silently accepting broken or fake data."""
    if not candles:
        return False
    if len(candles) < MIN_VALID_CANDLES:
        return False

    # Timestamps must be strictly increasing (no dupes, no out-of-order rows).
    timestamps = [c.timestamp for c in candles]
    if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        return False

    # No NaN / non-positive prices, no negative volume.
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
        # Order defines fallback priority: OKX first, MEXC second.
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

            if not _validate_candles(candles, timeframe):
                msg = f"{provider.name} returned invalid/insufficient OHLCV for {base_asset} {timeframe}"
                logger.warning(msg)
                errors.append(msg)
                continue

            return OHLCVResult(
                symbol=base_asset,
                timeframe=timeframe,
                candles=candles,
                source=provider.name,
                is_fallback=is_fallback,
                error=None,
            )

        # Every provider failed. Report this explicitly — never fabricate data.
        return OHLCVResult(
            symbol=base_asset,
            timeframe=timeframe,
            candles=[],
            source=None,
            is_fallback=False,
            error="; ".join(errors) if errors else "No providers available",
        )
