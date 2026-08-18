"""
Abstract interface every exchange provider must implement.

The router only ever talks to this interface. Nothing about OKX or MEXC
specifics leaks past this boundary.
"""

from abc import ABC, abstractmethod
from typing import List

from models.types import Candle


class ProviderError(Exception):
    """Raised by a provider when it cannot supply valid OHLCV data."""
    pass


class BaseProvider(ABC):
    name: str = "BASE"

    @abstractmethod
    def to_exchange_symbol(self, base_asset: str, quote: str) -> str:
        """Convert a watchlist base asset (e.g. 'BTC') into this exchange's
        symbol format (e.g. 'BTC/USDT')."""
        raise NotImplementedError

    @abstractmethod
    def fetch_ohlcv(self, base_asset: str, timeframe: str, limit: int) -> List[Candle]:
        """Fetch raw candles. Must raise ProviderError on any failure
        (network error, symbol not found, empty response, etc.) rather
        than returning empty/partial data silently."""
        raise NotImplementedError
