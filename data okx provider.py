"""
OKX spot OHLCV provider.
"""

from typing import List

import ccxt

from data.base_provider import BaseProvider, ProviderError
from models.types import Candle


class OKXProvider(BaseProvider):
    name = "OKX"

    def __init__(self):
        self._client = ccxt.okx({"enableRateLimit": True})

    def to_exchange_symbol(self, base_asset: str, quote: str) -> str:
        return f"{base_asset}/{quote}"

    def fetch_ohlcv(self, base_asset: str, timeframe: str, limit: int) -> List[Candle]:
        symbol = self.to_exchange_symbol(base_asset, "USDT")
        try:
            raw = self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:
            raise ProviderError(f"OKX fetch failed for {symbol} {timeframe}: {exc}") from exc

        if not raw:
            raise ProviderError(f"OKX returned empty OHLCV for {symbol} {timeframe}")

        return [
            Candle(
                timestamp=row[0],
                open=row[1],
                high=row[2],
                low=row[3],
                close=row[4],
                volume=row[5],
            )
            for row in raw
        ]
